import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.model_executor.parallel_utils.parallel_state import GraphAllReduce
from minivllm.model_executor.parallel_utils.tensor_parallel import layers


RUN_CUDA_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_TP_TESTS") == "1"
    and torch.cuda.is_available()
)


def run_nccl_projection(rank, init_method):
    """Two actual GPU ranks must agree with one unsharded linear projection."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        parallel_state.initialize_model_parallel(2)
        launcher = GraphAllReduce(17, 4, torch.float16)
        weight = torch.arange(32, device="cuda", dtype=torch.float16).view(4, 8) / 32
        layer = layers.RowParallelLinear(
            8, 4, bias=False, input_is_parallel=True, params_dtype=torch.float16,
        )
        with torch.no_grad():
            layer.weight.copy_(weight[:, rank * 4:(rank + 1) * 4])
        pointer = launcher.buffer.data_ptr()
        with patch.object(layers, "get_all_reduce_launcher", return_value=launcher):
            with torch.inference_mode():
                for length in (17, 1, 7, 8, 9, 16, 24, 1):
                    launcher.buffer.fill_(float("nan"))
                    inputs = torch.arange(
                        length * 8, device="cuda", dtype=torch.float16,
                    ).view(length, 8) / 128
                    output, _ = layer(inputs[:, rank * 4:(rank + 1) * 4])
                    expected = torch.nn.functional.linear(inputs, weight)
                    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
                    assert launcher.buffer.data_ptr() == pointer
                    bucket = (length + 7) // 8 * 8
                    torch.testing.assert_close(
                        launcher.buffer[length:bucket],
                        torch.zeros_like(launcher.buffer[length:bucket]),
                    )
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(RUN_CUDA_TESTS, "set MINIVLLM_RUN_CUDA_TP_TESTS=1 with CUDA available")
class GraphAllReduceCudaTest(unittest.TestCase):
    def test_real_graph_replay_with_a_simulated_collective(self):
        # This checks real CUDA capture/storage on one GPU, not NCCL correctness.
        # Multiplication stands in for the sum of two identical rank inputs.
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(dtype=dtype), patch.object(
                parallel_state, "get_tensor_model_parallel_world_size", return_value=2,
            ), patch.object(
                parallel_state, "get_tensor_model_parallel_group", return_value=object(),
            ), patch.object(dist, "all_reduce", side_effect=lambda x, group: x.mul_(2)):
                launcher = GraphAllReduce(17, 4, dtype)
                pointer = launcher.buffer.data_ptr()
                for length in (17, 1, 7, 8, 9, 16, 24, 0):
                    launcher.buffer.fill_(float("nan"))
                    x = launcher.get_buffer(length)
                    x.fill_(3)
                    result = launcher.launch(x)
                    torch.testing.assert_close(result, torch.full_like(x, 6))
                    self.assertEqual(launcher.buffer.data_ptr(), pointer)
                    bucket = (length + 7) // 8 * 8
                    torch.testing.assert_close(
                        launcher.buffer[length:bucket],
                        torch.zeros_like(launcher.buffer[length:bucket]),
                    )
                torch.cuda.synchronize()

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
        "requires two CUDA GPUs and NCCL",
    )
    def test_two_gpu_nccl_projection_matches_unsharded_linear(self):
        with tempfile.TemporaryDirectory() as directory:
            init_method = (Path(directory) / "store").as_uri()
            mp.spawn(run_nccl_projection, args=(init_method,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
