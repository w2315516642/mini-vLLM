import os
import unittest

import torch

from minivllm.spec_decode.dspark_cuda import markov_argmax


RUN_CUDA_DSPARK_TESTS = (
    os.environ.get("MINIVLLM_RUN_CUDA_DSPARK_TESTS") == "1"
    and torch.cuda.is_available()
)


@unittest.skipUnless(
    RUN_CUDA_DSPARK_TESTS,
    "set MINIVLLM_RUN_CUDA_DSPARK_TESTS=1 after rebuilding extensions",
)
class MarkovArgmaxCudaTest(unittest.TestCase):

    def test_matches_torch_for_multiple_vocab_tiles(self):
        torch.manual_seed(23)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            base = torch.randn(3, 777, device="cuda", dtype=torch.float32)
            previous = torch.randn(3, 17, device="cuda", dtype=dtype)
            weight = torch.randn(777, 17, device="cuda", dtype=dtype)

            actual = markov_argmax(base, previous, weight)
            expected = torch.argmax(
                base + previous.float() @ weight.float().T,
                dim=-1,
            )

            torch.testing.assert_close(actual, expected)

    def test_ties_choose_first_vocabulary_id(self):
        base = torch.zeros(1, 300, device="cuda", dtype=torch.float32)
        previous = torch.zeros(1, 4, device="cuda", dtype=torch.float16)
        weight = torch.zeros(300, 4, device="cuda", dtype=torch.float16)

        actual = markov_argmax(base, previous, weight)

        self.assertEqual(actual.tolist(), [0])

    def test_tiled_matches_scalar_and_strided_base(self):
        from minivllm import dspark_ops
        torch.manual_seed(87)
        for b, v, rank in [(1, 777, 17), (16, 4097, 128), (19, 1025, 65)]:
            for dtype in (torch.float16, torch.bfloat16):
                base = torch.randn(b, 3, v, device='cuda')[:, 1, :]
                previous = torch.randn(b, rank, device='cuda', dtype=dtype)
                weight = torch.randn(v, rank, device='cuda', dtype=dtype)
                actual = markov_argmax(base, previous, weight)
                expected = dspark_ops.markov_argmax(base.contiguous(), previous, weight)
                torch.testing.assert_close(actual, expected)

    def test_cross_tile_tie_and_graph_replay(self):
        base = torch.zeros(16, 777, device='cuda')
        base[:, 127] = base[:, 600] = 1
        previous = torch.zeros(16, 32, device='cuda', dtype=torch.bfloat16)
        weight = torch.zeros(777, 32, device='cuda', dtype=torch.bfloat16)
        markov_argmax(base, previous, weight)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = markov_argmax(base, previous, weight)
        graph.replay()
        self.assertEqual(output.tolist(), [127] * 16)
        base[:, 600] = 2
        graph.replay()
        self.assertEqual(output.tolist(), [600] * 16)

    def test_ampere_codegen(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from minivllm.spec_decode.markov_triton import _partial
        base = torch.empty(16, 256, device='cuda')
        previous = torch.empty(16, 128, device='cuda', dtype=torch.bfloat16)
        weight = torch.empty(256, 128, device='cuda', dtype=torch.bfloat16)
        scores = torch.empty(16, 2, device='cuda')
        ids = torch.empty(16, 2, device='cuda', dtype=torch.int32)
        compiled = _partial.warmup(base, previous, weight, scores, ids,
            16, 256, 128, 256, 128, 128, 2, 16, 128, 32,
            num_warps=4, grid=(1, 2))
        ampere = triton.compile(compiled.src, target=GPUTarget('cuda', 80, 32),
                               options={'num_warps': 4})
        self.assertIn('.target sm_80', ampere.asm['ptx'])


if __name__ == "__main__":
    unittest.main()
