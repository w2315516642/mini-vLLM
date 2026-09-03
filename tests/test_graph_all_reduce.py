import unittest
from unittest.mock import Mock, patch

import torch

from minivllm.model_executor.parallel_utils import parallel_state
from minivllm.model_executor.parallel_utils.parallel_state import GraphAllReduce
from minivllm.model_executor.parallel_utils.tensor_parallel import layers


def make_launcher(max_num_tokens=17, disable_graph=False):
    """Exercise real buffer/dispatch code with a CPU stand-in for the collective."""
    allocate = torch.empty

    def cpu_buffer(*, size, dtype, device):
        return allocate(size, dtype=dtype, device="cpu")

    def build_graph(launcher, num_tokens):
        return Mock(replay=Mock(
            side_effect=lambda: launcher.buffer[:num_tokens].mul_(2),
        ))

    with patch.object(
        parallel_state, "get_tensor_model_parallel_world_size", return_value=2,
    ), patch.object(
        parallel_state, "get_tensor_model_parallel_group", return_value=object(),
    ), patch.object(
        parallel_state.torch, "empty", side_effect=cpu_buffer,
    ), patch.object(GraphAllReduce, "_build_graph", autospec=True, side_effect=build_graph):
        return GraphAllReduce(max_num_tokens, 4, torch.float32, disable_graph)


class GraphAllReduceTest(unittest.TestCase):
    def test_seven_tokens_use_the_eight_token_graph(self):
        launcher = make_launcher()
        launcher.buffer.fill_(99)
        x = launcher.buffer[:7]
        x.fill_(3)

        result = launcher.launch(x)

        self.assertIs(result, x)
        torch.testing.assert_close(result, torch.full((7, 4), 6.0))
        torch.testing.assert_close(launcher.buffer[7:8], torch.zeros(1, 4))
        launcher.graphs[8].replay.assert_called_once()

    def test_capacity_and_graphs_include_the_last_aligned_bucket(self):
        for maximum, capacity in ((1, 8), (7, 8), (8, 8), (9, 16), (17, 24)):
            with self.subTest(maximum=maximum):
                launcher = make_launcher(maximum)
                self.assertEqual(launcher.buffer.shape, (capacity, 4))
                self.assertEqual(list(launcher.graphs), list(range(8, capacity + 1, 8)))

    def test_varying_lengths_reuse_storage_and_clear_only_padding(self):
        launcher = make_launcher()
        pointer = launcher.buffer.data_ptr()
        for length in (17, 1, 7, 8, 9, 16, 24, 1):
            with self.subTest(length=length):
                launcher.buffer.fill_(99)
                x = launcher.get_buffer(length)
                x.fill_(3)
                result = launcher.launch(x)
                bucket = (length + 7) // 8 * 8
                self.assertEqual(launcher.buffer.data_ptr(), pointer)
                self.assertEqual(result.shape, (length, 4))
                torch.testing.assert_close(result, torch.full_like(result, 6))
                torch.testing.assert_close(
                    launcher.buffer[length:bucket],
                    torch.zeros_like(launcher.buffer[length:bucket]),
                )
                torch.testing.assert_close(
                    launcher.buffer[bucket:],
                    torch.full_like(launcher.buffer[bucket:], 99),
                )

    def test_empty_input_does_not_replay_a_graph(self):
        launcher = make_launcher()
        x = launcher.get_buffer(0)
        self.assertIs(launcher.launch(x), x)
        for graph in launcher.graphs.values():
            graph.replay.assert_not_called()

    def test_disabled_graph_reduces_only_valid_rows(self):
        launcher = make_launcher(disable_graph=True)
        launcher.buffer.fill_(99)
        x = launcher.get_buffer(7)
        x.fill_(3)
        with patch.object(
            torch.distributed, "all_reduce", side_effect=lambda x, group: x.mul_(2),
        ) as reduce:
            self.assertIs(launcher.launch(x), x)
        self.assertIs(reduce.call_args.args[0], x)
        torch.testing.assert_close(x, torch.full_like(x, 6))
        torch.testing.assert_close(launcher.buffer[7:], torch.full_like(launcher.buffer[7:], 99))

    def test_buffer_capacity_is_checked_before_projection_can_resize_it(self):
        launcher = make_launcher()
        for length in (-1, 25):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "capacity"):
                    launcher.get_buffer(length)

    def test_replay_requires_the_captured_buffer_prefix(self):
        launcher = make_launcher()
        for x in (
            torch.empty(7, 4),
            launcher.buffer[1:8],
            launcher.buffer[:7, :2],
            launcher.buffer[:8:2],
        ):
            with self.subTest(shape=x.shape, stride=x.stride()):
                with self.assertRaisesRegex(ValueError, "buffer prefix"):
                    launcher.launch(x)

    def test_row_parallel_projection_accepts_unpadded_tokens(self):
        launcher = make_launcher()
        with patch.object(layers, "get_tensor_model_parallel_world_size", return_value=2):
            layer = layers.RowParallelLinear(
                8, 4, bias=False, input_is_parallel=True,
                use_cpu_initialization=True, params_dtype=torch.float32,
            )
        with torch.no_grad():
            layer.weight.fill_(0.5)
        inputs = torch.arange(28, dtype=torch.float32).reshape(7, 4)
        with patch.object(
            layers, "get_tensor_model_parallel_world_size", return_value=2,
        ), patch.object(layers, "get_all_reduce_launcher", return_value=launcher):
            with torch.inference_mode():
                output, bias = layer(inputs)
        self.assertIsNone(bias)
        torch.testing.assert_close(output, 2 * torch.nn.functional.linear(inputs, layer.weight))


if __name__ == "__main__":
    unittest.main()
