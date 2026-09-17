"""Offline smoke-checkpoint conversion, independent of CUDA inference."""
import unittest

import torch

from scripts.qwen_fp8_quantize import quantize_block_weight


class FP8OfflineQuantizeTest(unittest.TestCase):
    def test_zero_blocks_have_positive_scale_and_zero_weight(self):
        weight, scale = quantize_block_weight(torch.zeros(3, 5), block_size=2)
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (2, 3))
        torch.testing.assert_close(scale, torch.ones(2, 3))
        torch.testing.assert_close(weight.float(), torch.zeros(3, 5))

    def test_tail_blocks_and_source_unchanged(self):
        source = torch.arange(-7, 8, dtype=torch.float32).reshape(3, 5)
        before = source.clone()
        weight, scale = quantize_block_weight(source, block_size=2)
        for row in range(3):
            for col in range(5):
                r, c = row//2, col//2
                expected_scale = source[r*2:r*2+2, c*2:c*2+2].abs().max() / 448
                self.assertEqual(scale[r, c], expected_scale)
                error = (weight[row, col].float() * scale[r, c] - source[row, col]).abs()
                self.assertLessEqual(error, 16.1 * scale[r, c])
        torch.testing.assert_close(source, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
