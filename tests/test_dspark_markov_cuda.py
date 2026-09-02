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


if __name__ == "__main__":
    unittest.main()
