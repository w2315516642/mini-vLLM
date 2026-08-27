#include <torch/extension.h>

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void sigmoid_and_mul(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& gate);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  m.def(
    "sigmoid_and_mul",
    &sigmoid_and_mul,
    "Apply the sigmoid output gate used by Qwen full attention.");
}
