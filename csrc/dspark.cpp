#include <torch/extension.h>

torch::Tensor markov_argmax_cuda(
  const torch::Tensor& base_logits,
  const torch::Tensor& previous_embeddings,
  const torch::Tensor& projection_weight);

torch::Tensor markov_argmax(
  const torch::Tensor& base_logits,
  const torch::Tensor& previous_embeddings,
  const torch::Tensor& projection_weight) {
  TORCH_CHECK(base_logits.is_cuda(), "base_logits must be a CUDA tensor");
  TORCH_CHECK(previous_embeddings.is_cuda(), "previous_embeddings must be CUDA");
  TORCH_CHECK(projection_weight.is_cuda(), "projection_weight must be CUDA");
  TORCH_CHECK(base_logits.is_contiguous(), "base_logits must be contiguous");
  TORCH_CHECK(
    previous_embeddings.is_contiguous(),
    "previous_embeddings must be contiguous");
  TORCH_CHECK(
    projection_weight.is_contiguous(),
    "projection_weight must be contiguous");
  TORCH_CHECK(
    base_logits.scalar_type() == at::ScalarType::Float,
    "base_logits must use float32");
  TORCH_CHECK(
    previous_embeddings.scalar_type() == projection_weight.scalar_type(),
    "Markov embedding and projection weight must use the same dtype");
  const auto markov_dtype = previous_embeddings.scalar_type();
  TORCH_CHECK(
    markov_dtype == at::ScalarType::Float ||
      markov_dtype == at::ScalarType::Half ||
      markov_dtype == at::ScalarType::BFloat16,
    "Markov tensors must use float32, float16, or bfloat16");
  TORCH_CHECK(
    base_logits.device() == previous_embeddings.device() &&
      base_logits.device() == projection_weight.device(),
    "All Markov tensors must be on the same device");
  TORCH_CHECK(
    base_logits.dim() == 2,
    "base_logits must have shape [batch, vocab]");
  TORCH_CHECK(
    previous_embeddings.dim() == 2,
    "previous_embeddings must have shape [batch, rank]");
  TORCH_CHECK(
    projection_weight.dim() == 2,
    "projection_weight must have shape [vocab, rank]");
  TORCH_CHECK(
    base_logits.size(0) > 0 && base_logits.size(1) > 0,
    "Markov batch and vocabulary dimensions must be positive");
  TORCH_CHECK(
    previous_embeddings.size(0) == base_logits.size(0),
    "Markov batch dimensions must match");
  TORCH_CHECK(
    projection_weight.size(0) == base_logits.size(1) &&
      projection_weight.size(1) == previous_embeddings.size(1),
    "Markov projection dimensions must match logits and embeddings");
  TORCH_CHECK(previous_embeddings.size(1) > 0, "Markov rank must be positive");
  return markov_argmax_cuda(
    base_logits, previous_embeddings, projection_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "markov_argmax",
    &markov_argmax,
    "Apply a low-rank Markov bias and return the row-wise argmax.");
}
