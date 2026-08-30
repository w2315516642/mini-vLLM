#include <torch/extension.h>

namespace {

void check_cuda_contiguous(
  const torch::Tensor& tensor,
  const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_supported_dtype(
  const torch::Tensor& tensor,
  const char* name) {
  const auto dtype = tensor.scalar_type();
  TORCH_CHECK(
    dtype == at::ScalarType::Float ||
      dtype == at::ScalarType::Half ||
      dtype == at::ScalarType::BFloat16,
    name,
    " must use float32, float16, or bfloat16");
}

void check_same_device_and_dtype(
  const torch::Tensor& tensor,
  const torch::Tensor& reference,
  const char* name) {
  TORCH_CHECK(
    tensor.device() == reference.device(),
    name,
    " must be on the same device as the input");
  TORCH_CHECK(
    tensor.scalar_type() == reference.scalar_type(),
    name,
    " must have the same dtype as the input");
}

}  // namespace

void causal_conv1d_update_cuda(
  torch::Tensor& output,
  const torch::Tensor& projected_qkv,
  torch::Tensor& conv_state,
  const torch::Tensor& weight);

void gated_delta_rule_decode_cuda(
  torch::Tensor& output,
  const torch::Tensor& query,
  const torch::Tensor& key,
  const torch::Tensor& value,
  const torch::Tensor& log_decay,
  const torch::Tensor& beta,
  torch::Tensor& recurrent_state);

void gated_delta_rule_prefill_cuda(
  torch::Tensor& output,
  const torch::Tensor& query,
  const torch::Tensor& key,
  const torch::Tensor& value,
  const torch::Tensor& log_decay,
  const torch::Tensor& beta,
  torch::Tensor& recurrent_state,
  int64_t chunk_size);

void causal_conv1d_update(
  torch::Tensor& output,
  const torch::Tensor& projected_qkv,
  torch::Tensor& conv_state,
  const torch::Tensor& weight) {
  check_cuda_contiguous(output, "output");
  check_cuda_contiguous(projected_qkv, "projected_qkv");
  check_cuda_contiguous(conv_state, "conv_state");
  check_cuda_contiguous(weight, "weight");
  check_supported_dtype(projected_qkv, "projected_qkv");
  check_same_device_and_dtype(output, projected_qkv, "output");
  check_same_device_and_dtype(weight, projected_qkv, "weight");
  TORCH_CHECK(
    conv_state.device() == projected_qkv.device(),
    "conv_state must be on the same device as projected_qkv");
  TORCH_CHECK(
    conv_state.scalar_type() == at::ScalarType::Float,
    "conv_state must use float32");

  TORCH_CHECK(
    projected_qkv.dim() == 2,
    "projected_qkv must have shape [batch, channels]");
  TORCH_CHECK(
    conv_state.dim() == 3,
    "conv_state must have shape [batch, channels, kernel_size]");
  TORCH_CHECK(
    weight.dim() == 2,
    "weight must have shape [channels, kernel_size]");
  TORCH_CHECK(
    output.sizes() == projected_qkv.sizes(),
    "output must have the same shape as projected_qkv");

  const auto batch_size = projected_qkv.size(0);
  const auto channels = projected_qkv.size(1);
  TORCH_CHECK(batch_size > 0 && channels > 0, "input dimensions must be positive");
  TORCH_CHECK(
    conv_state.size(0) == batch_size && conv_state.size(1) == channels,
    "conv_state batch and channel dimensions must match projected_qkv");
  TORCH_CHECK(
    weight.size(0) == channels &&
      weight.size(1) == conv_state.size(2),
    "weight must match conv_state channels and kernel_size");
  TORCH_CHECK(conv_state.size(2) > 0, "kernel_size must be positive");

  causal_conv1d_update_cuda(output, projected_qkv, conv_state, weight);
}

void gated_delta_rule_decode(
  torch::Tensor& output,
  const torch::Tensor& query,
  const torch::Tensor& key,
  const torch::Tensor& value,
  const torch::Tensor& log_decay,
  const torch::Tensor& beta,
  torch::Tensor& recurrent_state) {
  check_cuda_contiguous(output, "output");
  check_cuda_contiguous(query, "query");
  check_cuda_contiguous(key, "key");
  check_cuda_contiguous(value, "value");
  check_cuda_contiguous(log_decay, "log_decay");
  check_cuda_contiguous(beta, "beta");
  check_cuda_contiguous(recurrent_state, "recurrent_state");
  check_supported_dtype(query, "query");

  check_same_device_and_dtype(output, query, "output");
  check_same_device_and_dtype(key, query, "key");
  check_same_device_and_dtype(value, query, "value");
  check_same_device_and_dtype(beta, query, "beta");
  TORCH_CHECK(
    log_decay.device() == query.device(),
    "log_decay must be on the same device as query");
  TORCH_CHECK(
    log_decay.scalar_type() == at::ScalarType::Float,
    "log_decay must use float32");
  TORCH_CHECK(
    recurrent_state.device() == query.device(),
    "recurrent_state must be on the same device as query");
  TORCH_CHECK(
    recurrent_state.scalar_type() == at::ScalarType::Float,
    "recurrent_state must use float32");

  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, key_dim]");
  TORCH_CHECK(key.sizes() == query.sizes(), "key must have the same shape as query");
  TORCH_CHECK(value.dim() == 3, "value must have shape [batch, heads, value_dim]");
  TORCH_CHECK(log_decay.dim() == 2, "log_decay must have shape [batch, heads]");
  TORCH_CHECK(beta.sizes() == log_decay.sizes(), "beta must match log_decay shape");
  TORCH_CHECK(
    recurrent_state.dim() == 4,
    "recurrent_state must have shape [batch, heads, key_dim, value_dim]");

  const auto batch_size = query.size(0);
  const auto num_heads = query.size(1);
  const auto key_dim = query.size(2);
  TORCH_CHECK(
    batch_size > 0 && num_heads > 0 && key_dim > 0,
    "query dimensions must be positive");
  TORCH_CHECK(
    value.size(0) == batch_size && value.size(1) == num_heads &&
      value.size(2) > 0,
    "value batch and head dimensions must match query");
  TORCH_CHECK(
    log_decay.size(0) == batch_size && log_decay.size(1) == num_heads,
    "log_decay must match query batch and head dimensions");
  TORCH_CHECK(
    recurrent_state.size(0) == batch_size &&
      recurrent_state.size(1) == num_heads &&
      recurrent_state.size(2) == key_dim &&
      recurrent_state.size(3) == value.size(2),
    "recurrent_state dimensions must match query and value");
  TORCH_CHECK(output.sizes() == value.sizes(), "output must have the same shape as value");

  gated_delta_rule_decode_cuda(
    output,
    query,
    key,
    value,
    log_decay,
    beta,
    recurrent_state);
}

void gated_delta_rule_prefill(
  torch::Tensor& output,
  const torch::Tensor& query,
  const torch::Tensor& key,
  const torch::Tensor& value,
  const torch::Tensor& log_decay,
  const torch::Tensor& beta,
  torch::Tensor& recurrent_state,
  int64_t chunk_size) {
  check_cuda_contiguous(output, "output");
  check_cuda_contiguous(query, "query");
  check_cuda_contiguous(key, "key");
  check_cuda_contiguous(value, "value");
  check_cuda_contiguous(log_decay, "log_decay");
  check_cuda_contiguous(beta, "beta");
  check_cuda_contiguous(recurrent_state, "recurrent_state");
  check_supported_dtype(query, "query");

  check_same_device_and_dtype(output, query, "output");
  check_same_device_and_dtype(key, query, "key");
  check_same_device_and_dtype(value, query, "value");
  check_same_device_and_dtype(beta, query, "beta");
  TORCH_CHECK(
    log_decay.device() == query.device(),
    "log_decay must be on the same device as query");
  TORCH_CHECK(
    log_decay.scalar_type() == at::ScalarType::Float,
    "log_decay must use float32");
  TORCH_CHECK(
    recurrent_state.device() == query.device(),
    "recurrent_state must be on the same device as query");
  TORCH_CHECK(
    recurrent_state.scalar_type() == at::ScalarType::Float,
    "recurrent_state must use float32");

  TORCH_CHECK(query.dim() == 4, "query must have shape [batch, sequence, heads, key_dim]");
  TORCH_CHECK(key.sizes() == query.sizes(), "key must have the same shape as query");
  TORCH_CHECK(value.dim() == 4, "value must have shape [batch, sequence, heads, value_dim]");
  TORCH_CHECK(log_decay.dim() == 3, "log_decay must have shape [batch, sequence, heads]");
  TORCH_CHECK(beta.sizes() == log_decay.sizes(), "beta must match log_decay shape");
  TORCH_CHECK(
    recurrent_state.dim() == 4,
    "recurrent_state must have shape [batch, heads, key_dim, value_dim]");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

  const auto batch_size = query.size(0);
  const auto sequence_length = query.size(1);
  const auto num_heads = query.size(2);
  const auto key_dim = query.size(3);
  TORCH_CHECK(
    batch_size > 0 && sequence_length > 0 && num_heads > 0 && key_dim > 0,
    "query dimensions must be positive");
  TORCH_CHECK(
    value.size(0) == batch_size && value.size(1) == sequence_length &&
      value.size(2) == num_heads && value.size(3) > 0,
    "value batch, sequence, and head dimensions must match query");
  TORCH_CHECK(
    log_decay.size(0) == batch_size &&
      log_decay.size(1) == sequence_length &&
      log_decay.size(2) == num_heads,
    "log_decay must match query batch, sequence, and head dimensions");
  TORCH_CHECK(
    recurrent_state.size(0) == batch_size &&
      recurrent_state.size(1) == num_heads &&
      recurrent_state.size(2) == key_dim &&
      recurrent_state.size(3) == value.size(3),
    "recurrent_state dimensions must match query and value");
  TORCH_CHECK(output.sizes() == value.sizes(), "output must have the same shape as value");

  gated_delta_rule_prefill_cuda(
    output,
    query,
    key,
    value,
    log_decay,
    beta,
    recurrent_state,
    chunk_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "causal_conv1d_update",
    &causal_conv1d_update,
    "Update the causal convolution state for one token.");
  m.def(
    "gated_delta_rule_decode",
    &gated_delta_rule_decode,
    "Apply one recurrent Gated Delta Rule step.");
  m.def(
    "gated_delta_rule_prefill",
    &gated_delta_rule_prefill,
    "Apply chunked Gated Delta Rule prefill.");
}
