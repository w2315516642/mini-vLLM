#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

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

void causal_conv1d_varlen_cuda(
  torch::Tensor&, const torch::Tensor&, torch::Tensor&, const torch::Tensor&,
  const torch::Tensor&, const c10::optional<torch::Tensor>&);
void gated_delta_rule_varlen_cuda(
  torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
  const torch::Tensor&, const torch::Tensor&, torch::Tensor&, const torch::Tensor&,
  int64_t, const c10::optional<torch::Tensor>&);

namespace {
void check_varlen_metadata(
  const torch::Tensor& cu_seqlens,
  const c10::optional<torch::Tensor>& lengths,
  int64_t batch, const torch::Tensor& input) {
  // Values come from CPU-validated sequence lengths. Do not read CUDA scalars
  // here: every layer consumes the same immutable packed layout.
  check_cuda_contiguous(cu_seqlens, "cu_seqlens");
  TORCH_CHECK(batch > 0 && cu_seqlens.dim() == 1 && cu_seqlens.size(0) == batch + 1,
              "cu_seqlens must have shape [batch + 1]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.device() == input.device(),
              "cu_seqlens must be int32 on the input device");
  if (lengths) {
    check_cuda_contiguous(*lengths, "lengths");
    TORCH_CHECK(lengths->dim() == 1 && lengths->size(0) == batch &&
                lengths->scalar_type() == at::kInt && lengths->device() == input.device(),
                "lengths must be int32 [batch] on the input device");
  }
}
}

void causal_conv1d_varlen(
  torch::Tensor& output, const torch::Tensor& projected_qkv,
  torch::Tensor& conv_state, const torch::Tensor& weight,
  const torch::Tensor& cu_seqlens, const c10::optional<torch::Tensor>& lengths) {
  for (const auto& tensor : {output, projected_qkv, conv_state, weight})
    check_cuda_contiguous(tensor, "conv tensor");
  check_supported_dtype(projected_qkv, "projected_qkv");
  check_same_device_and_dtype(output, projected_qkv, "output");
  check_same_device_and_dtype(weight, projected_qkv, "weight");
  TORCH_CHECK(projected_qkv.dim() == 2 && conv_state.dim() == 3 && weight.dim() == 2,
              "expected packed [tokens, channels], state [batch, channels, kernel], weight [channels, kernel]");
  TORCH_CHECK(output.sizes() == projected_qkv.sizes() && projected_qkv.size(1) > 0 &&
              conv_state.size(1) == projected_qkv.size(1) && conv_state.size(2) > 0 &&
              weight.size(0) == conv_state.size(1) && weight.size(1) == conv_state.size(2),
              "conv shapes must agree on channels and kernel size");
  TORCH_CHECK(conv_state.scalar_type() == at::kFloat && conv_state.device() == projected_qkv.device(),
              "conv_state must be float32 on the input device");
  check_varlen_metadata(cu_seqlens, lengths, conv_state.size(0), projected_qkv);
  const c10::cuda::CUDAGuard guard(projected_qkv.device());
  causal_conv1d_varlen_cuda(output, projected_qkv, conv_state, weight, cu_seqlens, lengths);
}

void gated_delta_rule_varlen(
  torch::Tensor& output, const torch::Tensor& query, const torch::Tensor& key,
  const torch::Tensor& value, const torch::Tensor& log_decay, const torch::Tensor& beta,
  torch::Tensor& state, const torch::Tensor& cu_seqlens, int64_t max_seqlen,
  const c10::optional<torch::Tensor>& lengths) {
  for (const auto& tensor : {output, query, key, value, log_decay, beta, state})
    check_cuda_contiguous(tensor, "GDN tensor");
  check_supported_dtype(query, "query");
  for (const auto& tensor : {output, key, value, beta})
    check_same_device_and_dtype(tensor, query, "GDN input");
  TORCH_CHECK(query.dim() == 3 && value.dim() == 3 && log_decay.dim() == 2 && state.dim() == 4,
              "expected packed Q/K/V [tokens, heads, dim], decay [tokens, heads], state [batch, heads, Dk, Dv]");
  TORCH_CHECK(query.size(1) > 0 && query.size(2) > 0 && value.size(2) > 0 &&
              key.sizes() == query.sizes() && output.sizes() == value.sizes() &&
              value.size(0) == query.size(0) && value.size(1) == query.size(1) &&
              log_decay.size(0) == query.size(0) && log_decay.size(1) == query.size(1) &&
              beta.sizes() == log_decay.sizes() && state.size(1) == query.size(1) &&
              state.size(2) == query.size(2) && state.size(3) == value.size(2),
              "packed GDN tensor dimensions must agree");
  for (const auto& tensor : {log_decay, state})
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.device() == query.device(),
                "decay and state must be float32 on the input device");
  TORCH_CHECK(max_seqlen >= 0 && max_seqlen <= query.size(0), "invalid max_seqlen");
  check_varlen_metadata(cu_seqlens, lengths, state.size(0), query);
  const c10::cuda::CUDAGuard guard(query.device());
  gated_delta_rule_varlen_cuda(output, query, key, value, log_decay, beta, state,
                              cu_seqlens, max_seqlen, lengths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_conv1d_varlen", &causal_conv1d_varlen,
        "Update packed variable-length convolution sequences.");
  m.def("gated_delta_rule_varlen", &gated_delta_rule_varlen,
        "Update packed variable-length GDN sequences.");
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
