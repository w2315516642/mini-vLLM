#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace vllm {

template<typename scalar_t>
__global__ void causal_conv1d_update_kernel(
  scalar_t* __restrict__ output,              // [batch, channels]
  const scalar_t* __restrict__ projected_qkv, // [batch, channels]
  float* __restrict__ conv_state,             // [batch, channels, kernel_size]
  const scalar_t* __restrict__ weight,         // [channels, kernel_size]
  const int batch_size,
  const int channels,
  const int kernel_size) {
  // Shift one channel state, append the current
  // projected value, accumulate the depthwise convolution in FP32, and write
  // its SiLU activation. One thread should own one (batch, channel) pair.
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int bid = tid / channels;
  const int cid = tid % channels;

  const int stride_b = channels * kernel_size;
  const int stride_c = kernel_size;

  const int num_channels = batch_size * channels;
  const int chn_ptr = bid * stride_b + cid * stride_c;

  if (tid >= num_channels) return;

  float sum = 0.0f;
  for (int i = 0; i < kernel_size - 1; i++) {
    float state = conv_state[chn_ptr + i + 1];
    conv_state[chn_ptr + i] = state;

    float weight_val = static_cast<float>(weight[cid * kernel_size + i]);
    sum += state * weight_val;
  }

  float state = static_cast<float>(projected_qkv[bid * channels + cid]);
  conv_state[chn_ptr + kernel_size - 1] = state;
  float weight_val = static_cast<float>(weight[cid * kernel_size + kernel_size - 1]);
  sum += state * weight_val;

  float out = sum / (1.0f + expf(-sum));

  output[bid * channels + cid] = static_cast<scalar_t>(out);
}

template<typename scalar_t>
__global__ void gated_delta_rule_decode_kernel(
  scalar_t* __restrict__ output,              // [batch, heads, value_dim]
  const scalar_t* __restrict__ query,         // [batch, heads, key_dim]
  const scalar_t* __restrict__ key,           // [batch, heads, key_dim]
  const scalar_t* __restrict__ value,         // [batch, heads, value_dim]
  const float* __restrict__ log_decay,        // [batch, heads]
  const scalar_t* __restrict__ beta,          // [batch, heads]
  float* __restrict__ recurrent_state,        // [batch, heads, key_dim, value_dim]
  const int batch_size,
  const int num_heads,
  const int key_dim,
  const int value_dim) {
  // Let one block own a flattened (batch, head) and
  // let each thread own one or more value dimensions. Apply decay, read the
  // old value, form the delta, update the FP32 state, then read the output
  // from the updated state.
  const int bid = blockIdx.x / num_heads;
  const int hid = blockIdx.x % num_heads;

  const int stride_b = num_heads * key_dim * value_dim;
  const int stride_h = key_dim * value_dim;
  const int head_ptr = bid * stride_b + hid * stride_h;

  const int key_ptr = (bid * num_heads + hid) * key_dim;
  const int value_ptr = (bid * num_heads + hid) * value_dim;

  extern __shared__ float shmem[];
  float* q_shared = shmem;
  float* k_shared = &shmem[key_dim];

  // load qk
  for (int k = threadIdx.x; k < key_dim; k += blockDim.x) {
    q_shared[k] = static_cast<float>(query[key_ptr + k]);
    k_shared[k] = static_cast<float>(key[key_ptr + k]);
  }
  __syncthreads();

  float decay = expf(log_decay[bid * num_heads + hid]);
  float v_beta = static_cast<float>(beta[bid * num_heads + hid]);

  for (int v = threadIdx.x; v < value_dim; v += blockDim.x) {
    // old_value: gemv
    float old_value = 0.0f;
    for (int k = 0; k < key_dim; k++) {
      float state_val = recurrent_state[head_ptr + k * value_dim + v];
      state_val *= decay;

      recurrent_state[head_ptr + k * value_dim + v] = state_val;
      float k_val = k_shared[k];
      float k_state = state_val;
      old_value += k_val * k_state;
    }
    // 算 delta
    float v_val = value[value_ptr + v];
    float delta = v_val - old_value;
    delta *= v_beta;
    // 写入 + 读取
    float dot = 0.0f;
    for (int k = 0; k < key_dim; k++) {
      float q_val = q_shared[k];
      float k_val = k_shared[k];

      float state_val = recurrent_state[head_ptr + k * value_dim + v];
      state_val += k_val * delta;
      recurrent_state[head_ptr + k * value_dim + v] = state_val;

      dot += q_val * state_val;
    }
    output[value_ptr + v] = static_cast<scalar_t>(dot);
  }
}

template<typename scalar_t>
__global__ void gated_delta_rule_prefill_chunk_kernel(
  scalar_t* __restrict__ output,              // [batch, sequence, heads, value_dim]
  const scalar_t* __restrict__ query,         // [batch, sequence, heads, key_dim]
  const scalar_t* __restrict__ key,           // [batch, sequence, heads, key_dim]
  const scalar_t* __restrict__ value,         // [batch, sequence, heads, value_dim]
  const float* __restrict__ log_decay,        // [batch, sequence, heads]
  const scalar_t* __restrict__ beta,          // [batch, sequence, heads]
  float* __restrict__ recurrent_state,        // [batch, heads, key_dim, value_dim]
  const int batch_size,
  const int sequence_length,
  const int num_heads,
  const int key_dim,
  const int value_dim,
  const int chunk_start,
  const int chunk_end) {
  // Reuse the decode recurrence for every token in
  // [chunk_start, chunk_end). State is shared between successive tokens and
  // must remain live for the next chunk launch on the same CUDA stream.
  const int bid = blockIdx.x / num_heads;
  const int hid = blockIdx.x % num_heads;

  extern __shared__ float shmem[];
  float* q_shared = shmem;
  float* k_shared = &shmem[key_dim];

  const int stride_b = num_heads * key_dim * value_dim;
  const int stride_h = key_dim * value_dim;
  const int head_ptr = bid * stride_b + hid * stride_h;
  for (int token = chunk_start; token < chunk_end; token++) {
    const int head_idx = bid * num_heads * sequence_length + token * num_heads + hid;

    const int key_ptr = head_idx * key_dim;
    const int value_ptr = head_idx * value_dim;

    // load qk
    for (int k = threadIdx.x; k < key_dim; k += blockDim.x) {
      q_shared[k] = static_cast<float>(query[key_ptr + k]);
      k_shared[k] = static_cast<float>(key[key_ptr + k]);
    }
    __syncthreads();

    float decay = expf(log_decay[head_idx]);
    float v_beta = static_cast<float>(beta[head_idx]);

    for (int v = threadIdx.x; v < value_dim; v += blockDim.x) {
      // old_value: gemv
      float old_value = 0.0f;
      for (int k = 0; k < key_dim; k++) {
        float state_val = recurrent_state[head_ptr + k * value_dim + v];
        state_val *= decay;

        recurrent_state[head_ptr + k * value_dim + v] = state_val;
        float k_val = k_shared[k];
        float k_state = state_val;
        old_value += k_val * k_state;
      }
      // 算 delta
      float v_val = value[value_ptr + v];
      float delta = v_val - old_value;
      delta *= v_beta;
      // 写入 + 读取
      float dot = 0.0f;
      for (int k = 0; k < key_dim; k++) {
        float q_val = q_shared[k];
        float k_val = k_shared[k];

        float state_val = recurrent_state[head_ptr + k * value_dim + v];
        state_val += k_val * delta;
        recurrent_state[head_ptr + k * value_dim + v] = state_val;

        dot += q_val * state_val;
      }
      output[value_ptr + v] = static_cast<scalar_t>(dot);
    }
    __syncthreads();
  }
}

}  // namespace vllm

void causal_conv1d_update_cuda(
  torch::Tensor& output,
  const torch::Tensor& projected_qkv,
  torch::Tensor& conv_state,
  const torch::Tensor& weight) {
  // Dispatch float/half/bfloat16 and launch
  // causal_conv1d_update_kernel on PyTorch's current CUDA stream.
  int batch_size = conv_state.size(0);
  int channels = conv_state.size(1);
  int kernel_size = conv_state.size(2);

  int num_channels = batch_size * channels;

  int threads = std::min(channels, 1024);
  int blocks = (num_channels + threads - 1) / threads;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    projected_qkv.scalar_type(),
    "causal_conv1d_update_kernel",
    [&] {
      vllm::causal_conv1d_update_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        projected_qkv.data_ptr<scalar_t>(),
        conv_state.data_ptr<float>(),
        weight.data_ptr<scalar_t>(),
        batch_size,
        channels,
        kernel_size);
    });
}

void gated_delta_rule_decode_cuda(
  torch::Tensor& output,
  const torch::Tensor& query,
  const torch::Tensor& key,
  const torch::Tensor& value,
  const torch::Tensor& log_decay,
  const torch::Tensor& beta,
  torch::Tensor& recurrent_state) {
  // Flatten (batch, head) into grid.x, dispatch the
  // input dtype, and launch on PyTorch's current CUDA stream.
  int batch_size = recurrent_state.size(0);
  int num_heads = recurrent_state.size(1);
  int key_dim = recurrent_state.size(2);
  int value_dim = recurrent_state.size(3);

  const size_t shmem_size = 2 * key_dim * sizeof(float);

  int threads = std::min(value_dim, 1024);
  int blocks = batch_size * num_heads;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    query.scalar_type(),
    "gated_delta_rule_decode_kernel",
    [&] {
      vllm::gated_delta_rule_decode_kernel<scalar_t><<<blocks, threads, shmem_size, stream>>>(
        output.data_ptr<scalar_t>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        log_decay.data_ptr<float>(),
        beta.data_ptr<scalar_t>(),
        recurrent_state.data_ptr<float>(),
        batch_size,
        num_heads,
        key_dim,
        value_dim);
    });
}

void gated_delta_rule_prefill_cuda(
  torch::Tensor& output,
  const torch::Tensor& query,
  const torch::Tensor& key,
  const torch::Tensor& value,
  const torch::Tensor& log_decay,
  const torch::Tensor& beta,
  torch::Tensor& recurrent_state,
  int64_t chunk_size) {
  // Dispatch once, then launch one chunk kernel for
  // each [chunk_start, chunk_end) on the same current CUDA stream. The final
  // chunk may contain fewer than chunk_size tokens.
  int batch_size = recurrent_state.size(0);
  int num_heads = recurrent_state.size(1);
  int key_dim = recurrent_state.size(2);
  int value_dim = recurrent_state.size(3);
  int sequence_length = query.size(1);

  const size_t shmem_size = 2 * key_dim * sizeof(float);

  int threads = std::min(value_dim, 1024);
  int blocks = batch_size * num_heads;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int chunk_step = static_cast<int>(std::min<int64_t>(chunk_size, sequence_length));
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    query.scalar_type(),
    "gated_delta_rule_prefill_chunk_kernel",
    [&] {
      for (int chunk = 0; chunk < sequence_length; chunk += chunk_step) {
        int chunk_start = chunk;
        int chunk_end = std::min(chunk + chunk_step, sequence_length);
        vllm::gated_delta_rule_prefill_chunk_kernel<scalar_t><<<blocks, threads, shmem_size, stream>>>(
          output.data_ptr<scalar_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          value.data_ptr<scalar_t>(),
          log_decay.data_ptr<float>(),
          beta.data_ptr<scalar_t>(),
          recurrent_state.data_ptr<float>(),
          batch_size,
          sequence_length,
          num_heads,
          key_dim,
          value_dim,
          chunk_start,
          chunk_end);
      }
    });
}
