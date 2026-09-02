#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

__device__ bool is_better(
  float candidate_score,
  int64_t candidate_id,
  float current_score,
  int64_t current_id) {
  return candidate_score > current_score ||
    (candidate_score == current_score && candidate_id < current_id);
}

template<typename scalar_t>
__global__ void markov_partial_argmax_kernel(
  const float* __restrict__ base_logits,
  const scalar_t* __restrict__ previous_embeddings,
  const scalar_t* __restrict__ projection_weight,
  float* __restrict__ partial_scores,
  int64_t* __restrict__ partial_ids,
  int vocab_size,
  int rank,
  int num_tiles) {
  const int batch_id = blockIdx.y;
  const int token_id = blockIdx.x * blockDim.x + threadIdx.x;
  float score = -std::numeric_limits<float>::infinity();
  int64_t selected_id = vocab_size;
  if (token_id < vocab_size) {
    score = base_logits[batch_id * vocab_size + token_id];
    const int embedding_offset = batch_id * rank;
    const int weight_offset = token_id * rank;
    for (int index = 0; index < rank; ++index) {
      score += static_cast<float>(previous_embeddings[embedding_offset + index])
        * static_cast<float>(projection_weight[weight_offset + index]);
    }
    selected_id = token_id;
  }

  __shared__ float scores[kThreads];
  __shared__ int64_t ids[kThreads];
  scores[threadIdx.x] = score;
  ids[threadIdx.x] = selected_id;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset && is_better(
        scores[threadIdx.x + offset],
        ids[threadIdx.x + offset],
        scores[threadIdx.x],
        ids[threadIdx.x])) {
      scores[threadIdx.x] = scores[threadIdx.x + offset];
      ids[threadIdx.x] = ids[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int output_index = batch_id * num_tiles + blockIdx.x;
    partial_scores[output_index] = scores[0];
    partial_ids[output_index] = ids[0];
  }
}

__global__ void reduce_partial_argmax_kernel(
  const float* __restrict__ partial_scores,
  const int64_t* __restrict__ partial_ids,
  int64_t* __restrict__ output,
  int num_tiles) {
  const int batch_id = blockIdx.x;
  float score = -std::numeric_limits<float>::infinity();
  int64_t selected_id = std::numeric_limits<int64_t>::max();
  for (int tile = threadIdx.x; tile < num_tiles; tile += blockDim.x) {
    const int index = batch_id * num_tiles + tile;
    if (is_better(
        partial_scores[index], partial_ids[index], score, selected_id)) {
      score = partial_scores[index];
      selected_id = partial_ids[index];
    }
  }

  __shared__ float scores[kThreads];
  __shared__ int64_t ids[kThreads];
  scores[threadIdx.x] = score;
  ids[threadIdx.x] = selected_id;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset && is_better(
        scores[threadIdx.x + offset],
        ids[threadIdx.x + offset],
        scores[threadIdx.x],
        ids[threadIdx.x])) {
      scores[threadIdx.x] = scores[threadIdx.x + offset];
      ids[threadIdx.x] = ids[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[batch_id] = ids[0];
}

}  // namespace

torch::Tensor markov_argmax_cuda(
  const torch::Tensor& base_logits,
  const torch::Tensor& previous_embeddings,
  const torch::Tensor& projection_weight) {
  const int batch_size = base_logits.size(0);
  const int vocab_size = base_logits.size(1);
  const int rank = previous_embeddings.size(1);
  const int num_tiles = (vocab_size + kThreads - 1) / kThreads;
  auto partial_scores = torch::empty(
    {batch_size, num_tiles}, base_logits.options());
  auto partial_ids = torch::empty(
    {batch_size, num_tiles},
    base_logits.options().dtype(torch::kInt64));
  auto output = torch::empty(
    {batch_size}, base_logits.options().dtype(torch::kInt64));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_tiles, batch_size);
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    previous_embeddings.scalar_type(),
    "markov_partial_argmax",
    [&] {
      markov_partial_argmax_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
        base_logits.data_ptr<float>(),
        previous_embeddings.data_ptr<scalar_t>(),
        projection_weight.data_ptr<scalar_t>(),
        partial_scores.data_ptr<float>(),
        partial_ids.data_ptr<int64_t>(),
        vocab_size,
        rank,
        num_tiles);
    });
  reduce_partial_argmax_kernel<<<batch_size, kThreads, 0, stream>>>(
    partial_scores.data_ptr<float>(),
    partial_ids.data_ptr<int64_t>(),
    output.data_ptr<int64_t>(),
    num_tiles);
  return output;
}
