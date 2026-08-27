/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
// #include <cuda_pipeline.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float *red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// Grid: (num_heads, num_seqs)
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
    scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t
        *__restrict__ v_cache, // [num_blocks, num_kv_heads, head_size, block_size]
    const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const int num_kv_heads, const int q_stride) {
  // block-size是一个block里面的token数量
  // THREAD_GROUP_SIZE 表示多少个线程处理block中的一个token的特征
  // 例如这里可以是32/16=2个线程处理一个token的特征
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  // NUM_TOKENS_PER_THREAD_GROUP 避免block size大于warp
  // size时，一个线程组处理不了一个token 例如当block
  // size=64时，这个参数等于2，一个线程需要处理两个token的特征
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int seq_idx = blockIdx.y;

  const int queries_per_kv = num_heads / num_kv_heads;
  const int head_idx_kv = head_idx / queries_per_kv;

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t *q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  Q_vec q_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[i] = *reinterpret_cast<const Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
  }

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float *logits = reinterpret_cast<float *>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int *block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks;
       block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t *k_ptr =
            k_cache +
            physical_block_number * num_kv_heads * HEAD_SIZE * BLOCK_SIZE +
            head_idx_kv * HEAD_SIZE * BLOCK_SIZE + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec *>(
            k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      const float qk =
          scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
      const bool mask = token_idx >= context_len;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        logits[token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  for (int block_idx = warp_idx; block_idx < num_blocks;
       block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec,
               *reinterpret_cast<Float_L_vec *>(logits + token_idx));

    const scalar_t *v_ptr =
        v_cache + physical_block_number * num_kv_heads * HEAD_SIZE * BLOCK_SIZE +
        head_idx_kv * HEAD_SIZE * BLOCK_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec *>(v_ptr + offset);
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float *out_smem = reinterpret_cast<float *>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];   // 总共放了head-size个float元素进去
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t *out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_blocks_per_seq, num_heads, num_seqs)
// NUM_THREADS = 128
// TM是一个warp一次处理的query行数
// Qi需要用掉 TM * (head_size * data type) / 32 个寄存器，需要注意
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, const int TM = 2>
__global__ void varlen_query_cached_kv_attention_kernel(
    scalar_t *__restrict__ out,           // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ q,       // [num_tokens, num_heads, head_size],
                                          // packed by sequence
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                           // head_size/x, block_size, x]
    const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads, head_size, block_size]
    const int *__restrict__ cu_seqlens_q, // [num_seqs + 1]
    const int max_seqlen_q, const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int max_num_blocks_per_seq,
    const int num_kv_heads, const int q_stride) {
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int NUM_THREAD_GROUPS_PER_WARP = WARP_SIZE / THREAD_GROUP_SIZE;

  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane_idx = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.z;

  const int num_blocks_per_seq = gridDim.x;
  const int num_heads = gridDim.y;
  const int num_seqs = gridDim.z;

  const int queries_per_kv = num_heads / num_kv_heads;
  const int head_idx_kv = head_idx / queries_per_kv;

  const int num_tokens_per_block =
      (max_seqlen_q + num_blocks_per_seq - 1) / num_blocks_per_seq;
  
  // Queries are packed by sequence; cu_seqlens_q stores each sequence offset.
  const int q_offset = cu_seqlens_q[seq_idx];
  // 这个是当前block处理的seq的token offset，不是在整个q内的idx
  const int m_block_offset = blockIdx.x * num_tokens_per_block;
  // 计算当前block.y对应的seq的真实长度
  const int seqlen_q = cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx];
  // 计算当前block处理的Query的token范围
  const int num_q_tokens =
      MIN(num_tokens_per_block, seqlen_q - m_block_offset);
  // Queries are packed; extra grid blocks exit when this seq has no queries.
  if (num_q_tokens <= 0) return;

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  // 一个vec里面包含多少个scalar_t类型数据，当block-size=16、数据类型为half时
  // VEC_SIZE=2，即一个vec里面包含2个half类型数据
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
      
  // 每个线程处理head-size维度元素（scalar_t）的数量
  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  // 每个线程要处理的vec的数量
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  // Warp-local thread-group index.
  const int thread_group_idx = lane_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = lane_idx % THREAD_GROUP_SIZE;
  
  // 一个线程一次处理TM*(NUM_VECS * VEC_SIZE)大小的Query tile
  Q_vec Qi[TM * NUM_VECS_PER_THREAD];
  
  const scalar_t *q_ptr = q + (q_offset + m_block_offset) * q_stride
                            + head_idx * HEAD_SIZE;

  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  // v cache: [num_blocks, num_heads, head_size, block_size]
  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  constexpr int x = 16 / sizeof(scalar_t);

  // Workspace:
  // - block_logits: [NUM_WARPS, TM, BLOCK_SIZE]
  // Used to keep one KV block's logits per warp and per query row.
  extern __shared__ char shared_mem[];
  float* block_logits_smem = reinterpret_cast<float*>(shared_mem);

  const int *block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  // context_len includes both the cached prefix and the current query suffix.
  const int query_start = context_len - seqlen_q;
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // const int num_tiles_m = (num_q_tokens + TM - 1) / TM;
  // Q矩阵外循环，不同warp处理不同行，每个warp处理一个完整的kv行，就不用全局规约了
  for (int loop = warp_idx * TM; loop < num_q_tokens; loop += NUM_WARPS * TM) {
    for (int row_offset = 0; row_offset < TM; row_offset++) {
      // 填充当前Qi子块
      // q里面对应的token-idx起始下标
      const int token_idx = loop + row_offset;
      if (token_idx < num_q_tokens) {
#pragma unroll
        for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
          const int ele_idx = token_idx * q_stride + VEC_SIZE *
                              (thread_group_offset + i * THREAD_GROUP_SIZE);
          Qi[row_offset * NUM_VECS_PER_THREAD + i] = 
              *reinterpret_cast<const Q_vec*>(q_ptr + ele_idx);
        }
      }
    } // 搬运Qi完毕

    float running_max[TM];
    float running_l[TM];
    float accs[TM * NUM_ROWS_PER_THREAD];

    for (int row_offset = 0; row_offset < TM; row_offset++) {
      running_max[row_offset] = -FLT_MAX;
      running_l[row_offset] = 0.f;
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        accs[row_offset * NUM_ROWS_PER_THREAD + i] = 0.f;
      }
    }

    // Online softmax over KV blocks.
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
      const int physical_block_id = block_table[block_idx];
      float block_max_tg[TM];
      float block_max[TM];

      for (int row_offset = 0; row_offset < TM; row_offset++) {
        block_max_tg[row_offset] = -FLT_MAX;
      }

      // Compute block logits and block max.
      for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
        const int physical_block_offset =
            thread_group_idx + i * NUM_THREAD_GROUPS_PER_WARP;
        // tg/warp * tokens/tg = tokens/warp
        // 在一个warp负责一个block的前提下，下面的这个就是不可能事件
        // if (physical_block_offset >= BLOCK_SIZE) {  
        //   continue;
        // }
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

        K_vec k_vecs[NUM_VECS_PER_THREAD];
        const scalar_t* k_ptr =
            k_cache + physical_block_id * num_kv_heads * HEAD_SIZE * BLOCK_SIZE +
            head_idx_kv * HEAD_SIZE * BLOCK_SIZE + physical_block_offset * x;
#pragma unroll
        for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
          const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
          const int offset1 = (vec_idx * VEC_SIZE) / x;   // head size/x维度下标
          const int offset2 = (vec_idx * VEC_SIZE) % x;   // x维度下标
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }

        for (int row_offset = 0; row_offset < TM; row_offset++) {
          const int q_token_idx = loop + row_offset;
          // if (q_token_idx >= num_q_tokens) { // 这样会多执行一个指令
          //   continue;
          // }
          if (q_token_idx < num_q_tokens) {
            const Q_vec (&q_vecs)[NUM_VECS_PER_THREAD] =
              *reinterpret_cast<const Q_vec (*)[NUM_VECS_PER_THREAD]>(
                  &Qi[row_offset * NUM_VECS_PER_THREAD]);
            const float qk =
                scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
            if (thread_group_offset == 0) {
              float* row_logits = block_logits_smem +
                                  (warp_idx * TM + row_offset) * BLOCK_SIZE;
              const int query_position =
                  query_start + m_block_offset + q_token_idx;
              const bool mask = token_idx > query_position;
              row_logits[physical_block_offset] = mask ? -FLT_MAX : qk;
              block_max_tg[row_offset] = mask ? block_max_tg[row_offset] : 
                                                fmaxf(block_max_tg[row_offset], qk);
              // 避免warp divergence
              // if (token_idx < context_len) { 
              //   row_logits[physical_block_offset] = qk;
              //   block_max_tg[row_offset] =
              //       fmaxf(block_max_tg[row_offset], qk);
              // } else {
              //   row_logits[physical_block_offset] = -FLT_MAX;
              // }
            }
          }
        }
      }
      // Reduce block max inside each warp.
      for (int row_offset = 0; row_offset < TM; row_offset++) {
        const int q_token_idx = loop + row_offset;
        float m = block_max_tg[row_offset];
        if (q_token_idx < num_q_tokens) {
#pragma unroll
          for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE;
               mask /= 2) {
            m = fmaxf(m, __shfl_xor_sync(uint32_t(-1), m, mask));
          }
          // 广播到warp内所有线程
          m = __shfl_sync(uint32_t(-1), m, 0);
        }
        block_max[row_offset] = m;
      }
      __syncwarp();

      // Convert block logits to exp(logit - block_max), and reduce block sum.
      float block_l[TM];
      for (int row_offset = 0; row_offset < TM; row_offset++) {
        const int q_token_idx = loop + row_offset;
        float sum = 0.f;
        if (q_token_idx < num_q_tokens) {
          float* row_logits = block_logits_smem +
                              (warp_idx * TM + row_offset) * BLOCK_SIZE;
          const int query_position =
              query_start + m_block_offset + q_token_idx;
          for (int i = lane_idx; i < BLOCK_SIZE; i += WARP_SIZE) {
            const int token_idx = block_idx * BLOCK_SIZE + i;
            const bool mask = token_idx > query_position;
            float p = mask ? 0.f : __expf(row_logits[i] - block_max[row_offset]);
            // if (token_idx < context_len) {
            //   p = __expf(row_logits[i] - block_max[row_offset]);
            // }
            row_logits[i] = p;
            sum += p;
          }
#pragma unroll
          for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
            sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
          }
          sum = __shfl_sync(uint32_t(-1), sum, 0);
        }
        block_l[row_offset] = sum;
      }
      __syncwarp();

      // Compute unnormalized block output and merge using online softmax.
      for (int row_offset = 0; row_offset < TM; row_offset++) {
        const int q_token_idx = loop + row_offset;
        if (q_token_idx >= num_q_tokens) {
          continue;
        }

        float block_acc[NUM_ROWS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          block_acc[i] = 0.f;
        }

        const int physical_block_offset =
            (lane_idx % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
        L_vec probs_vec;
        float* row_logits = block_logits_smem +
                            (warp_idx * TM + row_offset) * BLOCK_SIZE;
        from_float(probs_vec, *reinterpret_cast<Float_L_vec*>(
                                  row_logits + physical_block_offset));

        const scalar_t* v_ptr =
            v_cache + physical_block_id * num_kv_heads * HEAD_SIZE * BLOCK_SIZE +
            head_idx_kv * HEAD_SIZE * BLOCK_SIZE;
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          const int row_idx =
              lane_idx / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
          if (row_idx < HEAD_SIZE) {
            const int offset_r = row_idx * BLOCK_SIZE + physical_block_offset;
            V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset_r);
            block_acc[i] += dot(probs_vec, v_vec);
          }
        }

        // Reduce duplicated work among lanes that process the same row.
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          float acc = block_acc[i];
#pragma unroll
          for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
            acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
          }
          block_acc[i] = acc;
        }
        // 状态更新
        const float prev_m = running_max[row_offset];
        const float prev_l = running_l[row_offset];
        const float cur_m = block_max[row_offset];
        const float new_m = fmaxf(prev_m, cur_m);
        const float alpha = prev_m == -FLT_MAX ? 0.f : __expf(prev_m - new_m);
        const float beta = __expf(cur_m - new_m);

        running_max[row_offset] = new_m;
        running_l[row_offset] = prev_l * alpha + block_l[row_offset] * beta;
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          accs[row_offset * NUM_ROWS_PER_THREAD + i] =
              accs[row_offset * NUM_ROWS_PER_THREAD + i] * alpha +
              block_acc[i] * beta;
        }
      }
    }

    // Normalize by final denominator.
    for (int row_offset = 0; row_offset < TM; row_offset++) {
      const int q_token_idx = loop + row_offset;
      if (q_token_idx < num_q_tokens) {
        const float inv_sum = __fdividef(1.f, running_l[row_offset] + 1e-6f);
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          accs[row_offset * NUM_ROWS_PER_THREAD + i] *= inv_sum;
        }
      }
    }

    // 更新输出子块Oi
    for (int row_offset = 0; row_offset < TM; row_offset++) {
      const int q_token_idx = loop + row_offset;
      if (q_token_idx < num_q_tokens) {
        const int token_idx = q_offset + m_block_offset + q_token_idx;
        scalar_t* out_ptr =
            out + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          const int row_idx =
              lane_idx / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
          if (row_idx < HEAD_SIZE && lane_idx % NUM_V_VECS_PER_ROW == 0) {
            from_float(*(out_ptr + row_idx),
                       accs[row_offset * NUM_ROWS_PER_THREAD + i]);
          }
        }
      }
    }
  }
}

} // namespace vllm

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)         \
  vllm::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE,      \
                                                NUM_THREADS>                   \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale,           \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,          \
          num_kv_heads, query_stride);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, float scale, torch::Tensor &block_tables,
    torch::Tensor &context_lens, int max_context_len) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int num_kv_heads = key_cache.size(1);
  TORCH_CHECK(num_kv_heads > 0, "KV head count must be positive");
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(value_cache.size(1) == num_kv_heads,
              "Key and value caches must have the same KV head count");

  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int query_stride = query.stride(0);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len =
      ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
  // NOTE(woosuk): To reduce the compilation time, we omitted head sizes
  // 32, 160, 192.
  // case 32:
  //   LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
  //   break;
  case 64:
    LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
    break;
  case 80:
    LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
    break;
  case 96:
    LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
    break;
  case 128:
    LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
    break;
  // case 160:
  //   LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE, NUM_THREADS);
  //   break;
  // case 192:
  //   LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE, NUM_THREADS);
  //   break;
  case 256:
    LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                    \
  single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(                    \
      out, query, key_cache, value_cache, scale, block_tables, context_lens,   \
      max_context_len);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                     \
  switch (block_size) {                                                        \
  /* case 1:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 1);   */                                        \
  /*   break;                        */                                        \
  /* case 2:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 2);   */                                        \
  /*   break;                        */                                        \
  /* case 4:                         */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 4);   */                                        \
  /*   break;                        */                                        \
  case 8:                                                                      \
    CALL_KERNEL_LAUNCHER(T, 8);                                                \
    break;                                                                     \
  case 16:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 16);                                               \
    break;                                                                     \
  case 32:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 32);                                               \
    break;                                                                     \
  /* case 64:                        */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 64);  */                                        \
  /*   break;                        */                                        \
  /* case 128:                       */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 128); */                                        \
  /*   break;                        */                                        \
  /* case 256:                       */                                        \
  /*   CALL_KERNEL_LAUNCHER(T, 256); */                                        \
  /*   break;                        */                                        \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }

void single_query_cached_kv_attention(
    torch::Tensor &out,   // [num_seqs, num_heads, head_size]
    torch::Tensor &query, // [num_seqs, num_heads, head_size]
    torch::Tensor
        &key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor
        &value_cache, // [num_blocks, num_heads, head_size, block_size]
    float scale,
    torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &context_lens, // [num_seqs]
    int block_size, int max_context_len) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}


#define LAUNCH_VARLEN_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, TM)   \
  vllm::varlen_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE,           \
                                                NUM_THREADS, TM>                    \
      <<<grid, block, shared_mem_size, stream>>>(                                   \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, cu_seqlens_q_ptr,     \
          max_seqlen_q, scale, block_tables_ptr, context_lens_ptr,                  \
          max_num_blocks_per_seq, num_kv_heads, query_stride)

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128, int TM = 4>
void varlen_query_cached_kv_attention_launcher(
    torch::Tensor &out,               // [num_tokens, num_heads, head_size]            
    torch::Tensor &query,             // [num_tokens, num_heads, head_size],
                                      // packed by sequence
    torch::Tensor &key_cache,         // [num_blocks, num_kv_heads,
                                      //  head_size/x, block_size, x]
    torch::Tensor &value_cache,       // [num_blocks, num_kv_heads, head_size, block_size]
    torch::Tensor &cu_seqlens_q,      // [num_seqs + 1]      
    int max_seqlen_q, float scale,            
    torch::Tensor &block_tables,      // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &context_lens,      // [num_seqs]      
    int max_context_len) {
  int num_seqs = cu_seqlens_q.size(0) - 1;
  int num_heads = query.size(1);
  int num_kv_heads = key_cache.size(1);
  TORCH_CHECK(num_kv_heads > 0, "KV head count must be positive");
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "Query heads must be divisible by KV heads");
  TORCH_CHECK(value_cache.size(1) == num_kv_heads,
              "Key and value caches must have the same KV head count");

  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int query_stride = query.stride(0);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();
  int *cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = 
      ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
      // Default output data type is float.
  int shared_mem_size = NUM_WARPS * TM * BLOCK_SIZE * sizeof(float);
  
  // 一个block内一次循环处理NUM_WARPS * TM行，默认处理四次
  constexpr int TOKENS_PER_BLOCK = NUM_WARPS * TM * 4;
  int max_blocks_per_seq = 
      (max_seqlen_q + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
  dim3 grid(max_blocks_per_seq, num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch(head_size) {
  case 64:
    LAUNCH_VARLEN_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS, TM);
    break;
  case 80:
    LAUNCH_VARLEN_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS, TM);
    break;
  case 96:
    LAUNCH_VARLEN_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS, TM);
    break;
  case 128:
    LAUNCH_VARLEN_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS, TM);
    break;
  case 256:
    LAUNCH_VARLEN_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS, TM);
    break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_VARLEN_KERNEL_LAUNCHER(T, BLOCK_SIZE)                      \
  varlen_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(             \
    out, query, key_cache, value_cache, cu_seqlens_q,                   \
    max_seqlen_q, scale, block_tables, context_lens, max_context_len);

#define CALL_VARLEN_KERNEL_LAUNCHER_BLOCK_SIZE(T)                       \
    switch (block_size) {                                               \
    case 8:                                                             \
      CALL_VARLEN_KERNEL_LAUNCHER(T, 8);                                \
      break;                                                            \
    case 16:                                                            \
      CALL_VARLEN_KERNEL_LAUNCHER(T, 16);                               \
      break;                                                            \
    case 32:                                                            \
      CALL_VARLEN_KERNEL_LAUNCHER(T, 32);                               \
      break;                                                            \
    default:                                                            \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);       \
      break;                                                            \
    }


void varlen_query_cached_kv_attention(
    torch::Tensor &out,               // [num_tokens, num_heads, head_size]     
    torch::Tensor &query,             // [num_tokens, num_heads, head_size], packed by seq
    torch::Tensor &key_cache,         // [num_blocks, num_heads, head_size/x, block_size, x]      
    torch::Tensor &value_cache,       // [num_blocks, num_heads, head_size, block_size]        
    torch::Tensor &cu_seqlens_q,      // [num_seqs + 1]
    int max_seqlen_q, float scale,
    torch::Tensor &block_tables,      // [num_seqs, max_num_blocks_per_seq]          
    torch::Tensor &context_lens,      // [num_seqs]
    int block_size, int max_context_len) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_VARLEN_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_VARLEN_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_VARLEN_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
