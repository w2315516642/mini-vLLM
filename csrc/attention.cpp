#include <torch/extension.h>

void single_query_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len);

void varlen_query_cached_kv_attention(
  torch::Tensor &out,               // [num_tokens, num_heads, head_size]     
  torch::Tensor &query,             // [num_tokens, num_heads, head_size], packed by seq
  torch::Tensor &key_cache,         // [num_blocks, num_heads, head_size/x, block_size, x]      
  torch::Tensor &value_cache,       // [num_blocks, num_heads, head_size, block_size]        
  torch::Tensor &cu_seqlens_q,      // [num_seqs + 1]
  int max_seqlen_q, float scale,
  torch::Tensor &block_tables,      // [num_seqs, max_num_blocks_per_seq]          
  torch::Tensor &context_lens,      // [num_seqs]
  int block_size, int max_context_len);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "single_query_cached_kv_attention",
    &single_query_cached_kv_attention,
    "Compute the attention between an input query and the cached key/value tensors");
  
  m.def(
    "varlen_query_cached_kv_attention",
    &varlen_query_cached_kv_attention,
    "Compute the attention between multiple queries and the cached key/value tensors");
}
