#pragma once

#include "../../tensor/tensor.hpp"
#include <cstddef>

namespace llaisys::ops {
// PagedAttention：单序列、GQA、online softmax。
// KV 来自分页物理池，按 block_table 寻址（逻辑块 -> 物理块）。
void paged_self_attention(tensor_t attn_val, tensor_t q,
                          tensor_t k_pool, tensor_t v_pool,
                          tensor_t block_table,
                          size_t context_len, float scale);

// 多序列 batched PagedAttention（连续批处理核心）。
// block_table: [num_seqs, max_blocks]；context_lens/query_lens: [num_seqs]。
void batched_paged_self_attention(tensor_t attn_val, tensor_t q,
                                  tensor_t k_pool, tensor_t v_pool,
                                  tensor_t block_table,
                                  tensor_t context_lens, tensor_t query_lens,
                                  float scale);
}
