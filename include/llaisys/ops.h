#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);

    // ---- INT8 weight-only 量化 ----
    // 浮点权重量化：weight [N,K] fp -> q [N,K] int8，scale [N] fp32（对称 per-row）。
    __export void llaisysQuantizeWeight(llaisysTensor_t q, llaisysTensor_t scale, llaisysTensor_t weight);
    // 融合在线反量化的线性层：out = in @ (q_weight * scale)^T + bias（bias 可传 NULL）。
    __export void llaisysQuantLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t q_weight, llaisysTensor_t scale, llaisysTensor_t bias);

    // ---- PagedAttention（分页 KV-Cache）----
    // 单序列、GQA、online softmax。KV 来自分页物理池，按 block_table 寻址。
    //   out:        [q_len, nh, dh]
    //   q:          [q_len, nh, dh]
    //   k_pool/v_pool: [num_blocks, block_tokens, nkvh, dh] 物理池
    //   block_table:   [num_logical_blocks] int32，元素为物理块 id
    //   context_len:   真实 KV token 数（最后一块可能未满）
    //   scale:         1/sqrt(dh)
    __export void llaisysPagedSelfAttention(llaisysTensor_t out, llaisysTensor_t q,
        llaisysTensor_t k_pool, llaisysTensor_t v_pool, llaisysTensor_t block_table,
        size_t context_len, float scale);

    // 多序列 batched PagedAttention（连续批处理核心）：
    //   out/q: [total_tokens, nh, dh]，各序列 query 按序拼接（total_tokens=Σ query_lens）
    //   k_pool/v_pool: [num_blocks, block_tokens, nkvh, dh] 共享物理池
    //   block_table:   [num_seqs, max_blocks] int32
    //   context_lens:  [num_seqs] int32，每序列当前 KV 长度
    //   query_lens:    [num_seqs] int32，本次每序列 query 数
    __export void llaisysBatchedPagedSelfAttention(llaisysTensor_t out, llaisysTensor_t q,
        llaisysTensor_t k_pool, llaisysTensor_t v_pool, llaisysTensor_t block_table,
        llaisysTensor_t context_lens, llaisysTensor_t query_lens, float scale);
}

#endif
