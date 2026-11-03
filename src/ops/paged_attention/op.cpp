#include "op.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstdint>

#include "../../utils.hpp"

namespace llaisys::ops {

// 单个 (query i, head h) 的 PagedAttention。online softmax 遍历 KV 一次，
// 关键区别：第 t 个 KV 不连续存放，先用 block_table 把逻辑位置映射到物理块。
template <typename T>
void paged_attention_one(const T *q_ptr, const T *k_pool, const T *v_pool,
                         const int32_t *btable, T *out_ptr,
                         size_t i, size_t h, size_t qlen, size_t qstart,
                         size_t nhead, size_t qd, size_t nkvh,
                         size_t group_size, size_t block_tokens, float scale) {
    const size_t kv_head = h / group_size;
    // q 位置 i 在序列中的绝对位置 = qstart + i，因果上它只能看到 [0, qstart+i]。
    const size_t context_end = qstart + i;

    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum = 0.0f;
    std::vector<float> weighted(qd, 0.0f);

    for (size_t t = 0; t <= context_end; ++t) {
        // 逻辑位置 -> (逻辑块, 块内偏移) -> 物理块。
        size_t lb = t / block_tokens;
        size_t off = t % block_tokens;
        int pb = btable[lb];
        // 物理池中该 token 的起始偏移（池 layout: [num_blocks, block, nkvh, dh]）。
        size_t base = ((static_cast<size_t>(pb) * block_tokens + off) * nkvh
                       + kv_head) * qd;

        // Q·K。
        size_t q_base = (i * nhead + h) * qd;
        float score = 0.0f;
        for (size_t d = 0; d < qd; ++d) {
            score += utils::cast<float>(q_ptr[q_base + d]) *
                     utils::cast<float>(k_pool[base + d]);
        }
        score *= scale;

        // online softmax 更新。
        const float new_max = std::max(running_max, score);
        const float old_scale = running_sum == 0.0f
            ? 0.0f
            : std::exp(running_max - new_max);
        const float e = std::exp(score - new_max);
        running_sum = running_sum * old_scale + e;
        for (size_t d = 0; d < qd; ++d) {
            weighted[d] = weighted[d] * old_scale +
                          e * utils::cast<float>(v_pool[base + d]);
        }
        running_max = new_max;
    }

    for (size_t d = 0; d < qd; ++d) {
        size_t out_base = (i * nhead + h) * qd;
        out_ptr[out_base + d] =
            utils::cast<T>(running_sum != 0.0f ? weighted[d] / running_sum : 0.0f);
    }
}

// 全量执行：query×head 展平并行。
template <typename T>
void paged_attention_all(const void *q_v, const void *kp_v, const void *vp_v,
                         const void *bt_v, void *out_v,
                         size_t qlen, size_t qstart, size_t nhead, size_t qd,
                         size_t nkvh, size_t block_tokens, float scale) {
    const T *q_ptr = reinterpret_cast<const T *>(q_v);
    const T *k_pool = reinterpret_cast<const T *>(kp_v);
    const T *v_pool = reinterpret_cast<const T *>(vp_v);
    const int32_t *btable = reinterpret_cast<const int32_t *>(bt_v);
    T *out_ptr = reinterpret_cast<T *>(out_v);

    size_t group_size = nhead / nkvh;
    long long total = static_cast<long long>(qlen * nhead);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long idx = 0; idx < total; ++idx) {
        size_t i = static_cast<size_t>(idx) / nhead;
        size_t h = static_cast<size_t>(idx) % nhead;
        paged_attention_one<T>(q_ptr, k_pool, v_pool, btable, out_ptr,
                               i, h, qlen, qstart, nhead, qd, nkvh,
                               group_size, block_tokens, scale);
    }
}

void paged_self_attention(tensor_t attn_val, tensor_t q,
                          tensor_t k_pool, tensor_t v_pool,
                          tensor_t block_table,
                          size_t context_len, float scale) {
    PROFILE("paged_self_attention");

    // ---- 形状校验 ----
    if (q->shape().size() != 3 || attn_val->shape().size() != 3)
        throw std::runtime_error("paged_attention: q and out must be rank-3");
    if (k_pool->shape().size() != 4 || v_pool->shape().size() != 4)
        throw std::runtime_error("paged_attention: k/v pool must be rank-4");
    if (block_table->shape().size() != 1)
        throw std::runtime_error("paged_attention: block_table must be rank-1");

    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t qd = q->shape()[2];
    size_t nb = k_pool->shape()[0];
    size_t block_tokens = k_pool->shape()[1];
    size_t nkvh = k_pool->shape()[2];
    size_t pd = k_pool->shape()[3];

    if (context_len < qlen)
        throw std::runtime_error("paged_attention: context_len must be >= qlen");
    if (qd != pd || nhead == 0 || nkvh == 0 || nhead % nkvh != 0)
        throw std::runtime_error("paged_attention: head dimensions mismatch");
    if (v_pool->shape()[0] != nb || v_pool->shape()[1] != block_tokens ||
        v_pool->shape()[2] != nkvh || v_pool->shape()[3] != qd)
        throw std::runtime_error("paged_attention: k/v pool shapes mismatch");
    size_t needed_logical = (context_len + block_tokens - 1) / block_tokens;
    if (block_table->shape()[0] < needed_logical)
        throw std::runtime_error("paged_attention: block_table too short for context");

    // ---- dtype / device 校验 ----
    if (block_table->dtype() != LLAISYS_DTYPE_I32)
        throw std::runtime_error("paged_attention: block_table must be int32");
    if (q->dtype() != k_pool->dtype() || q->dtype() != v_pool->dtype() ||
        q->dtype() != attn_val->dtype() || attn_val->shape() != q->shape())
        throw std::runtime_error("paged_attention: shape or dtype mismatch");
    if (q->deviceType() != k_pool->deviceType() ||
        q->deviceType() != v_pool->deviceType() ||
        q->deviceType() != attn_val->deviceType() ||
        q->deviceType() != block_table->deviceType())
        throw std::runtime_error("paged_attention: device mismatch");
    if (!q->isContiguous() || !k_pool->isContiguous() ||
        !v_pool->isContiguous() || !attn_val->isContiguous() ||
        !block_table->isContiguous())
        throw std::runtime_error("paged_attention: tensors must be contiguous");

    // query 在序列中的起始位置（prefill 时为 0，decode 时为 context_len-1）。
    size_t qstart = context_len - qlen;

    llaisysDataType_t type = q->dtype();
    if (type == LLAISYS_DTYPE_F32) {
        paged_attention_all<float>(q->data(), k_pool->data(), v_pool->data(),
                                   block_table->data(), attn_val->data(),
                                   qlen, qstart, nhead, qd, nkvh,
                                   block_tokens, scale);
    } else if (type == LLAISYS_DTYPE_F16) {
        paged_attention_all<fp16_t>(q->data(), k_pool->data(), v_pool->data(),
                                    block_table->data(), attn_val->data(),
                                    qlen, qstart, nhead, qd, nkvh,
                                    block_tokens, scale);
    } else if (type == LLAISYS_DTYPE_BF16) {
        paged_attention_all<bf16_t>(q->data(), k_pool->data(), v_pool->data(),
                                    block_table->data(), attn_val->data(),
                                    qlen, qstart, nhead, qd, nkvh,
                                    block_tokens, scale);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

// ------------------------------------------------------------------
// 多序列 batched PagedAttention（连续批处理）
// ------------------------------------------------------------------

template <typename T>
void batched_paged_attention_all(const void *q_v, const void *kp_v, const void *vp_v,
                                 const void *bt_v, const void *cl_v, const void *ql_v,
                                 void *out_v,
                                 size_t nseq, size_t nhead, size_t qd,
                                 size_t nkvh, size_t max_blocks,
                                 size_t block_tokens, float scale) {
    const T *q_ptr = reinterpret_cast<const T *>(q_v);
    const T *k_pool = reinterpret_cast<const T *>(kp_v);
    const T *v_pool = reinterpret_cast<const T *>(vp_v);
    const int32_t *bt = reinterpret_cast<const int32_t *>(bt_v);
    const int32_t *cl = reinterpret_cast<const int32_t *>(cl_v);
    const int32_t *ql = reinterpret_cast<const int32_t *>(ql_v);
    T *out_ptr = reinterpret_cast<T *>(out_v);

    const size_t group_size = nhead / nkvh;
    // 每序列 query 在拼接张量中的起始偏移（前缀和）。
    std::vector<size_t> qoff(nseq + 1, 0);
    for (size_t s = 0; s < nseq; ++s) qoff[s + 1] = qoff[s] + ql[s];

    long long total = static_cast<long long>(qoff[nseq] * nhead);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long idx = 0; idx < total; ++idx) {
        size_t gq = static_cast<size_t>(idx) / nhead;
        size_t h = static_cast<size_t>(idx) % nhead;
        // 用前缀和定位该全局 query 所属序列 s 与序列内位置 i。
        size_t s = static_cast<size_t>(
            std::upper_bound(qoff.begin(), qoff.end(), gq) - qoff.begin() - 1);
        size_t i = gq - qoff[s];
        const size_t ctx = cl[s];
        const size_t qlen = ql[s];
        const size_t kvh = h / group_size;
        const size_t context_end = ctx - qlen + i;  // 因果可见范围

        float running_max = -std::numeric_limits<float>::infinity();
        float running_sum = 0.0f;
        std::vector<float> weighted(qd, 0.0f);

        for (size_t t = 0; t <= context_end; ++t) {
            size_t lb = t / block_tokens, off = t % block_tokens;
            int pb = bt[s * max_blocks + lb];
            size_t base = ((static_cast<size_t>(pb) * block_tokens + off)
                           * nkvh + kvh) * qd;
            size_t qbase = (gq * nhead + h) * qd;
            float score = 0.0f;
            for (size_t d = 0; d < qd; ++d)
                score += utils::cast<float>(q_ptr[qbase + d]) *
                         utils::cast<float>(k_pool[base + d]);
            score *= scale;
            const float new_max = std::max(running_max, score);
            const float old_scale = running_sum == 0.0f
                ? 0.0f : std::exp(running_max - new_max);
            const float e = std::exp(score - new_max);
            running_sum = running_sum * old_scale + e;
            for (size_t d = 0; d < qd; ++d)
                weighted[d] = weighted[d] * old_scale +
                              e * utils::cast<float>(v_pool[base + d]);
            running_max = new_max;
        }
        for (size_t d = 0; d < qd; ++d)
            out_ptr[(gq * nhead + h) * qd + d] =
                utils::cast<T>(running_sum != 0.0f ? weighted[d] / running_sum : 0.0f);
    }
}

void batched_paged_self_attention(tensor_t attn_val, tensor_t q,
                                  tensor_t k_pool, tensor_t v_pool,
                                  tensor_t block_table,
                                  tensor_t context_lens, tensor_t query_lens,
                                  float scale) {
    PROFILE("batched_paged_self_attention");

    if (q->shape().size() != 3 || attn_val->shape().size() != 3)
        throw std::runtime_error("batched_paged: q/out must be rank-3");
    if (k_pool->shape().size() != 4 || v_pool->shape().size() != 4)
        throw std::runtime_error("batched_paged: pool must be rank-4");
    if (block_table->shape().size() != 2)
        throw std::runtime_error("batched_paged: block_table must be rank-2 [num_seqs,max_blocks]");
    if (context_lens->shape().size() != 1 || query_lens->shape().size() != 1)
        throw std::runtime_error("batched_paged: context_lens/query_lens must be rank-1");

    size_t total_q = q->shape()[0], nhead = q->shape()[1], qd = q->shape()[2];
    size_t nb = k_pool->shape()[0], block_tokens = k_pool->shape()[1];
    size_t nkvh = k_pool->shape()[2], pd = k_pool->shape()[3];
    size_t nseq = block_table->shape()[0], max_blocks = block_table->shape()[1];

    if (qd != pd || nhead % nkvh != 0)
        throw std::runtime_error("batched_paged: head dimensions mismatch");
    if (context_lens->shape()[0] != nseq || query_lens->shape()[0] != nseq)
        throw std::runtime_error("batched_paged: length tensors must have num_seqs entries");
    if (block_table->dtype() != LLAISYS_DTYPE_I32 ||
        context_lens->dtype() != LLAISYS_DTYPE_I32 ||
        query_lens->dtype() != LLAISYS_DTYPE_I32)
        throw std::runtime_error("batched_paged: block_table/context_lens/query_lens must be int32");
    if (q->dtype() != k_pool->dtype() || q->dtype() != v_pool->dtype() ||
        q->dtype() != attn_val->dtype() || attn_val->shape() != q->shape())
        throw std::runtime_error("batched_paged: shape or dtype mismatch");

    // 读入长度（CPU 主线），校验 Σ query = total_q、ctx>=qlen、块数足够。
    const int32_t *cl = reinterpret_cast<const int32_t *>(context_lens->data());
    const int32_t *ql = reinterpret_cast<const int32_t *>(query_lens->data());
    long long sum_q = 0;
    for (size_t s = 0; s < nseq; ++s) {
        sum_q += ql[s];
        if (cl[s] < ql[s])
            throw std::runtime_error("batched_paged: context_len must be >= query_len");
        size_t needed = (cl[s] + block_tokens - 1) / block_tokens;
        if (needed > max_blocks)
            throw std::runtime_error("batched_paged: block_table row too short");
        if (needed > nb)
            throw std::runtime_error("batched_paged: context_len exceeds pool capacity");
    }
    if (static_cast<size_t>(sum_q) != total_q)
        throw std::runtime_error("batched_paged: Σ query_lens must equal q tokens");

    llaisysDataType_t type = q->dtype();
    if (type == LLAISYS_DTYPE_F32) {
        batched_paged_attention_all<float>(q->data(), k_pool->data(), v_pool->data(),
            block_table->data(), context_lens->data(), query_lens->data(),
            attn_val->data(), nseq, nhead, qd, nkvh, max_blocks, block_tokens, scale);
    } else if (type == LLAISYS_DTYPE_F16) {
        batched_paged_attention_all<fp16_t>(q->data(), k_pool->data(), v_pool->data(),
            block_table->data(), context_lens->data(), query_lens->data(),
            attn_val->data(), nseq, nhead, qd, nkvh, max_blocks, block_tokens, scale);
    } else if (type == LLAISYS_DTYPE_BF16) {
        batched_paged_attention_all<bf16_t>(q->data(), k_pool->data(), v_pool->data(),
            block_table->data(), context_lens->data(), query_lens->data(),
            attn_val->data(), nseq, nhead, qd, nkvh, max_blocks, block_tokens, scale);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops
