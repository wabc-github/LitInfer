#include "op.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#include "../../utils.hpp"

namespace llaisys::ops {

// 单个 (query i, head h) 的注意力。使用 online softmax：遍历 K/V 一次，
// 维护 running max、归一化因子和加权 value，避免为每个 head 分配 scores。
template <typename T>
void attention_one(const T *q_ptr, const T *k_ptr, const T *v_ptr, T *out_ptr,
                   size_t i, size_t h, size_t qlen, size_t nhead, size_t qd,
                   size_t klen, size_t nkvh, size_t group_size, float scale) {
    const size_t kv_head = h / group_size;
    const size_t context_end = i + (klen - qlen);
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum = 0.0f;
    std::vector<float> weighted(qd, 0.0f);

    for (size_t j = 0; j < klen; ++j) {
        if (j > context_end) continue;
        float score = 0.0f;
        for (size_t d = 0; d < qd; ++d) {
            size_t q_idx = i * nhead * qd + h * qd + d;
            size_t k_idx = j * nkvh * qd + kv_head * qd + d;
            score += utils::cast<float>(q_ptr[q_idx]) *
                     utils::cast<float>(k_ptr[k_idx]);
        }
        score *= scale;
        const float new_max = std::max(running_max, score);
        const float old_scale = running_sum == 0.0f
            ? 0.0f
            : std::exp(running_max - new_max);
        const float e = std::exp(score - new_max);
        running_sum = running_sum * old_scale + e;
        for (size_t d = 0; d < qd; ++d) {
            size_t v_idx = j * nkvh * qd + kv_head * qd + d;
            weighted[d] = weighted[d] * old_scale +
                          e * utils::cast<float>(v_ptr[v_idx]);
        }
        running_max = new_max;
    }
    for (size_t d = 0; d < qd; ++d) {
        size_t out_idx = i * nhead * qd + h * qd + d;
        out_ptr[out_idx] =
            utils::cast<T>(running_sum != 0.0f ? weighted[d] / running_sum : 0.0f);
    }
}

// 全量执行：query×head 展平并行。
template <typename T>
void attention_all(const void *q_v, const void *k_v, const void *v_v, void *out_v,
                   size_t qlen, size_t nhead, size_t qd,
                   size_t klen, size_t nkvh, float scale) {
    const T *q_ptr = reinterpret_cast<const T *>(q_v);
    const T *k_ptr = reinterpret_cast<const T *>(k_v);
    const T *v_ptr = reinterpret_cast<const T *>(v_v);
    T *out_ptr = reinterpret_cast<T *>(out_v);

    size_t group_size = nhead / nkvh;
    long long total = (long long)(qlen * nhead);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long idx = 0; idx < total; ++idx) {
        size_t i = (size_t)idx / nhead;
        size_t h = (size_t)idx % nhead;
        attention_one<T>(q_ptr, k_ptr, v_ptr, out_ptr, i, h, qlen, nhead, qd,
                         klen, nkvh, group_size, scale);
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v,
                    float scale) {
    PROFILE("self_attention");  // 计时：self_attention
    if (q->shape().size() != 3 || k->shape().size() != 3 ||
        v->shape().size() != 3 || attn_val->shape().size() != 3) {
        throw std::runtime_error("self_attention: expected rank-3 tensors");
    }
    size_t qlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t qd = q->shape()[2];
    size_t klen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    size_t kd = k->shape()[2];
    size_t vd = v->shape()[2];
    if (qd != kd || qd != vd || klen != v->shape()[0] ||
        nhead == 0 || nkvh == 0 || nhead % nkvh != 0 || klen < qlen) {
        throw std::runtime_error("Query, Key, and Value dimensions must match");
    }
    if (q->dtype() != k->dtype() || q->dtype() != v->dtype() ||
        q->dtype() != attn_val->dtype() || attn_val->shape() != q->shape()) {
        throw std::runtime_error("self_attention: shape or dtype mismatch");
    }
    if (q->deviceType() != k->deviceType() ||
        q->deviceType() != v->deviceType() ||
        q->deviceType() != attn_val->deviceType() ||
        q->deviceId() != k->deviceId() ||
        q->deviceId() != v->deviceId() ||
        q->deviceId() != attn_val->deviceId()) {
        throw std::runtime_error("self_attention: device mismatch");
    }
    if (!q->isContiguous() || !k->isContiguous() || !v->isContiguous() ||
        !attn_val->isContiguous()) {
        throw std::runtime_error("self_attention: tensors must be contiguous");
    }
    llaisysDataType_t type = q->dtype();

    // 并行区外先校验 dtype。
    if (type == LLAISYS_DTYPE_F32) {
        attention_all<float>(q->data(), k->data(), v->data(), attn_val->data(),
                             qlen, nhead, qd, klen, nkvh, scale);
    } else if (type == LLAISYS_DTYPE_F16) {
        attention_all<fp16_t>(q->data(), k->data(), v->data(), attn_val->data(),
                              qlen, nhead, qd, klen, nkvh, scale);
    } else if (type == LLAISYS_DTYPE_BF16) {
        attention_all<bf16_t>(q->data(), k->data(), v->data(), attn_val->data(),
                              qlen, nhead, qd, klen, nkvh, scale);
    } else {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops
