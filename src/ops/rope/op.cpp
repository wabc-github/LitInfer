#include "op.hpp"
#include <cmath>
#include <vector>

#include "../../utils.hpp"

namespace llaisys::ops {

// 单个 (token i, head h) 的旋转：用预计算好的 cos/sin 表。
template <typename T>
void rope_one(const T *in, T *out, size_t head_dim, size_t half,
              const float *cosv, const float *sinv) {
    for (size_t j = 0; j < half; ++j) {
        float a = utils::cast<float>(in[j]);
        float b = utils::cast<float>(in[j + half]);
        float c = cosv[j], s = sinv[j];
        // a' = a*cos - b*sin ; b' = b*cos + a*sin
        out[j] = utils::cast<T>(a * c - b * s);
        out[j + half] = utils::cast<T>(b * c + a * s);
    }
}

// 全量执行：先预计算 cos/sin（每 token 一次），再并行处理所有 token×head。
template <typename T>
void rope_all(const void *in_v, void *out_v, const int64_t *pos,
              size_t seq, size_t heads, size_t head_dim, float theta) {
    const T *in = reinterpret_cast<const T *>(in_v);
    T *out = reinterpret_cast<T *>(out_v);
    size_t half = head_dim / 2;

    // 预计算每个 token、每个半维的 cos/sin（原本在 head 维度被重复计算）。
    std::vector<float> cos_t(seq * half), sin_t(seq * half);
    for (size_t i = 0; i < seq; ++i) {
        float p = static_cast<float>(pos[i]);
        for (size_t j = 0; j < half; ++j) {
            float freq = p / std::pow(theta, static_cast<float>(2 * j) / head_dim);
            cos_t[i * half + j] = std::cos(freq);
            sin_t[i * half + j] = std::sin(freq);
        }
    }

    // 并行：token×head 展平，互不重叠，各写各的输出。
    long long total = (long long)(seq * heads);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long idx = 0; idx < total; ++idx) {
        size_t i = (size_t)idx / heads;
        size_t h = (size_t)idx % heads;
        size_t base = (i * heads + h) * head_dim;
        rope_one<T>(in + base, out + base, head_dim, half,
                    &cos_t[i * half], &sin_t[i * half]);
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    PROFILE("rope");  // 计时：rope
    if (in->shape().size() != 3 || out->shape() != in->shape() ||
        pos_ids->shape().size() != 1 || pos_ids->shape()[0] != in->shape()[0] ||
        pos_ids->dtype() != LLAISYS_DTYPE_I64 || out->dtype() != in->dtype() ||
        out->deviceType() != in->deviceType() ||
        pos_ids->deviceType() != in->deviceType() ||
        out->deviceId() != in->deviceId() || pos_ids->deviceId() != in->deviceId() ||
        !in->isContiguous() || !out->isContiguous() || !pos_ids->isContiguous() ||
        theta <= 0.0f) {
        throw std::runtime_error("rope: invalid input or output tensors");
    }
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    if (n_heads == 0 || head_dim == 0 || head_dim % 2 != 0) {
        throw std::runtime_error("Head dimension must be even for RoPE");
    }
    const int64_t *pos_ids_data =
        reinterpret_cast<const int64_t *>(pos_ids->data());
    llaisysDataType_t dtype = in->dtype();

    // 并行区外先校验 dtype。
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_all<float>(in->data(), out->data(), pos_ids_data,
                        seq_len, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_all<fp16_t>(in->data(), out->data(), pos_ids_data,
                         seq_len, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_all<bf16_t>(in->data(), out->data(), pos_ids_data,
                         seq_len, n_heads, head_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
