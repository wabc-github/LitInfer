#include "op.hpp"
#include <cmath>

#include "../../utils.hpp"

namespace llaisys::ops {

// 单条 rms_norm 处理：对第 i 行（长度 d）做归一化。
// Y_ij = W_j * X_ij / sqrt(mean(X_i.^2) + eps)
template <typename T>
void rms_norm_row(const T *in, const T *weight, T *out, size_t d, float eps) {
    // 第一遍：求平方和（读一次输入）。
    float sum_sq = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        float v = utils::cast<float>(in[j]);
        sum_sq += v * v;
    }
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + eps);
    // 第二遍：乘以权重与缩放，写输出。
    for (size_t j = 0; j < d; ++j) {
        float x = utils::cast<float>(in[j]);
        float w = utils::cast<float>(weight[j]);
        out[j] = utils::cast<T>(w * x * inv_rms);
    }
}

// 按行并行执行（dtype 已在外部校验）。
template <typename T>
void rms_norm_all(const void *in_v, const void *weight_v, void *out_v,
                  size_t rows, size_t d, float eps) {
    const T *in = reinterpret_cast<const T *>(in_v);
    const T *weight = reinterpret_cast<const T *>(weight_v);
    T *out = reinterpret_cast<T *>(out_v);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < (long long)rows; ++ii) {
        size_t i = (size_t)ii;
        rms_norm_row<T>(in + i * d, weight, out + i * d, d, eps);
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    PROFILE("rms_norm");  // 计时：rms_norm
    if (in->shape().size() != 2 || out->shape() != in->shape() ||
        weight->shape().size() != 1 || weight->shape()[0] != in->shape()[1] ||
        out->dtype() != in->dtype() || weight->dtype() != in->dtype() ||
        out->deviceType() != in->deviceType() ||
        weight->deviceType() != in->deviceType() ||
        out->deviceId() != in->deviceId() || weight->deviceId() != in->deviceId() ||
        !in->isContiguous() || !out->isContiguous() || !weight->isContiguous() ||
        eps < 0.0f || in->shape()[0] == 0 || in->shape()[1] == 0) {
        throw std::runtime_error("rms_norm: invalid input or output tensors");
    }
    size_t batch_size = in->shape()[0];   // 行数
    size_t feature_size = in->shape()[1]; // 每行元素数 (d)
    llaisysDataType_t dtype = in->dtype();

    // 并行区外先做类型校验（避免在 OpenMP 线程内抛异常）。
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_all<float>(in->data(), weight->data(), out->data(),
                            batch_size, feature_size, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_all<fp16_t>(in->data(), weight->data(), out->data(),
                             batch_size, feature_size, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_all<bf16_t>(in->data(), weight->data(), out->data(),
                             batch_size, feature_size, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
