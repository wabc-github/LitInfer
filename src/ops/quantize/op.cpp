#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/quantize_cpu.hpp"

namespace llaisys::ops {

// 权重量化：浮点 [N,K] -> int8 [N,K] + scale [N]。
void quantize_weight(tensor_t q, tensor_t scale, tensor_t weight) {
    PROFILE("quantize_weight");  // 计时：quantize_weight
    // 形状/类型校验。
    if (weight->shape().size() != 2) {
        throw std::runtime_error("quantize_weight: weight must be rank-2 [N,K]");
    }
    size_t N = weight->shape()[0];
    size_t K = weight->shape()[1];
    if (q->shape().size() != 2 || q->shape()[0] != N || q->shape()[1] != K ||
        q->dtype() != LLAISYS_DTYPE_I8) {
        throw std::runtime_error("quantize_weight: q must be int8 [N,K]");
    }
    if (scale->shape().size() != 1 || scale->shape()[0] != N ||
        scale->dtype() != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("quantize_weight: scale must be float32 [N]");
    }
    // 浮点权重类型检查。
    llaisysDataType_t wdt = weight->dtype();
    if (wdt != LLAISYS_DTYPE_F16 && wdt != LLAISYS_DTYPE_F32 &&
        wdt != LLAISYS_DTYPE_BF16) {
        throw std::runtime_error("quantize_weight: weight must be float");
    }
    // 连续性要求（当前框架约定）。
    ASSERT(q->isContiguous() && scale->isContiguous() && weight->isContiguous(),
           "quantize_weight: tensors must be contiguous");

    // 交给 CPU 内核。
    quantize::cpu::quantize_weight_forward(wdt, weight->data(),
                                 reinterpret_cast<int8_t *>(q->data()),
                                 reinterpret_cast<float *>(scale->data()),
                                 N, K);
}

// 量化线性：out = in @ (q_weight * scale)^T + bias。
void quant_linear(tensor_t out, tensor_t in, tensor_t q_weight,
                  tensor_t scale, tensor_t bias) {
    PROFILE("quant_linear");  // 计时：quant_linear
    // 形状校验。
    if (in->shape().empty() || q_weight->shape().size() != 2 ||
        out->shape().empty()) {
        throw std::runtime_error("quant_linear: invalid tensor rank");
    }
    size_t K = in->shape().back();
    size_t N = q_weight->shape()[0];
    if (q_weight->shape()[1] != K) {
        throw std::runtime_error("quant_linear: K dimension mismatch");
    }
    // 激活行数 M 由元素数推导（支持 [tokens, heads, head_dim] 展平）。
    if (in->numel() % K != 0) {
        throw std::runtime_error("quant_linear: invalid input numel");
    }
    size_t M = in->numel() / K;

    // 类型校验：权重 int8，scale fp32，激活 fp16/fp32。
    llaisysDataType_t act_dtype = in->dtype();
    if (q_weight->dtype() != LLAISYS_DTYPE_I8) {
        throw std::runtime_error("quant_linear: q_weight must be int8");
    }
    if (scale->shape().size() != 1 || scale->shape()[0] != N ||
        scale->dtype() != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("quant_linear: scale must be float32 [N]");
    }
    if (act_dtype != LLAISYS_DTYPE_F16 && act_dtype != LLAISYS_DTYPE_F32) {
        throw std::runtime_error("quant_linear: activation must be fp16/fp32");
    }
    if (out->dtype() != act_dtype || out->numel() != M * N) {
        throw std::runtime_error("quant_linear: output dtype/shape mismatch");
    }
    if (bias && (bias->shape().size() != 1 || bias->shape()[0] != N ||
                 bias->dtype() != act_dtype)) {
        throw std::runtime_error("quant_linear: invalid bias");
    }
    // 连续性要求。
    ASSERT(in->isContiguous() && q_weight->isContiguous() &&
               scale->isContiguous() && out->isContiguous() &&
               (!bias || bias->isContiguous()),
           "quant_linear: tensors must be contiguous");

    const auto *qw = reinterpret_cast<const int8_t *>(q_weight->data());
    const auto *sc = reinterpret_cast<const float *>(scale->data());

    // 运行时选择 AVX2 向量化内核（fp16 激活且 CPU 支持）。
    if (act_dtype == LLAISYS_DTYPE_F16 && quantize::cpu::cpu_supports_avx2_f16c_fma() &&
        quantize::cpu::quant_linear_avx_compiled()) {
        quantize::cpu::quant_linear_avx_f16(in->data(), qw, sc,
                                  bias ? bias->data() : nullptr,
                                  out->data(), M, N, K);
        return;
    }
    // 通用内核。
    quantize::cpu::quant_linear_forward(act_dtype, in->data(), qw, sc,
                              bias ? bias->data() : nullptr,
                              out->data(), M, N, K);
}

} // namespace llaisys::ops
