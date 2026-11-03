#include "../../../utils.hpp"
#include "quantize_cpu.hpp"

#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace llaisys::ops::quantize::cpu {
namespace {

// ------------------------------------------------------------------
// 1) 权重量化：对称 per-channel（per-row）。
// 对权重的每一行：amax = max|w|，scale = amax / 127，q = round(w / scale)。
// ------------------------------------------------------------------
template <typename T>
void quantize_rows(const T *weight, int8_t *q, float *scale,
                   size_t N, size_t K) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long nn = 0; nn < static_cast<long long>(N); ++nn) {
        const size_t n = static_cast<size_t>(nn);
        const T *wrow = weight + n * K;
        int8_t *qrow = q + n * K;

        // 第一遍：求该行绝对值最大值 amax。
        float amax = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float v = std::fabs(utils::cast<float>(wrow[k]));
            if (v > amax) amax = v;
        }
        // scale = amax / 127。全零行时给 scale=1 防止除零。
        float s = amax / 127.0f;
        if (!(s > 0.0f)) s = 1.0f;
        scale[n] = s;

        // 第二遍：量化并 clamp 到 [-127,127]（对称量化，-128 保留）。
        for (size_t k = 0; k < K; ++k) {
            long iv = std::lround(utils::cast<float>(wrow[k]) / s);
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            qrow[k] = static_cast<int8_t>(iv);
        }
    }
}

// ------------------------------------------------------------------
// 2) 量化线性：out = x @ (q * scale)^T + bias。
// 权重 int8 在线反量化，激活统一转 fp32 计算。
// ------------------------------------------------------------------

// GEMV（M=1，decode）：按输出行并行。
template <typename T>
void quant_gemv(const T *x, const int8_t *qw, const float *scale,
                const T *bias, T *out, size_t N, size_t K) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long nn = 0; nn < static_cast<long long>(N); ++nn) {
        const size_t n = static_cast<size_t>(nn);
        const int8_t *wrow = qw + n * K;
        float sum = 0.0f;
        // Σ q[n,k] * x[k]，先不乘 scale（scale 是整行公共因子，最后乘）。
        for (size_t k = 0; k < K; ++k) {
            sum += static_cast<float>(wrow[k]) * utils::cast<float>(x[k]);
        }
        sum *= scale[n];  // 反量化：乘该行 scale
        if (bias) sum += utils::cast<float>(bias[n]);
        out[n] = utils::cast<T>(sum);
    }
}

// GEMM（M>1，prefill）：按输入行并行，每线程私有输出行。
template <typename T>
void quant_gemm(const T *x, const int8_t *qw, const float *scale,
                const T *bias, T *out, size_t M, size_t N, size_t K) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long mm = 0; mm < static_cast<long long>(M); ++mm) {
        const size_t m = static_cast<size_t>(mm);
        const T *xrow = x + m * K;
        T *orow = out + m * N;
        for (size_t n = 0; n < N; ++n) {
            const int8_t *wrow = qw + n * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<float>(wrow[k]) * utils::cast<float>(xrow[k]);
            }
            sum *= scale[n];
            if (bias) sum += utils::cast<float>(bias[n]);
            orow[n] = utils::cast<T>(sum);
        }
    }
}

} // namespace

// 运行时检测 CPU 是否支持 AVX2/F16C/FMA（与 linear 中相同的 cpuid 逻辑）。
bool cpu_supports_avx2_f16c_fma() {
#if defined(_MSC_VER) && defined(_M_X64)
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 0);
    const int max_leaf = info[0];
    if (max_leaf < 1) return false;

    __cpuidex(info, 1, 0);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool fma = (info[2] & (1 << 12)) != 0;
    const bool f16c = (info[2] & (1 << 29)) != 0;
    if (!osxsave || !avx || !fma || !f16c) return false;
    if ((_xgetbv(0) & 0x6) != 0x6 || max_leaf < 7) return false;

    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  // AVX2
#elif (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}

// 权重量化入口。
void quantize_weight_forward(llaisysDataType_t weight_dtype,
                             const void *weight,
                             int8_t *q, float *scale,
                             size_t N, size_t K) {
    switch (weight_dtype) {
    case LLAISYS_DTYPE_F16:
        quantize_rows<fp16_t>(reinterpret_cast<const fp16_t *>(weight), q, scale, N, K);
        break;
    case LLAISYS_DTYPE_F32:
        quantize_rows<float>(reinterpret_cast<const float *>(weight), q, scale, N, K);
        break;
    case LLAISYS_DTYPE_BF16:
        quantize_rows<bf16_t>(reinterpret_cast<const bf16_t *>(weight), q, scale, N, K);
        break;
    default:
        throw std::runtime_error("quantize_weight: unsupported float weight dtype");
    }
}

// 量化线性入口（通用版本；AVX2 加速在 op.cpp 里按运行时选择）。
void quant_linear_forward(llaisysDataType_t act_dtype,
                          const void *in,
                          const int8_t *q_weight,
                          const float *scale,
                          const void *bias,
                          void *out,
                          size_t M, size_t N, size_t K) {
    if (act_dtype == LLAISYS_DTYPE_F16) {
        const auto *x = reinterpret_cast<const fp16_t *>(in);
        const auto *b = reinterpret_cast<const fp16_t *>(bias);
        auto *o = reinterpret_cast<fp16_t *>(out);
        if (M == 1) quant_gemv<fp16_t>(x, q_weight, scale, b, o, N, K);
        else quant_gemm<fp16_t>(x, q_weight, scale, b, o, M, N, K);
    } else if (act_dtype == LLAISYS_DTYPE_F32) {
        const auto *x = reinterpret_cast<const float *>(in);
        const auto *b = reinterpret_cast<const float *>(bias);
        auto *o = reinterpret_cast<float *>(out);
        if (M == 1) quant_gemv<float>(x, q_weight, scale, b, o, N, K);
        else quant_gemm<float>(x, q_weight, scale, b, o, M, N, K);
    } else {
        throw std::runtime_error("quant_linear: unsupported activation dtype");
    }
}

} // namespace llaisys::ops::quantize::cpu
