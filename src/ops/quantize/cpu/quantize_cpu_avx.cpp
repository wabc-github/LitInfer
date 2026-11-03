#include "../../../utils.hpp"
#include "quantize_cpu.hpp"

#include <cstdint>

#if defined(LLAISYS_COMPILE_AVX_QUANT) && \
    ((defined(_MSC_VER) && defined(_M_X64)) || \
     (defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)))
#include <immintrin.h>
#define LLAISYS_HAS_AVX_QUANT 1
#else
#define LLAISYS_HAS_AVX_QUANT 0
#endif

namespace llaisys::ops::quantize::cpu {

bool quant_linear_avx_compiled() {
#if LLAISYS_HAS_AVX_QUANT
    return true;
#else
    return false;
#endif
}

#if LLAISYS_HAS_AVX_QUANT
namespace {

// 256 位向量的水平求和（8 个 fp32 累加成 1 个标量）。
inline float horizontal_sum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// 单个输出行的点积：int8 权重在线反量化（不乘 scale），fp16 激活。
// 返回 Σ_k q[n,k] * x[k]（scale 由调用方最后乘）。
inline float dot_int8_f16(const int8_t *w, const uint16_t *x, size_t K) {
    __m256 accumulator = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        // 8 个 int8 权重 = 8 字节：扩成 8 个 int32，再转 fp32。
        const __m128i q8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(w + k));
        const __m256i q32 = _mm256_cvtepi8_epi32(q8);
        const __m256 wf = _mm256_cvtepi32_ps(q32);
        // 8 个 fp16 激活 = 16 字节：F16C 转成 8 个 fp32。
        const __m128i x16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(x + k));
        const __m256 xf = _mm256_cvtph_ps(x16);
        // 融合乘加：acc += wf * xf。
        accumulator = _mm256_fmadd_ps(wf, xf, accumulator);
    }
    float sum = horizontal_sum(accumulator);
    // 尾部不足 8 个元素用标量处理。
    for (; k < K; ++k) {
        fp16_t xv{};
        xv._v = x[k];
        sum += static_cast<float>(w[k]) * utils::cast<float>(xv);
    }
    return sum;
}

} // namespace
#endif

// AVX2 量化线性入口（仅 fp16 激活）。
void quant_linear_avx_f16(const void *in_v,
                          const int8_t *q_weight,
                          const float *scale,
                          const void *bias_v,
                          void *out_v,
                          size_t M, size_t N, size_t K) {
#if LLAISYS_HAS_AVX_QUANT
    const auto *x = reinterpret_cast<const uint16_t *>(in_v);
    const auto *bias = reinterpret_cast<const fp16_t *>(bias_v);
    auto *out = reinterpret_cast<fp16_t *>(out_v);

    if (M == 1) {
        // GEMV（decode）：按输出行并行。
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long long nn = 0; nn < static_cast<long long>(N); ++nn) {
            const size_t n = static_cast<size_t>(nn);
            float sum = dot_int8_f16(q_weight + n * K, x, K);
            sum *= scale[n];  // 反量化：乘该行 scale
            if (bias) sum += utils::cast<float>(bias[n]);
            out[n] = utils::cast<fp16_t>(sum);
        }
        return;
    }

    // GEMM（prefill）：按输入行并行。
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long mm = 0; mm < static_cast<long long>(M); ++mm) {
        const size_t m = static_cast<size_t>(mm);
        const uint16_t *xrow = x + m * K;
        fp16_t *orow = out + m * N;
        for (size_t n = 0; n < N; ++n) {
            float sum = dot_int8_f16(q_weight + n * K, xrow, K);
            sum *= scale[n];
            if (bias) sum += utils::cast<float>(bias[n]);
            orow[n] = utils::cast<fp16_t>(sum);
        }
    }
#else
    (void)in_v;
    (void)q_weight;
    (void)scale;
    (void)bias_v;
    (void)out_v;
    (void)M;
    (void)N;
    (void)K;
#endif
}

} // namespace llaisys::ops::quantize::cpu
