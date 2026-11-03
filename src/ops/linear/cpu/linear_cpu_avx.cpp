#include "../../../utils.hpp"
#include "linear_cpu.hpp"

#include <cstdint>

#if defined(LLAISYS_COMPILE_AVX_LINEAR) && \
    ((defined(_MSC_VER) && defined(_M_X64)) || \
     (defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)))
#include <immintrin.h>
#define LLAISYS_HAS_AVX_LINEAR 1
#else
#define LLAISYS_HAS_AVX_LINEAR 0
#endif

namespace llaisys::ops::cpu {

bool linear_avx_kernel_compiled() {
#if LLAISYS_HAS_AVX_LINEAR
    return true;
#else
    return false;
#endif
}

#if LLAISYS_HAS_AVX_LINEAR
namespace {

inline float horizontal_sum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline float dot_f16_fma(const uint16_t *weight, const uint16_t *input,
                         size_t in_features) {
    __m256 accumulator = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= in_features; k += 8) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(weight + k));
        const __m128i input_packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(input + k));
        const __m256 weight_f32 = _mm256_cvtph_ps(packed);
        const __m256 input_f32 = _mm256_cvtph_ps(input_packed);
        accumulator = _mm256_fmadd_ps(weight_f32, input_f32, accumulator);
    }

    float sum = horizontal_sum(accumulator);
    for (; k < in_features; ++k) {
        fp16_t value{};
        value._v = weight[k];
        fp16_t input_value{};
        input_value._v = input[k];
        sum += utils::cast<float>(value) * utils::cast<float>(input_value);
    }
    return sum;
}

} // namespace
#endif

void linear_forward_avx_f16(const void *input_v, const void *weight_v,
                            const void *bias_v, void *output_v,
                            size_t rows, size_t out_features,
                            size_t in_features) {
#if LLAISYS_HAS_AVX_LINEAR
    const auto *input = reinterpret_cast<const fp16_t *>(input_v);
    const auto *weight = reinterpret_cast<const fp16_t *>(weight_v);
    const auto *bias = reinterpret_cast<const fp16_t *>(bias_v);
    auto *output = reinterpret_cast<fp16_t *>(output_v);

    if (rows == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long long jj = 0; jj < static_cast<long long>(out_features); ++jj) {
            const size_t j = static_cast<size_t>(jj);
            float sum = dot_f16_fma(
                reinterpret_cast<const uint16_t *>(weight + j * in_features),
                reinterpret_cast<const uint16_t *>(input), in_features);
            if (bias) sum += utils::cast<float>(bias[j]);
            output[j] = utils::cast<fp16_t>(sum);
        }
        return;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(rows); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        const auto *input_row = reinterpret_cast<const uint16_t *>(
            input + i * in_features);
        for (size_t j = 0; j < out_features; ++j) {
            float sum = dot_f16_fma(
                reinterpret_cast<const uint16_t *>(weight + j * in_features),
                input_row, in_features);
            if (bias) sum += utils::cast<float>(bias[j]);
            output[i * out_features + j] = utils::cast<fp16_t>(sum);
        }
    }
#else
    (void)input_v;
    (void)weight_v;
    (void)bias_v;
    (void)output_v;
    (void)rows;
    (void)out_features;
    (void)in_features;
#endif
}

} // namespace llaisys::ops::cpu
