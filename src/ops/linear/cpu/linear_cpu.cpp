#include "../../../utils.hpp"
#include "linear_cpu.hpp"

#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace llaisys::ops::cpu {
namespace {

template <typename T>
inline float to_float(T value) {
    return utils::cast<float>(value);
}

template <typename T>
void linear_forward_scalar(const T *input, const T *weight, const T *bias,
                           T *output, size_t rows, size_t out_features,
                           size_t in_features) {
    // Decode is GEMV: parallelize output rows because there is only one
    // input row to distribute.
    if (rows == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long long jj = 0; jj < static_cast<long long>(out_features); ++jj) {
            const size_t j = static_cast<size_t>(jj);
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                sum += to_float(input[k]) * to_float(weight[j * in_features + k]);
            }
            if (bias) sum += to_float(bias[j]);
            output[j] = utils::cast<T>(sum);
        }
        return;
    }

    // Prefill is GEMM-like. Keep each input row private to avoid write races
    // and preserve contiguous output-row writes.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < static_cast<long long>(rows); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        const T *input_row = input + i * in_features;
        T *output_row = output + i * out_features;
        for (size_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            const T *weight_row = weight + j * in_features;
            for (size_t k = 0; k < in_features; ++k) {
                sum += to_float(input_row[k]) * to_float(weight_row[k]);
            }
            if (bias) sum += to_float(bias[j]);
            output_row[j] = utils::cast<T>(sum);
        }
    }
}

} // namespace

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
    return (info[1] & (1 << 5)) != 0; // AVX2
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

void linear_dispatch(llaisysDataType_t dtype,
                     const void *input, const void *weight,
                     const void *bias, void *output,
                     size_t rows, size_t out_features,
                     size_t in_features) {
    if (dtype == LLAISYS_DTYPE_F16 && cpu_supports_avx2_f16c_fma() &&
        linear_avx_kernel_compiled()) {
        linear_forward_avx_f16(input, weight, bias, output,
                               rows, out_features, in_features);
        return;
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_forward_scalar<float>(
            reinterpret_cast<const float *>(input),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            reinterpret_cast<float *>(output), rows, out_features, in_features);
        break;
    case LLAISYS_DTYPE_F16:
        linear_forward_scalar<fp16_t>(
            reinterpret_cast<const fp16_t *>(input),
            reinterpret_cast<const fp16_t *>(weight),
            reinterpret_cast<const fp16_t *>(bias),
            reinterpret_cast<fp16_t *>(output), rows, out_features, in_features);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_forward_scalar<bf16_t>(
            reinterpret_cast<const bf16_t *>(input),
            reinterpret_cast<const bf16_t *>(weight),
            reinterpret_cast<const bf16_t *>(bias),
            reinterpret_cast<bf16_t *>(output), rows, out_features, in_features);
        break;
    default:
        throw std::runtime_error("linear: unsupported dtype");
    }
}

} // namespace llaisys::ops::cpu
