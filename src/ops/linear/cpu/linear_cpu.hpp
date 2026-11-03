#pragma once

#include <cstddef>

#include "llaisys.h"

namespace llaisys::ops::cpu {

// The generic implementation is always available. The AVX2/FMA/F16C kernel
// is compiled separately and selected only after runtime CPU feature checks.
bool cpu_supports_avx2_f16c_fma();
bool linear_avx_kernel_compiled();

void linear_forward_avx_f16(const void *input, const void *weight,
                            const void *bias, void *output,
                            size_t rows, size_t out_features,
                            size_t in_features);

void linear_dispatch(llaisysDataType_t dtype,
                     const void *input, const void *weight,
                     const void *bias, void *output,
                     size_t rows, size_t out_features,
                     size_t in_features);

} // namespace llaisys::ops::cpu
