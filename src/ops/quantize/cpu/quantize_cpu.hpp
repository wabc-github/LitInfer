#pragma once

#include <cstddef>

#include "llaisys.h"

namespace llaisys::ops::quantize::cpu {

// 运行时 CPU 是否支持 AVX2 / F16C / FMA（用于选择向量化 quant GEMV）。
bool cpu_supports_avx2_f16c_fma();
// AVX2 向量化内核是否已编译进来（由单独编译单元提供）。
bool quant_linear_avx_compiled();

// 权重量化 CPU 内核：
//   weight 浮点 [N,K] -> q int8 [N,K] + scale f32 [N]。
// weight_dtype 为浮点权重的类型；按行并行。
void quantize_weight_forward(llaisysDataType_t weight_dtype,
                             const void *weight,
                             int8_t *q, float *scale,
                             size_t N, size_t K);

// 量化线性 CPU 内核：
//   激活 in [M,K]（act_dtype）+ int8 权重 [N,K] + scale [N] -> out [M,N]（act_dtype）。
// bias 可为 nullptr。
void quant_linear_forward(llaisysDataType_t act_dtype,
                          const void *in,
                          const int8_t *q_weight,
                          const float *scale,
                          const void *bias,
                          void *out,
                          size_t M, size_t N, size_t K);

// AVX2 + F16C + FMA 向量化量化线性（仅 fp16 激活），签名与上面一致，
// 内部在 GEMV/GEMM 时用 _mm256_cvtepi8_epi32 在线反量化 int8 权重。
void quant_linear_avx_f16(const void *in,
                          const int8_t *q_weight,
                          const float *scale,
                          const void *bias,
                          void *out,
                          size_t M, size_t N, size_t K);

} // namespace llaisys::ops::quantize::cpu
