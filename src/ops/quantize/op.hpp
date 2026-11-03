#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// 对称 per-channel（per-row）权重量化。
//   weight: 浮点权重，shape [N, K]，dtype F16/F32/BF16
//   q:      输出 int8 权重，shape [N, K]，dtype I8
//   scale:  输出每行反量化 scale，shape [N]，dtype F32
// 量化关系：weight ≈ q * scale（逐行）。
void quantize_weight(tensor_t q, tensor_t scale, tensor_t weight);

// 融合"在线反量化"的线性层：out = x @ (q_weight * scale)^T + bias
//   out:      shape [M, N]，dtype 与激活 in 相同
//   in:       激活，shape [M, K]，dtype F16/F32
//   q_weight: int8 权重，shape [N, K]，dtype I8
//   scale:    每行 scale，shape [N]，dtype F32
//   bias:     可选，shape [N]，dtype 与激活相同；无则传 nullptr
void quant_linear(tensor_t out, tensor_t in, tensor_t q_weight,
                  tensor_t scale, tensor_t bias);

} // namespace llaisys::ops
