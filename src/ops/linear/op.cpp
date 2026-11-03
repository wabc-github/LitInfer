#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    PROFILE("linear");  // 计时：linear（含 GEMV/GEMM）
    if (in->shape().empty() || weight->shape().size() != 2 ||
        out->shape().empty()) {
        throw std::runtime_error("linear: invalid tensor rank");
    }
    // 获取输入参数维度信息
    size_t in_features = in->shape().back(); // K
    size_t out_features = weight->shape()[0]; // N
    size_t weight_features = weight->shape()[1]; // K

    // Qwen2 exposes Q/K/V outputs as [tokens, heads, head_dim]. The storage
    // kernel still computes a row-major matrix, so derive M from element
    // counts instead of requiring the output tensor to be rank-2.
    if (in_features == 0 || out_features == 0 ||
        in->numel() % in_features != 0) {
        throw std::runtime_error("linear: invalid matrix dimensions");
    }
    size_t batch_size = in->numel() / in_features; // M

    // 确保输入特征数量匹配
    if (in_features != weight_features) {
        throw std::runtime_error("linear: input and weight features must match");
    }

    // 目前只支持连续张量（框架现有约定）。
    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "linear: tensors must be contiguous for now");

    // 获取数据类型
    llaisysDataType_t dtype = in->dtype();
    if (out->dtype() != dtype || weight->dtype() != dtype) {
        throw std::runtime_error("linear: dtype mismatch");
    }
    if (in->deviceType() != weight->deviceType() ||
        in->deviceId() != weight->deviceId() ||
        in->deviceType() != out->deviceType() ||
        in->deviceId() != out->deviceId()) {
        throw std::runtime_error("linear: device mismatch");
    }
    if (out->numel() != batch_size * out_features) {
        throw std::runtime_error("linear: output element count mismatch");
    }
    if (bias && (bias->shape().size() != 1 || bias->shape()[0] != out_features ||
                 bias->dtype() != dtype || bias->deviceType() != in->deviceType() ||
                 bias->deviceId() != in->deviceId() || !bias->isContiguous())) {
        throw std::runtime_error("linear: invalid bias");
    }

    const void *bias_data = bias ? bias->data() : nullptr;
    // CPU 路径
    cpu::linear_dispatch(dtype, in->data(), weight->data(), bias_data, out->data(),
                         batch_size, out_features, in_features);
}
} // namespace llaisys::ops
