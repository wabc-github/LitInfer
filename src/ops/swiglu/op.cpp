#include "op.hpp"
#include <cmath>

#include "../../utils.hpp"

namespace llaisys::ops {

// 逐元素：out = up * swish(gate)，swish(gate) = gate / (1 + exp(-gate))。
template <typename T>
void swiglu_all(const void *gate_v, const void *up_v, void *out_v, size_t numel) {
    const T *gate = reinterpret_cast<const T *>(gate_v);
    const T *up = reinterpret_cast<const T *>(up_v);
    T *out = reinterpret_cast<T *>(out_v);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long ii = 0; ii < (long long)numel; ++ii) {
        size_t i = (size_t)ii;
        float g = utils::cast<float>(gate[i]);
        float u = utils::cast<float>(up[i]);
        float swish = g / (1.0f + std::exp(-g));
        out[i] = utils::cast<T>(u * swish);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    PROFILE("swiglu");  // 计时：swiglu
    if (gate->shape() != up->shape() || gate->shape() != out->shape() ||
        gate->dtype() != up->dtype() || gate->dtype() != out->dtype() ||
        gate->deviceType() != up->deviceType() ||
        gate->deviceType() != out->deviceType() ||
        gate->deviceId() != up->deviceId() || gate->deviceId() != out->deviceId() ||
        !gate->isContiguous() || !up->isContiguous() || !out->isContiguous()) {
        throw std::runtime_error("swiglu: invalid input or output tensors");
    }
    llaisysDataType_t dtype = gate->dtype();
    size_t numel = gate->numel();

    // 并行区外先校验 dtype。
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_all<float>(gate->data(), up->data(), out->data(), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_all<fp16_t>(gate->data(), up->data(), out->data(), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_all<bf16_t>(gate->data(), up->data(), out->data(), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops
