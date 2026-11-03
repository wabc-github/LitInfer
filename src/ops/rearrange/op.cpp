#include "op.hpp"
#include <cstring>

#include "../../utils.hpp"

namespace llaisys::ops {
template<typename T>
void rearrange_kernel(T* out_data, const T* in_data, const std::vector<size_t>& shape, 
                      const std::vector<ptrdiff_t>& out_strides, const std::vector<ptrdiff_t>& in_strides,
                      size_t ndim, size_t total_elements) {
    // 创建索引向量用于遍历所有元素
    std::vector<size_t> indices(ndim, 0);
    
    for (size_t elem_idx = 0; elem_idx < total_elements; ++elem_idx) {
        // 计算输入和输出的线性偏移
        ptrdiff_t in_offset = 0;
        ptrdiff_t out_offset = 0;
        
        for (size_t dim = 0; dim < ndim; ++dim) {
            in_offset += indices[dim] * in_strides[dim];
            out_offset += indices[dim] * out_strides[dim];
        }
        
        // 复制数据
        out_data[out_offset] = in_data[in_offset];
        
        // 更新多维索引
        size_t curr_dim = ndim;
        do {
            curr_dim--;
            indices[curr_dim]++;
            if (indices[curr_dim] >= shape[curr_dim]) {
                indices[curr_dim] = 0;
            } else {
                break;
            }
        } while (curr_dim > 0);
    }
}

void rearrange(tensor_t out, tensor_t in) {
    PROFILE("rearrange");  // 计时：rearrange
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    
    // 如果两个张量都是连续的，可以直接复制内存
    if (out->isContiguous() && in->isContiguous()) {
        size_t nbytes = out->numel() * out->elementSize();
        std::memcpy(out->data(), in->data(), nbytes);
        return;
    }
    
    // 获取张量信息
    auto dtype = in->dtype();
    auto shape = in->shape();
    auto in_strides = in->strides();
    auto out_strides = out->strides();
    size_t ndim = in->ndim();
    size_t total_elements = in->numel();
    
    // 根据数据类型调用相应的内核函数
    if (dtype == LLAISYS_DTYPE_F32) {
        rearrange_kernel<float>(
            reinterpret_cast<float*>(out->data()),
            reinterpret_cast<const float*>(in->data()),
            shape, out_strides, in_strides, ndim, total_elements
        );
    }
    else if (dtype == LLAISYS_DTYPE_F16) {
        rearrange_kernel<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t*>(out->data()),
            reinterpret_cast<const llaisys::fp16_t*>(in->data()),
            shape, out_strides, in_strides, ndim, total_elements
        );
    }
    else if (dtype == LLAISYS_DTYPE_BF16) {
        rearrange_kernel<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t*>(out->data()),
            reinterpret_cast<const llaisys::bf16_t*>(in->data()),
            shape, out_strides, in_strides, ndim, total_elements
        );
    }
    else {
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops
