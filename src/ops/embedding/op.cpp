#include "op.hpp"
#include <cstring>

#include "../../utils.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    PROFILE("embedding");  // 计时：embedding
    if (index->shape().size() != 1 || weight->shape().size() != 2 ||
        out->shape().size() != 2 || index->dtype() != LLAISYS_DTYPE_I64 ||
        out->shape()[0] != index->shape()[0] ||
        out->shape()[1] != weight->shape()[1] ||
        out->dtype() != weight->dtype() ||
        index->deviceType() != weight->deviceType() ||
        index->deviceId() != weight->deviceId() ||
        out->deviceType() != weight->deviceType() ||
        out->deviceId() != weight->deviceId() || !index->isContiguous() ||
        !weight->isContiguous() || !out->isContiguous()) {
        throw std::runtime_error("embedding: invalid input or output tensors");
    }
    size_t out_rows = out->shape()[0];
    size_t out_cols = out->shape()[1];
    size_t weight_rows = weight->shape()[0];
    size_t weight_cols = weight->shape()[1];
    
    size_t elem_size = utils::dsize(out->dtype());
    
    const int64_t* indices = reinterpret_cast<const int64_t*>(index->data());
    const void* weight_data = weight->data();
    void* out_data = out->data();
    
    for (size_t i = 0; i < out_rows; ++i) {
        int64_t row_idx = indices[i];
        if (row_idx < 0 || static_cast<size_t>(row_idx) >= weight_rows) {
            throw std::out_of_range("embedding: index out of range");
        }
        const char* src_row = reinterpret_cast<const char*>(weight_data) + row_idx * weight_cols * elem_size;
        char* dst_row = reinterpret_cast<char*>(out_data) + i * out_cols * elem_size;
        std::memcpy(dst_row, src_row, weight_cols * elem_size);
    }
}
} // namespace llaisys::ops
