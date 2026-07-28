#include "op.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t input_size = vals->numel();
    
    switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32: {
            const float* input_data = reinterpret_cast<const float*>(vals->data());
            float* max_val_data = reinterpret_cast<float*>(max_val->data());
            int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx->data());
            
            float max_value = input_data[0];
            int64_t max_index = 0;
            for (size_t i = 1; i < input_size; ++i) {
                if (input_data[i] > max_value) {
                    max_value = input_data[i];
                    max_index = static_cast<int64_t>(i);
                }
            }
            max_val_data[0] = max_value;
            max_idx_data[0] = max_index;
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const fp16_t* input_data = reinterpret_cast<const fp16_t*>(vals->data());
            fp16_t* max_val_data = reinterpret_cast<fp16_t*>(max_val->data());
            int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx->data());
            
            fp16_t max_value = input_data[0];
            float max_value_f = utils::cast<float>(max_value);
            int64_t max_index = 0;
            for (size_t i = 1; i < input_size; ++i) {
                float current = utils::cast<float>(input_data[i]);
                if (current > max_value_f) {
                    max_value = input_data[i];
                    max_value_f = current;
                    max_index = static_cast<int64_t>(i);
                }
            }
            max_val_data[0] = max_value;
            max_idx_data[0] = max_index;
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const bf16_t* input_data = reinterpret_cast<const bf16_t*>(vals->data());
            bf16_t* max_val_data = reinterpret_cast<bf16_t*>(max_val->data());
            int64_t* max_idx_data = reinterpret_cast<int64_t*>(max_idx->data());
            
            bf16_t max_value = input_data[0];
            float max_value_f = utils::cast<float>(max_value);
            int64_t max_index = 0;
            for (size_t i = 1; i < input_size; ++i) {
                float current = utils::cast<float>(input_data[i]);
                if (current > max_value_f) {
                    max_value = input_data[i];
                    max_value_f = current;
                    max_index = static_cast<int64_t>(i);
                }
            }
            max_val_data[0] = max_value;
            max_idx_data[0] = max_index;
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
