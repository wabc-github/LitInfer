#pragma once
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include <vector>

namespace llaisys {
namespace models {

// KV-Cache 结构体（每层的K/V缓存）
struct Qwen2KVCache {
    llaisysTensor_t k_cache{nullptr};  // shape: [max_seq_len, nkvh, dh]
    llaisysTensor_t v_cache{nullptr};  // shape: [max_seq_len, nkvh, dh]
    size_t cur_len{0};    // 当前缓存的token长度
};

// Qwen2模型核心类
class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
          const LlaisysQwen2Weights &weights,
          llaisysDeviceType_t device,
          const std::vector<int> &device_ids);
    ~Qwen2Model();


    // 初始化/释放KV-Cache
    void init_kv_cache();
    void free_kv_cache();

    // 单步推理（输入token_ids，返回下一个token）
    int64_t infer(int64_t* token_ids, size_t ntoken);

    // 单步推理带采样（输入token_ids，返回下一个token，使用top-k, top-p, temperature采样）
    int64_t infer_with_sampling(int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature);

private:
    void validate_weights() const;
    // 模型元信息
    LlaisysQwen2Meta _meta{};
    // 权重
    const LlaisysQwen2Weights *_weights{nullptr};
    // 设备信息
    llaisysDeviceType_t _device{LLAISYS_DEVICE_CPU};
    std::vector<int> _device_ids;
    // KV-Cache（nlayer层）
    std::vector<Qwen2KVCache> _kv_caches;

    // 核心算子调用（封装self_attention/linear/rms_norm等）
    llaisysTensor_t forward_embedding(int64_t* token_ids, size_t ntoken);
    llaisysTensor_t forward_attention(const llaisysTensor_t& x, size_t layer_idx, size_t seq_len);
    llaisysTensor_t forward_mlp(const llaisysTensor_t& x, size_t layer_idx);
    llaisysTensor_t forward_rms_norm(const llaisysTensor_t& x, const llaisysTensor_t& weight, float eps);
};

} // namespace models
} // namespace llaisys
