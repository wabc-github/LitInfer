#pragma once
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include "../kv_cache/block_kv_cache.hpp"
#include <vector>
#include <mutex>

namespace llaisys {
namespace models {

// 兼容：连续 KV-Cache 结构（保留定义，分页路径不再使用）。
struct Qwen2KVCache {
    llaisysTensor_t k_cache{nullptr};  // shape: [max_seq_len, nkvh, dh]
    llaisysTensor_t v_cache{nullptr};  // shape: [max_seq_len, nkvh, dh]
    size_t cur_len{0};    // 当前缓存的token长度
};

// Qwen2模型核心类
class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
          LlaisysQwen2Weights &weights,
          llaisysDeviceType_t device,
          const std::vector<int> &device_ids);
    ~Qwen2Model();


    // 初始化/释放KV-Cache
    void init_kv_cache();
    void free_kv_cache();

    // 开启 INT8 weight-only 量化（在权重加载完成后调用一次）。
    // 内部把每个线性层权重按行对称量化为 int8（销毁原 fp16 权重、真正省内存），
    // 之后所有 linear 走"在线反量化"的 quant_linear 路径。
    void quantize_int8();

    // 单步推理（输入token_ids，返回下一个token）
    int64_t infer(int64_t* token_ids, size_t ntoken);

    // 单步推理带采样（输入token_ids，返回下一个token，使用top-k, top-p, temperature采样）
    int64_t infer_with_sampling(int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature);

    // ---- 连续批处理引擎（P2-2）----
    // 前提：已 init_kv_cache（共享池由其创建）。返回值见实现。
    // 添加一条请求（prompt token），返回该请求的 id（即数组下标，长期有效）。
    int64_t engine_add_request(const int64_t* tokens, size_t n, int64_t max_new_tokens);
    // 推进一次调度：所有未完成序列同时做一次前向（新序列 prefill、老序列 decode）。
    // 返回本步新完成的请求数。
    size_t engine_step(int top_k, float top_p, float temperature);
    bool engine_has_unfinished() const;
    size_t engine_num_seqs() const;
    bool engine_is_finished(int64_t id) const;
    // 取已完成请求的输出 token（不含 prompt，仅生成部分）。返回写入数；未完成返回 0。
    size_t engine_get_output(int64_t id, int64_t* out, size_t cap) const;
    // 请求是否因 EOS 结束（1）还是仍在/异常（0）。
    bool engine_is_eos_finished(int64_t id) const;

private:
    void validate_weights() const;
    // 量化单个 fp 权重：返回 int8 权重 + fp32 每行 scale。
    // destroy_src=true 时销毁源 fp 权重（默认）；tied embedding 场景需要 false。
    struct QuantResult { llaisysTensor_t qw{nullptr}; llaisysTensor_t scale{nullptr}; };
    QuantResult quantize_one(llaisysTensor_t fp_weight, bool destroy_src = true);
    // 释放本类持有的所有 scale（int8 权重由外层 weights 结构负责释放）。
    void free_quant_scales();

    // 模型元信息
    LlaisysQwen2Meta _meta{};
    // 权重（非 const：量化时需要把权重指针替换为 int8 版本）
    LlaisysQwen2Weights *_weights{nullptr};
    // 设备信息
    llaisysDeviceType_t _device{LLAISYS_DEVICE_CPU};
    std::vector<int> _device_ids;
    // ---- 分页 KV-Cache（P2）----
    // 每层一个物理 block 池 + 一张逻辑 block table（单序列时各层逻辑相同）。
    std::vector<KVBlockPool> _paged_pools;
    std::vector<SequenceKVMeta> _paged_metas;
    std::vector<llaisysTensor_t> _block_table_tensors;  // 每层 int32 块表
    int _num_kv_blocks{0};  // 每个物理池的块数（KV 容量预算）

    // ---- INT8 量化状态 ----
    bool _quantized{false};
    // 每个线性层的 per-row scale（fp32），与 weights 结构中的 int8 权重一一对应。
    std::vector<llaisysTensor_t> _scale_attn_q;
    std::vector<llaisysTensor_t> _scale_attn_k;
    std::vector<llaisysTensor_t> _scale_attn_v;
    std::vector<llaisysTensor_t> _scale_attn_o;
    std::vector<llaisysTensor_t> _scale_mlp_gate;
    std::vector<llaisysTensor_t> _scale_mlp_up;
    std::vector<llaisysTensor_t> _scale_mlp_down;
    llaisysTensor_t _scale_lm_head{nullptr};

    // 核心算子调用（封装self_attention/linear/rms_norm等）
    llaisysTensor_t forward_embedding(int64_t* token_ids, size_t ntoken);
    llaisysTensor_t forward_attention(const llaisysTensor_t& x, size_t layer_idx, size_t seq_len);
    llaisysTensor_t forward_mlp(const llaisysTensor_t& x, size_t layer_idx);
    llaisysTensor_t forward_rms_norm(const llaisysTensor_t& x, const llaisysTensor_t& weight, float eps);

    // ---- 连续批处理引擎状态（P2-2）----
    struct EngineSeq {
        bool finished{false};
        bool eos_finished{false};
        bool is_new{true};                 // 尚未做 prefill
        int64_t max_new_tokens{512};       // 生成长度上限
        size_t num_tokens{0};               // 已在各层池中的 KV 长度
        std::vector<int64_t> prompt;        // prompt token
        std::vector<int64_t> generated;     // 已生成 token
        // 每层的逻辑块->物理块映射：block_tables[layer][logical_block]
        std::vector<std::vector<int32_t>> block_tables;
    };
    std::vector<EngineSeq> _eseqs;
    // 保护 _eseqs：engine_add_request 由 HTTP 线程调用（push_back 可能重分配），
    // engine_step 由后台调度线程调用（遍历/修改 _eseqs），必须互斥。
    mutable std::mutex _engine_mtx;
    // 为某序列在所有层池分配满足 new_len 的新块（仅在块不够时分配）。
    void engine_ensure_blocks(EngineSeq& s, size_t new_len);
    // 本步逐 token 把 RoPE 后的 K/V 写入共享池。
    void engine_write_kv(size_t layer, const llaisysTensor_t& k_rot,
                         const llaisysTensor_t& v_rot,
                         const std::vector<size_t>& active,
                         const std::vector<size_t>& qoff,
                         const std::vector<size_t>& qlens,
                         const int64_t* positions);
};

} // namespace models
} // namespace llaisys
