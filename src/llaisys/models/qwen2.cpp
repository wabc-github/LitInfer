#include "qwen2.hpp"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "../../utils.hpp"
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <iostream>

namespace llaisys {
namespace models {

static bool qwen2_trace_enabled() {
    static const bool enabled = [] {
        const std::string value = read_environment("LLAISYS_TRACE");
        return value == "1" || value == "true" || value == "TRUE" ||
               value == "on" || value == "ON";
    }();
    return enabled;
}

#define QWEN2_TRACE if (!qwen2_trace_enabled()) {} else std::cerr

static std::vector<float> tensor_to_float_vector(llaisysTensor_t tensor, size_t count) {
    std::vector<float> values(count);
    auto dtype = tensorGetDataType(tensor);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float *data = reinterpret_cast<const float *>(tensorGetData(tensor));
        std::copy(data, data + count, values.begin());
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *data = reinterpret_cast<const fp16_t *>(tensorGetData(tensor));
        for (size_t i = 0; i < count; ++i) values[i] = utils::cast<float>(data[i]);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *data = reinterpret_cast<const bf16_t *>(tensorGetData(tensor));
        for (size_t i = 0; i < count; ++i) values[i] = utils::cast<float>(data[i]);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    return values;
}

void Qwen2Model::validate_weights() const {
    QWEN2_TRACE << "[TRACE] validate_weights enter weights=" << _weights << std::endl;
    if (!_weights) throw std::runtime_error("Qwen2 weights pointer is null");
    QWEN2_TRACE << "[TRACE] top weights in_embed=" << _weights->in_embed
                << " out_embed=" << _weights->out_embed
                << " out_norm_w=" << _weights->out_norm_w << std::endl;
    if (!_weights->in_embed) throw std::runtime_error("Missing weight: model.embed_tokens.weight");
    if (!_weights->out_embed) throw std::runtime_error("Missing weight: lm_head.weight or tied embed_tokens.weight");
    if (!_weights->out_norm_w) throw std::runtime_error("Missing weight: model.norm.weight");

    auto require_array = [](const char *name, llaisysTensor_t *arr) {
        if (!arr) {
            throw std::runtime_error(std::string("Missing weight array: ") + name);
        }
    };
    require_array("attn_norm_w", _weights->attn_norm_w);
    require_array("attn_q_w", _weights->attn_q_w);
    require_array("attn_k_w", _weights->attn_k_w);
    require_array("attn_v_w", _weights->attn_v_w);
    require_array("attn_o_w", _weights->attn_o_w);
    require_array("mlp_norm_w", _weights->mlp_norm_w);
    require_array("mlp_gate_w", _weights->mlp_gate_w);
    require_array("mlp_up_w", _weights->mlp_up_w);
    require_array("mlp_down_w", _weights->mlp_down_w);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto layer_name = [i](const char *name) {
            return std::string("Missing layer ") + std::to_string(i) + " weight: " + name;
        };
        if (!_weights->attn_norm_w[i]) throw std::runtime_error(layer_name("input_layernorm.weight"));
        if (!_weights->attn_q_w[i]) throw std::runtime_error(layer_name("self_attn.q_proj.weight"));
        if (!_weights->attn_k_w[i]) throw std::runtime_error(layer_name("self_attn.k_proj.weight"));
        if (!_weights->attn_v_w[i]) throw std::runtime_error(layer_name("self_attn.v_proj.weight"));
        if (!_weights->attn_o_w[i]) throw std::runtime_error(layer_name("self_attn.o_proj.weight"));
        if (!_weights->mlp_norm_w[i]) throw std::runtime_error(layer_name("post_attention_layernorm.weight"));
        if (!_weights->mlp_gate_w[i]) throw std::runtime_error(layer_name("mlp.gate_proj.weight"));
        if (!_weights->mlp_up_w[i]) throw std::runtime_error(layer_name("mlp.up_proj.weight"));
        if (!_weights->mlp_down_w[i]) throw std::runtime_error(layer_name("mlp.down_proj.weight"));
    }
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
             LlaisysQwen2Weights &weights,
             llaisysDeviceType_t device,
             const std::vector<int> &device_ids)
    : _meta(meta),
      _weights(&weights),
      _device(device),
      _device_ids(device_ids){
      }

Qwen2Model::~Qwen2Model() {
    free_quant_scales();
    free_kv_cache();
}



void Qwen2Model::init_kv_cache() {
    free_kv_cache();
    // 物理块数：默认覆盖单序列 maxseq，可用 LLAISYS_NUM_BLOCKS 调整容量。
    int num_blocks = static_cast<int>(
        (_meta.maxseq + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS);
    // max_position_embeddings 常达 13 万（RoPE 上限），并非实际 KV 内存预算；
    // 若按它预分配，每层要开 8192 个 block（数 GB），在普通机器上直接 OOM。
    // 默认只预留覆盖常见对话长度的块数，需要更大容量时用 LLAISYS_NUM_BLOCKS 显式指定。
    constexpr int kDefaultMaxBlocks = 256;  // 256 * 16 = 4096 tokens
    if (num_blocks > kDefaultMaxBlocks) num_blocks = kDefaultMaxBlocks;
    std::string env = read_environment("LLAISYS_NUM_BLOCKS");
    if (!env.empty()) {
        int v = std::atoi(env.c_str());
        if (v > 0) num_blocks = v;
    }
    _num_kv_blocks = num_blocks;

    _paged_pools.resize(_meta.nlayer);
    _paged_metas.resize(_meta.nlayer);
    _block_table_tensors.assign(_meta.nlayer, nullptr);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        // 每层一个物理 block 池（4D 张量）。
        _paged_pools[i].init(num_blocks, _meta.nkvh, _meta.dh,
                             _meta.dtype, _device, _device_ids[0]);
        _paged_metas[i].seq_len = 0;
        _paged_metas[i].block_table.clear();
        // 每层一张 int32 block table 张量。
        size_t bts[] = {static_cast<size_t>(num_blocks)};
        _block_table_tensors[i] = tensorCreate(
            bts, 1, LLAISYS_DTYPE_I32, _device, _device_ids[0]);
    }
}

void Qwen2Model::free_kv_cache() {
    for (size_t i = 0; i < _paged_pools.size(); ++i) {
        if (i < _paged_metas.size()) _paged_metas[i].release(_paged_pools[i]);
        _paged_pools[i].destroy();
        if (i < _block_table_tensors.size() && _block_table_tensors[i]) {
            tensorDestroy(_block_table_tensors[i]);
            _block_table_tensors[i] = nullptr;
        }
    }
    _paged_pools.clear();
    _paged_metas.clear();
    _block_table_tensors.clear();
    _num_kv_blocks = 0;
}

// ------------------------------------------------------------------
// INT8 weight-only 量化
// ------------------------------------------------------------------

// 量化单个 fp 权重：fp [N,K] -> int8 [N,K] + scale [N]。
Qwen2Model::QuantResult Qwen2Model::quantize_one(llaisysTensor_t fp_weight,
                                                 bool destroy_src) {
    if (!fp_weight) throw std::runtime_error("quantize: weight is null");
    if (tensorGetNdim(fp_weight) != 2)
        throw std::runtime_error("quantize: weight must be rank-2 [N,K]");

    // 读原权重 shape / 设备。
    size_t sh[2] = {0, 0};
    tensorGetShape(fp_weight, sh);
    const size_t N = sh[0], K = sh[1];
    const llaisysDeviceType_t dev = tensorGetDeviceType(fp_weight);
    const int devid = tensorGetDeviceId(fp_weight);

    // 创建 int8 权重与 per-row scale。
    QuantResult r;
    size_t qsh[2] = {N, K};
    r.qw = tensorCreate(qsh, 2, LLAISYS_DTYPE_I8, dev, devid);
    size_t ssh[1] = {N};
    r.scale = tensorCreate(ssh, 1, LLAISYS_DTYPE_F32, dev, devid);

    // 执行量化。
    llaisysQuantizeWeight(r.qw, r.scale, fp_weight);

    // 销毁源 fp 权重（默认）。调用方负责把 weights 结构指针替换为 r.qw。
    if (destroy_src) tensorDestroy(fp_weight);
    return r;
}

// 释放所有 scale（模型类析构时）。
void Qwen2Model::free_quant_scales() {
    auto destroy_vec = [](std::vector<llaisysTensor_t> &v) {
        for (llaisysTensor_t t : v) {
            if (t) tensorDestroy(t);
        }
        v.clear();
    };
    destroy_vec(_scale_attn_q);
    destroy_vec(_scale_attn_k);
    destroy_vec(_scale_attn_v);
    destroy_vec(_scale_attn_o);
    destroy_vec(_scale_mlp_gate);
    destroy_vec(_scale_mlp_up);
    destroy_vec(_scale_mlp_down);
    if (_scale_lm_head) {
        tensorDestroy(_scale_lm_head);
        _scale_lm_head = nullptr;
    }
}

// 开启 INT8 量化：量化所有线性层，权重 fp16 -> int8。
void Qwen2Model::quantize_int8() {
    if (_quantized) return;
    validate_weights();
    const size_t nlayer = _meta.nlayer;

    // 预留 scale 容器。
    _scale_attn_q.resize(nlayer);
    _scale_attn_k.resize(nlayer);
    _scale_attn_v.resize(nlayer);
    _scale_attn_o.resize(nlayer);
    _scale_mlp_gate.resize(nlayer);
    _scale_mlp_up.resize(nlayer);
    _scale_mlp_down.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        // attention 的 q/k/v（有 bias，bias 保持 fp16 不量化）。
        QuantResult rq = quantize_one(_weights->attn_q_w[i]);
        _weights->attn_q_w[i] = rq.qw;  _scale_attn_q[i] = rq.scale;
        QuantResult rk = quantize_one(_weights->attn_k_w[i]);
        _weights->attn_k_w[i] = rk.qw;  _scale_attn_k[i] = rk.scale;
        QuantResult rv = quantize_one(_weights->attn_v_w[i]);
        _weights->attn_v_w[i] = rv.qw;  _scale_attn_v[i] = rv.scale;
        // attention 输出投影 o（无 bias）。
        QuantResult ro = quantize_one(_weights->attn_o_w[i]);
        _weights->attn_o_w[i] = ro.qw;  _scale_attn_o[i] = ro.scale;
        // MLP gate/up/down（均无 bias）。
        QuantResult rg = quantize_one(_weights->mlp_gate_w[i]);
        _weights->mlp_gate_w[i] = rg.qw; _scale_mlp_gate[i] = rg.scale;
        QuantResult ru = quantize_one(_weights->mlp_up_w[i]);
        _weights->mlp_up_w[i] = ru.qw;   _scale_mlp_up[i] = ru.scale;
        QuantResult rd = quantize_one(_weights->mlp_down_w[i]);
        _weights->mlp_down_w[i] = rd.qw; _scale_mlp_down[i] = rd.scale;
    }

    // lm_head（输出投影）。
    // tied 情况：out_embed 与 in_embed 是同一 fp tensor，而 in_embed 还要用于
    // embedding gather（不能量化）。此时量化一份独立 int8 给 lm_head，保留 in_embed。
    const bool tied = (_weights->out_embed == _weights->in_embed);
    QuantResult rl = quantize_one(_weights->out_embed, /*destroy_src=*/!tied);
    _weights->out_embed = rl.qw;  // tied 时 out_embed 指向独立 int8，in_embed 仍为 fp
    _scale_lm_head = rl.scale;

    _quantized = true;
}

// 核心推理逻辑
int64_t Qwen2Model::infer(int64_t* token_ids, size_t ntoken) {
    QWEN2_TRACE << "[TRACE] Qwen2Model::infer enter ntoken=" << ntoken << std::endl;
    validate_weights();
    QWEN2_TRACE << "[TRACE] weights validated" << std::endl;
    if (!token_ids || ntoken == 0) throw std::runtime_error("infer: token_ids must not be empty");
    ScopedProfilePhase profile_phase(ntoken > 1 ? "prefill" : "decode");
    // 1. Embedding层
    QWEN2_TRACE << "[TRACE] forward_embedding start" << std::endl;
    llaisysTensor_t x = forward_embedding(token_ids, ntoken);
    QWEN2_TRACE << "[TRACE] forward_embedding done x=" << x << std::endl;
    // 2. 遍历所有层
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        QWEN2_TRACE << "[TRACE] layer " << i << " start" << std::endl;
        // 2.1 RMSNorm（attention输入）
        llaisysTensor_t x_norm = forward_rms_norm(x, _weights->attn_norm_w[i], _meta.epsilon);
        // 2.2 SelfAttention（带KV-Cache）
        llaisysTensor_t attn_out = forward_attention(x_norm, i, ntoken);
        // 2.3 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim = tensorGetNdim(x);
        size_t* x_shape = new size_t[x_ndim];
        tensorGetShape(x, x_shape);
        llaisysDataType_t x_dtype = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
        int x_device_id = tensorGetDeviceId(x);
        
        llaisysTensor_t x_residual_result = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
        llaisysAdd(x_residual_result, x, attn_out);
        tensorDestroy(x);
        x = x_residual_result;
        delete[] x_shape; // 释放动态分配的形状数组
        
        tensorDestroy(x_norm);
        tensorDestroy(attn_out);
        
        // 2.4 RMSNorm（MLP输入）
        llaisysTensor_t x_mlp_norm = forward_rms_norm(x, _weights->mlp_norm_w[i], _meta.epsilon);
        // 2.5 MLP（SwiGLU）
        llaisysTensor_t mlp_out = forward_mlp(x_mlp_norm, i);
        // 2.6 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim2 = tensorGetNdim(x);
        size_t* x_shape2 = new size_t[x_ndim2];
        tensorGetShape(x, x_shape2);
        llaisysDataType_t x_dtype2 = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type2 = tensorGetDeviceType(x);
        int x_device_id2 = tensorGetDeviceId(x);
        
        llaisysTensor_t x_mlp_result = tensorCreate(x_shape2, x_ndim2, x_dtype2, x_device_type2, x_device_id2);
        llaisysAdd(x_mlp_result, x, mlp_out);
        tensorDestroy(x);
        x = x_mlp_result;
        delete[] x_shape2; // 释放动态分配的形状数组
        
        tensorDestroy(x_mlp_norm);
        tensorDestroy(mlp_out);
    }
    // 3. 最后的RMSNorm
    llaisysTensor_t x_final = forward_rms_norm(x, _weights->out_norm_w, _meta.epsilon);
    // 4. 输出Embedding（linear层）
    // 获取x_final的形状信息用于创建logits张量
    size_t x_final_ndim = tensorGetNdim(x_final);
    size_t* x_final_shape = new size_t[x_final_ndim];
    tensorGetShape(x_final, x_final_shape);
    // 修改最后一维的大小为词汇表大小
    x_final_shape[x_final_ndim-1] = _meta.voc;
    llaisysDataType_t x_final_dtype = tensorGetDataType(x_final);
    llaisysDeviceType_t x_final_device_type = tensorGetDeviceType(x_final);
    int x_final_device_id = tensorGetDeviceId(x_final);
    
    llaisysTensor_t logits = tensorCreate(x_final_shape, x_final_ndim, x_final_dtype, x_final_device_type, x_final_device_id);
    delete[] x_final_shape;
    
    if (_quantized)
        llaisysQuantLinear(logits, x_final, _weights->out_embed, _scale_lm_head, nullptr);
    else
        llaisysLinear(logits, x_final, _weights->out_embed, nullptr);
    
    // 5. ArgMax取最大概率token
    // 为ArgMax创建输出张量
    size_t argmax_shape[] = {1}; 
    llaisysTensor_t max_idx = tensorCreate(argmax_shape, 1, LLAISYS_DTYPE_I64, x_final_device_type, x_final_device_id);
    llaisysTensor_t max_val = tensorCreate(argmax_shape, 1, x_final_dtype, x_final_device_type, x_final_device_id);
    llaisysTensor_t last_logits_2d = tensorSlice(logits, 0, ntoken - 1, ntoken);
    size_t last_logits_shape[] = {_meta.voc};
    llaisysTensor_t last_logits = tensorView(last_logits_2d, last_logits_shape, 1);
    
    llaisysArgmax(max_idx, max_val, last_logits);  
    
    int64_t next_token = *reinterpret_cast<int64_t*>(tensorGetData(max_idx));
    
    // 6. 释放临时张量
    tensorDestroy(x);
    tensorDestroy(x_final);
    tensorDestroy(logits);
    tensorDestroy(last_logits_2d);
    tensorDestroy(last_logits);
    tensorDestroy(max_idx);
    tensorDestroy(max_val);
    return next_token;
}

// 添加采样推理的实现
int64_t Qwen2Model::infer_with_sampling(int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature) {
    QWEN2_TRACE << "[TRACE] Qwen2Model::infer_with_sampling enter ntoken=" << ntoken << std::endl;
    validate_weights();
    if (!token_ids || ntoken == 0) throw std::runtime_error("infer_with_sampling: token_ids must not be empty");
    ScopedProfilePhase profile_phase(ntoken > 1 ? "prefill" : "decode");
    // 1. Embedding层
    llaisysTensor_t x = forward_embedding(token_ids, ntoken);
    // 2. 遍历所有层
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        // 2.1 RMSNorm（attention输入）
        llaisysTensor_t x_norm = forward_rms_norm(x, _weights->attn_norm_w[i], _meta.epsilon);
        // 2.2 SelfAttention（带KV-Cache）
        llaisysTensor_t attn_out = forward_attention(x_norm, i, ntoken);
        // 2.3 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim = tensorGetNdim(x);
        size_t* x_shape = new size_t[x_ndim];
        tensorGetShape(x, x_shape);
        llaisysDataType_t x_dtype = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
        int x_device_id = tensorGetDeviceId(x);
        
        llaisysTensor_t x_residual_result = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
        llaisysAdd(x_residual_result, x, attn_out);
        tensorDestroy(x);
        x = x_residual_result;
        delete[] x_shape; // 释放动态分配的形状数组
        
        tensorDestroy(x_norm);
        tensorDestroy(attn_out);
        
        // 2.4 RMSNorm（MLP输入）
        llaisysTensor_t x_mlp_norm = forward_rms_norm(x, _weights->mlp_norm_w[i], _meta.epsilon);
        // 2.5 MLP（SwiGLU）
        llaisysTensor_t mlp_out = forward_mlp(x_mlp_norm, i);
        // 2.6 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim2 = tensorGetNdim(x);
        size_t* x_shape2 = new size_t[x_ndim2];
        tensorGetShape(x, x_shape2);
        llaisysDataType_t x_dtype2 = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type2 = tensorGetDeviceType(x);
        int x_device_id2 = tensorGetDeviceId(x);
        
        llaisysTensor_t x_mlp_result = tensorCreate(x_shape2, x_ndim2, x_dtype2, x_device_type2, x_device_id2);
        llaisysAdd(x_mlp_result, x, mlp_out);
        tensorDestroy(x);
        x = x_mlp_result;
        delete[] x_shape2; // 释放动态分配的形状数组
        
        tensorDestroy(x_mlp_norm);
        tensorDestroy(mlp_out);
    }
    // 3. 最后的RMSNorm
    llaisysTensor_t x_final = forward_rms_norm(x, _weights->out_norm_w, _meta.epsilon);
    // 4. 输出Embedding（linear层）
    // 获取x_final的形状信息用于创建logits张量
    size_t x_final_ndim = tensorGetNdim(x_final);
    size_t* x_final_shape = new size_t[x_final_ndim];
    tensorGetShape(x_final, x_final_shape);
    // 修改最后一维的大小为词汇表大小
    x_final_shape[x_final_ndim-1] = _meta.voc;
    llaisysDataType_t x_final_dtype = tensorGetDataType(x_final);
    llaisysDeviceType_t x_final_device_type = tensorGetDeviceType(x_final);
    int x_final_device_id = tensorGetDeviceId(x_final);
    
    llaisysTensor_t logits = tensorCreate(x_final_shape, x_final_ndim, x_final_dtype, x_final_device_type, x_final_device_id);
    delete[] x_final_shape;
    
    if (_quantized)
        llaisysQuantLinear(logits, x_final, _weights->out_embed, _scale_lm_head, nullptr);
    else
        llaisysLinear(logits, x_final, _weights->out_embed, nullptr);
    llaisysTensor_t last_logits_2d = tensorSlice(logits, 0, ntoken - 1, ntoken);
    size_t last_logits_shape[] = {_meta.voc};
    llaisysTensor_t last_logits = tensorView(last_logits_2d, last_logits_shape, 1);

    std::vector<float> logits_data = tensor_to_float_vector(last_logits, _meta.voc);
    float safe_temperature = temperature > 0.0f ? temperature : 1.0f;
    for (auto &logit : logits_data) {
        logit /= safe_temperature;
    }

    // 6. Top-K 和 Top-P 采样
    // 这里需要实现Top-K和Top-P采样的逻辑
    // 为了实现Top-K和Top-P，我们需要获取logits数据并实现采样算法
    // 创建索引数组并根据logits值排序
    std::vector<std::pair<float, int>> logit_indices;
    for (size_t i = 0; i < _meta.voc; ++i) {
        logit_indices.push_back({logits_data[i], static_cast<int>(i)});
    }
    
    // 按logits值降序排序
    std::sort(logit_indices.begin(), logit_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // 应用Top-K
    if (top_k > 0 && top_k < static_cast<int>(_meta.voc)) {
        logit_indices.resize(top_k);
    }
    
    // 应用Top-P (nucleus sampling)
    if (top_p > 0.0f && top_p < 1.0f) {
        // 计算softmax概率
        std::vector<float> probs;
        float max_logit = logit_indices[0].first;  // 用于数值稳定性
        float sum_probs = 0.0f;
        
        for (auto& item : logit_indices) {
            float prob = std::exp(item.first - max_logit);
            probs.push_back(prob);
            sum_probs += prob;
        }
        
        // 归一化概率
        for (auto& prob : probs) {
            prob /= sum_probs;
        }
        
        // 应用Top-P - 选择累积概率超过top_p的最小集合
        float cum_prob = 0.0f;
        size_t cutoff_idx = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum_prob += probs[i];
            if (cum_prob >= top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        if (cutoff_idx > 0) {
            logit_indices.resize(cutoff_idx);
            probs.resize(cutoff_idx);
        }
    }
    
    // 从筛选后的候选中随机采样
    // 使用标准库的随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // 重新计算筛选后候选的softmax概率
    std::vector<float> final_probs;
    if (!logit_indices.empty()) {
        float max_logit = logit_indices[0].first;
        float sum_probs = 0.0f;
        
        for (auto& item : logit_indices) {
            float prob = std::exp(item.first - max_logit);
            final_probs.push_back(prob);
            sum_probs += prob;
        }
        
        // 归一化
        for (auto& prob : final_probs) {
            prob /= sum_probs;
        }
        
        // 使用累积概率进行随机采样
        std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
        int selected_idx = dist(gen);
        
        int64_t next_token = logit_indices[selected_idx].second;
        
        // 6. 释放临时张量
        tensorDestroy(x);
        tensorDestroy(x_final);
        tensorDestroy(logits);
        tensorDestroy(last_logits_2d);
        tensorDestroy(last_logits);
        
        return next_token;
    } else {
        // 如果没有候选词，则返回第一个词（退化情况）
        tensorDestroy(x);
        tensorDestroy(x_final);
        tensorDestroy(logits);
        tensorDestroy(last_logits_2d);
        tensorDestroy(last_logits);
        return 0;
    }
}

// 其他辅助函数（forward_embedding/forward_attention/forward_mlp/forward_rms_norm）

llaisysTensor_t Qwen2Model::forward_embedding(int64_t* token_ids, size_t ntoken) {
    llaisysTensor_t index_tensor = tensorCreate(&ntoken, 1, LLAISYS_DTYPE_I64, _device, _device_ids[0]);
    tensorLoad(index_tensor, token_ids);
    

    size_t embedding_shape[2] = {ntoken, _meta.hs};
    llaisysTensor_t out = tensorCreate(embedding_shape, 2, _meta.dtype, _device, _device_ids[0]);
    llaisysEmbedding(out, index_tensor, _weights->in_embed);
    tensorDestroy(index_tensor);
    return out;
}


llaisysTensor_t Qwen2Model::forward_attention(const llaisysTensor_t& x, size_t layer_idx, size_t seq_len) {
    (void)seq_len;
    // 获取当前层的权重参数
    llaisysTensor_t wq = _weights->attn_q_w[layer_idx];
    llaisysTensor_t wk = _weights->attn_k_w[layer_idx];
    llaisysTensor_t wv = _weights->attn_v_w[layer_idx];
    llaisysTensor_t wo = _weights->attn_o_w[layer_idx];

    // 分页物理池与本序列块表（单序列，每层一张）。
    auto& pool = _paged_pools[layer_idx];
    auto& meta = _paged_metas[layer_idx];
    const size_t cur_len = meta.seq_len;

    // 获取x的形状信息
    size_t x_ndim = tensorGetNdim(x);
    size_t* x_shape = new size_t[x_ndim];
    tensorGetShape(x, x_shape);
    llaisysDataType_t x_dtype = tensorGetDataType(x);
    llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
    int x_device_id = tensorGetDeviceId(x);
    const size_t q_len = x_shape[0];

    if (!pool.k_pool || !pool.v_pool) {
        delete[] x_shape;
        throw std::runtime_error("Paged KV pool is not initialized");
    }
    if (cur_len > _meta.maxseq || q_len > _meta.maxseq - cur_len) {
        delete[] x_shape;
        throw std::runtime_error("Sequence length exceeds maximum allowed length");
    }
    
    // 计算Q、K、V的输出形状
    // Q: [q_len, num_heads, head_dim]，num_heads = _meta.nh，head_dim = _meta.dh
    size_t q_shape[3] = {q_len, _meta.nh, _meta.dh};
    // K/V: [q_len, num_key_value_heads, head_dim]
    size_t k_shape[3] = {q_len, _meta.nkvh, _meta.dh};
    size_t v_shape[3] = {q_len, _meta.nkvh, _meta.dh};

    // 创建Q、K、V张量
    llaisysTensor_t q = tensorCreate(q_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t k = tensorCreate(k_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t v = tensorCreate(v_shape, 3, x_dtype, x_device_type, x_device_id);
    
    // 计算Q、K、V（量化模式走融合反量化的 quant_linear，bias 保持 fp16）
    if (_quantized) {
        llaisysQuantLinear(q, x, wq, _scale_attn_q[layer_idx], _weights->attn_q_b[layer_idx]);
        llaisysQuantLinear(k, x, wk, _scale_attn_k[layer_idx], _weights->attn_k_b[layer_idx]);
        llaisysQuantLinear(v, x, wv, _scale_attn_v[layer_idx], _weights->attn_v_b[layer_idx]);
    } else {
        llaisysLinear(q, x, wq, _weights->attn_q_b[layer_idx]);
        llaisysLinear(k, x, wk, _weights->attn_k_b[layer_idx]);
        llaisysLinear(v, x, wv, _weights->attn_v_b[layer_idx]);
    }

    std::vector<int64_t> pos_ids_vec(q_len);
    for (size_t token_idx = 0; token_idx < q_len; ++token_idx) {
        pos_ids_vec[token_idx] = static_cast<int64_t>(cur_len + token_idx);
    }
    size_t pos_shape[] = {x_shape[0]};
    llaisysTensor_t pos_ids = tensorCreate(pos_shape, 1, LLAISYS_DTYPE_I64, x_device_type, x_device_id);
    tensorLoad(pos_ids, pos_ids_vec.data());

    llaisysTensor_t q_rope = tensorCreate(q_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t k_rope = tensorCreate(k_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysROPE(q_rope, q, pos_ids, _meta.theta);
    llaisysROPE(k_rope, k, pos_ids, _meta.theta);
    
    
    // 根据数据类型计算元素大小
    size_t element_size;
    switch (_meta.dtype) {
        case LLAISYS_DTYPE_F16:
        case LLAISYS_DTYPE_BF16:
            element_size = sizeof(uint16_t);
            break;
        case LLAISYS_DTYPE_F32:
            element_size = sizeof(float);
            break;
        default:
            delete[] x_shape;
            throw std::runtime_error("Unsupported dtype for KV cache");
    }
    const size_t elements_per_token = _meta.nkvh * _meta.dh;

    // 为新 token 预留物理块（懒分页），池满则报错。
    if (!meta.reserve_for_tokens(q_len, pool)) {
        delete[] x_shape;
        throw std::runtime_error("Paged KV pool is full; increase LLAISYS_NUM_BLOCKS");
    }

    // 逐 token 把 RoPE 后的 K/V 写入物理池（按 block table 寻址）。
    void* k_pool_ptr = tensorGetData(pool.k_pool);
    void* v_pool_ptr = tensorGetData(pool.v_pool);
    void* k_ptr = tensorGetData(k_rope);
    void* v_ptr = tensorGetData(v);
    for (size_t token_idx = 0; token_idx < q_len; ++token_idx) {
        size_t t = cur_len + token_idx;
        int phys = 0, off = 0;
        meta.map_position(t, phys, off);
        // 物理池该 token 起始偏移（layout: [num_blocks, block, nkvh, dh]）。
        size_t dst_base = ((static_cast<size_t>(phys) * KV_BLOCK_TOKENS + off)
                           * _meta.nkvh) * _meta.dh;
        size_t src_base = token_idx * elements_per_token;
        memcpy((char*)k_pool_ptr + dst_base * element_size,
               (char*)k_ptr + src_base * element_size,
               elements_per_token * element_size);
        memcpy((char*)v_pool_ptr + dst_base * element_size,
               (char*)v_ptr + src_base * element_size,
               elements_per_token * element_size);
    }

    // 把逻辑 block table 同步到 int32 张量（供算子按物理块寻址）。
    {
        std::vector<int32_t> bt(_num_kv_blocks, 0);
        for (size_t li = 0; li < meta.block_table.size(); ++li)
            bt[li] = static_cast<int32_t>(meta.block_table[li]);
        tensorLoad(_block_table_tensors[layer_idx], bt.data());
    }

    meta.seq_len += q_len;

    // PagedAttention：按 block table 寻址的 online softmax，不落地连续 score。
    size_t attn_out_shape[3] = {q_len, _meta.nh, _meta.dh};
    llaisysTensor_t temp_attn = tensorCreate(attn_out_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysPagedSelfAttention(temp_attn, q_rope, pool.k_pool, pool.v_pool,
        _block_table_tensors[layer_idx], meta.seq_len,
        1.0f / std::sqrt(static_cast<float>(_meta.dh)));
    
    // 应用输出投影
    llaisysTensor_t output = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    size_t attn_view_shape[] = {q_len, _meta.hs};
    llaisysTensor_t temp_attn_2d = tensorView(temp_attn, attn_view_shape, 2);
    if (_quantized)
        llaisysQuantLinear(output, temp_attn_2d, wo, _scale_attn_o[layer_idx], nullptr);
    else
        llaisysLinear(output, temp_attn_2d, wo, nullptr);
    
    // 清理临时张量（物理池张量保留）
    delete[] x_shape;
    tensorDestroy(q);
    tensorDestroy(k);  // K已经被复制到物理池，可以安全销毁
    tensorDestroy(v);  // V已经被复制到物理池，可以安全销毁
    tensorDestroy(q_rope);
    tensorDestroy(k_rope);
    tensorDestroy(pos_ids);
    tensorDestroy(temp_attn_2d);
    tensorDestroy(temp_attn);
    
    return output;
}
llaisysTensor_t Qwen2Model::forward_mlp(const llaisysTensor_t& x, size_t layer_idx) {
    // 获取当前层的MLP权重参数
    llaisysTensor_t gate_w = _weights->mlp_gate_w[layer_idx];
    llaisysTensor_t up_w = _weights->mlp_up_w[layer_idx];
    llaisysTensor_t down_w = _weights->mlp_down_w[layer_idx];
    
    // 获取x的形状信息
    size_t x_ndim = tensorGetNdim(x);
    size_t* x_shape = new size_t[x_ndim];
    tensorGetShape(x, x_shape);
    llaisysDataType_t x_dtype = tensorGetDataType(x);
    llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
    int x_device_id = tensorGetDeviceId(x);
    
    // 计算中间层的形状
    size_t intermediate_shape[2];
    for(size_t i = 0; i < x_ndim - 1; i++) {
        intermediate_shape[i] = x_shape[i];
    }
    intermediate_shape[x_ndim - 1] = _meta.di; // 中间层维度
    
    // 创建中间张量
    llaisysTensor_t gate_proj = tensorCreate(intermediate_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t up_proj = tensorCreate(intermediate_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    
    // 执行SwiGLU操作：Swish(x*W_g) * (x*W_u)
    if (_quantized) {
        llaisysQuantLinear(gate_proj, x, gate_w, _scale_mlp_gate[layer_idx], nullptr);
        llaisysQuantLinear(up_proj, x, up_w, _scale_mlp_up[layer_idx], nullptr);
    } else {
        llaisysLinear(gate_proj, x, gate_w, nullptr);
        llaisysLinear(up_proj, x, up_w, nullptr);
    }
    
    // SwiGLU 输出仍处在 intermediate_size，随后由 down_proj 投回 hidden_size。
    llaisysTensor_t swiglu_output = tensorCreate(intermediate_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    llaisysSwiGLU(swiglu_output, gate_proj, up_proj);
    
    // 创建最终输出张量
    llaisysTensor_t output = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    
    // 通过down projection得到最终输出
    if (_quantized)
        llaisysQuantLinear(output, swiglu_output, down_w, _scale_mlp_down[layer_idx], nullptr);
    else
        llaisysLinear(output, swiglu_output, down_w, nullptr);
    
    // 清理临时张量
    delete[] x_shape;
    tensorDestroy(gate_proj);
    tensorDestroy(up_proj);
    tensorDestroy(swiglu_output);
    
    return output;
}

llaisysTensor_t Qwen2Model::forward_rms_norm(const llaisysTensor_t& x, const llaisysTensor_t& weight, float eps) {
    // 获取x的形状信息用于创建输出张量
    size_t x_ndim = tensorGetNdim(x);
    size_t* x_shape = new size_t[x_ndim];
    tensorGetShape(x, x_shape);
    llaisysDataType_t x_dtype = tensorGetDataType(x);
    llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
    int x_device_id = tensorGetDeviceId(x);
    
    llaisysTensor_t output = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    delete[] x_shape;
    llaisysRmsNorm(output, x, weight, eps);  
    return output;
}

// =====================================================================
// P2-2：连续批处理引擎
// =====================================================================

int64_t Qwen2Model::engine_add_request(const int64_t* tokens, size_t n,
                                       int64_t max_new_tokens) {
    if (!tokens || n == 0) return -1;
    std::lock_guard<std::mutex> lock(_engine_mtx);
    EngineSeq s;
    s.prompt.assign(tokens, tokens + n);
    s.max_new_tokens = max_new_tokens > 0 ? max_new_tokens : 512;
    s.block_tables.resize(_meta.nlayer);
    int64_t id = static_cast<int64_t>(_eseqs.size());
    _eseqs.push_back(std::move(s));
    return id;
}

bool Qwen2Model::engine_has_unfinished() const {
    std::lock_guard<std::mutex> lock(_engine_mtx);
    for (const auto& s : _eseqs) if (!s.finished) return true;
    return false;
}
size_t Qwen2Model::engine_num_seqs() const {
    std::lock_guard<std::mutex> lock(_engine_mtx);
    return _eseqs.size();
}
bool Qwen2Model::engine_is_finished(int64_t id) const {
    std::lock_guard<std::mutex> lock(_engine_mtx);
    if (id < 0 || id >= static_cast<int64_t>(_eseqs.size())) return true;
    return _eseqs[id].finished;
}
bool Qwen2Model::engine_is_eos_finished(int64_t id) const {
    std::lock_guard<std::mutex> lock(_engine_mtx);
    if (id < 0 || id >= static_cast<int64_t>(_eseqs.size())) return false;
    return _eseqs[id].eos_finished;
}
size_t Qwen2Model::engine_get_output(int64_t id, int64_t* out, size_t cap) const {
    std::lock_guard<std::mutex> lock(_engine_mtx);
    if (id < 0 || id >= static_cast<int64_t>(_eseqs.size())) return 0;
    const EngineSeq& s = _eseqs[id];
    if (!s.finished) return 0;
    size_t n = std::min(cap, s.generated.size());
    for (size_t i = 0; i < n; ++i) out[i] = s.generated[i];
    return n;
}

// 为某序列在所有层池分配满足 new_len 的块（仅在块不够时分配）。
void Qwen2Model::engine_ensure_blocks(EngineSeq& s, size_t new_len) {
    if (new_len == 0) return;
    size_t need = (new_len + KV_BLOCK_TOKENS - 1) / KV_BLOCK_TOKENS;
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        std::vector<int32_t>& bt = s.block_tables[layer];
        while (bt.size() < need) {
            int pb = _paged_pools[layer].alloc();
            if (pb < 0)
                throw std::runtime_error("Engine: paged KV pool is full (increase LLAISYS_NUM_BLOCKS)");
            bt.push_back(static_cast<int32_t>(pb));
        }
    }
    s.num_tokens = new_len;
}

// 逐 token 把 RoPE 后的 K/V 写入共享池（按各序列块表寻址）。
void Qwen2Model::engine_write_kv(size_t layer, const llaisysTensor_t& k_rot,
                                 const llaisysTensor_t& v,
                                 const std::vector<size_t>& active,
                                 const std::vector<size_t>& qoff,
                                 const std::vector<size_t>& qlens,
                                 const int64_t* positions) {
    (void)qlens;
    KVBlockPool& pool = _paged_pools[layer];
    const size_t nseq = active.size(), total = qoff[nseq];
    const size_t ept = _meta.nkvh * _meta.dh;
    size_t esize;
    switch (_meta.dtype) {
        case LLAISYS_DTYPE_F16:
        case LLAISYS_DTYPE_BF16: esize = 2; break;
        case LLAISYS_DTYPE_F32: esize = 4; break;
        default: throw std::runtime_error("Engine: unsupported KV dtype");
    }
    char* kp = reinterpret_cast<char*>(tensorGetData(pool.k_pool));
    char* vp = reinterpret_cast<char*>(tensorGetData(pool.v_pool));
    const char* ks = reinterpret_cast<const char*>(tensorGetData(k_rot));
    const char* vs = reinterpret_cast<const char*>(tensorGetData(v));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long gq_ = 0; gq_ < static_cast<long long>(total); ++gq_) {
        size_t gq = static_cast<size_t>(gq_);
        size_t a = static_cast<size_t>(
            std::upper_bound(qoff.begin(), qoff.end(), gq) - qoff.begin() - 1);
        size_t s = active[a];
        size_t t = positions[gq];
        size_t lb = t / KV_BLOCK_TOKENS, off = t % KV_BLOCK_TOKENS;
        int32_t pb = _eseqs[s].block_tables[layer][lb];
        size_t dst = (((static_cast<size_t>(pb) * KV_BLOCK_TOKENS + off)
                       * _meta.nkvh) * _meta.dh);
        size_t src = gq * ept;
        std::memcpy(kp + dst * esize, ks + src * esize, ept * esize);
        std::memcpy(vp + dst * esize, vs + src * esize, ept * esize);
    }
}

// 单条采样：temperature<=0 时 greedy，否则 top-k/top-p。
static int64_t engine_sample_one(std::vector<float> logits, int top_k,
                                 float top_p, float temperature,
                                 std::mt19937& gen) {
    if (temperature <= 0.0f)
        return static_cast<int64_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<std::pair<float, int>> li;
    li.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        li.push_back({logits[i] / temperature, static_cast<int>(i)});
    std::sort(li.begin(), li.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (top_k > 0 && top_k < static_cast<int>(li.size())) li.resize(top_k);
    if (top_p > 0.0f && top_p < 1.0f) {
        float mx = li[0].first, sum = 0.0f;
        std::vector<float> pr;
        for (auto& it : li) { float p = std::exp(it.first - mx); pr.push_back(p); sum += p; }
        float cum = 0.0f; size_t cutoff = 0;
        for (size_t i = 0; i < pr.size(); ++i) {
            cum += pr[i] / sum;
            if (cum >= top_p) { cutoff = i + 1; break; }
        }
        if (cutoff > 0) li.resize(cutoff);
    }
    float mx = li[0].first, sum = 0.0f;
    std::vector<float> fp;
    for (auto& it : li) { float p = std::exp(it.first - mx); fp.push_back(p); sum += p; }
    (void)sum;
    std::discrete_distribution<> dist(fp.begin(), fp.end());
    return static_cast<int64_t>(li[dist(gen)].second);
}

size_t Qwen2Model::engine_step(int top_k, float top_p, float temperature) {
    std::lock_guard<std::mutex> lock(_engine_mtx);
    validate_weights();
    std::vector<size_t> active;
    for (size_t i = 0; i < _eseqs.size(); ++i)
        if (!_eseqs[i].finished) active.push_back(i);
    const size_t nseq = active.size();
    if (nseq == 0) return 0;

    // 每序列本次 query tokens、长度、旧长度；并为新长度分配块。
    std::vector<std::vector<int64_t>> qtoks(nseq);
    std::vector<size_t> qlens(nseq), old_lens(nseq);
    for (size_t a = 0; a < nseq; ++a) {
        EngineSeq& s = _eseqs[active[a]];
        old_lens[a] = s.num_tokens;
        if (s.is_new) qtoks[a] = s.prompt;
        else qtoks[a].push_back(s.generated.back());
        qlens[a] = qtoks[a].size();
        engine_ensure_blocks(s, old_lens[a] + qlens[a]);
    }
    std::vector<size_t> qoff(nseq + 1, 0);
    for (size_t a = 0; a < nseq; ++a) qoff[a + 1] = qoff[a] + qlens[a];
    const size_t total = qoff[nseq];

    // 拼接 input_ids / positions / context_lens / query_lens。
    std::vector<int64_t> input_ids(total), positions(total);
    std::vector<int32_t> ctxlens(nseq), qlens32(nseq);
    for (size_t a = 0; a < nseq; ++a) {
        for (size_t i = 0; i < qlens[a]; ++i) {
            size_t gq = qoff[a] + i;
            input_ids[gq] = qtoks[a][i];
            positions[gq] = static_cast<int64_t>(old_lens[a] + i);
        }
        ctxlens[a] = static_cast<int32_t>(old_lens[a] + qlens[a]);
        qlens32[a] = static_cast<int32_t>(qlens[a]);
    }

    // Embedding（整个 batch）。
    llaisysTensor_t x = forward_embedding(input_ids.data(), total);
    const size_t hs = _meta.hs;

    auto make_t = [&](std::vector<size_t> sh, llaisysDataType_t dt) {
        return tensorCreate(sh.data(), sh.size(), dt, _device, 0);
    };

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        // attention 输入 norm。
        llaisysTensor_t x_norm = forward_rms_norm(x, _weights->attn_norm_w[layer], _meta.epsilon);
        // q/k/v linear（rank-3）。
        llaisysTensor_t q = make_t({total, _meta.nh, _meta.dh}, _meta.dtype);
        llaisysTensor_t k = make_t({total, _meta.nkvh, _meta.dh}, _meta.dtype);
        llaisysTensor_t v = make_t({total, _meta.nkvh, _meta.dh}, _meta.dtype);
        if (_quantized) {
            llaisysQuantLinear(q, x_norm, _weights->attn_q_w[layer], _scale_attn_q[layer], _weights->attn_q_b[layer]);
            llaisysQuantLinear(k, x_norm, _weights->attn_k_w[layer], _scale_attn_k[layer], _weights->attn_k_b[layer]);
            llaisysQuantLinear(v, x_norm, _weights->attn_v_w[layer], _scale_attn_v[layer], _weights->attn_v_b[layer]);
        } else {
            llaisysLinear(q, x_norm, _weights->attn_q_w[layer], _weights->attn_q_b[layer]);
            llaisysLinear(k, x_norm, _weights->attn_k_w[layer], _weights->attn_k_b[layer]);
            llaisysLinear(v, x_norm, _weights->attn_v_w[layer], _weights->attn_v_b[layer]);
        }
        // positions 张量（int64），RoPE 整个 batch。
        llaisysTensor_t pos_t = make_t({total}, LLAISYS_DTYPE_I64);
        tensorLoad(pos_t, positions.data());
        llaisysTensor_t q_rope = make_t({total, _meta.nh, _meta.dh}, _meta.dtype);
        llaisysTensor_t k_rope = make_t({total, _meta.nkvh, _meta.dh}, _meta.dtype);
        llaisysROPE(q_rope, q, pos_t, _meta.theta);
        llaisysROPE(k_rope, k, pos_t, _meta.theta);
        // 写 K/V 到共享池。
        engine_write_kv(layer, k_rope, v, active, qoff, qlens, positions.data());
        // 该层 2D block table。
        size_t max_blk = 0;
        for (size_t a = 0; a < nseq; ++a)
            max_blk = std::max(max_blk, _eseqs[active[a]].block_tables[layer].size());
        std::vector<int32_t> bt2d(nseq * max_blk, 0);
        for (size_t a = 0; a < nseq; ++a) {
            const auto& bt = _eseqs[active[a]].block_tables[layer];
            for (size_t li = 0; li < bt.size(); ++li)
                bt2d[a * max_blk + li] = bt[li];
        }
        llaisysTensor_t bt_t = make_t({nseq, max_blk}, LLAISYS_DTYPE_I32);
        tensorLoad(bt_t, bt2d.data());
        llaisysTensor_t cl_t = make_t({nseq}, LLAISYS_DTYPE_I32);
        llaisysTensor_t ql_t = make_t({nseq}, LLAISYS_DTYPE_I32);
        tensorLoad(cl_t, ctxlens.data());
        tensorLoad(ql_t, qlens32.data());
        // batched PagedAttention。
        llaisysTensor_t attn = make_t({total, _meta.nh, _meta.dh}, _meta.dtype);
        llaisysBatchedPagedSelfAttention(attn, q_rope,
            _paged_pools[layer].k_pool, _paged_pools[layer].v_pool,
            bt_t, cl_t, ql_t,
            1.0f / std::sqrt(static_cast<float>(_meta.dh)));
        // 输出投影。
        llaisysTensor_t attn_2d = tensorView(attn,
            std::vector<size_t>{total, hs}.data(), 2);
        llaisysTensor_t attn_out = make_t({total, hs}, _meta.dtype);
        if (_quantized)
            llaisysQuantLinear(attn_out, attn_2d, _weights->attn_o_w[layer], _scale_attn_o[layer], nullptr);
        else
            llaisysLinear(attn_out, attn_2d, _weights->attn_o_w[layer], nullptr);
        // 残差 1。
        llaisysTensor_t r1 = make_t({total, hs}, _meta.dtype);
        llaisysAdd(r1, x, attn_out);
        tensorDestroy(x); x = r1;
        // MLP norm + MLP + 残差 2。
        llaisysTensor_t mn = forward_rms_norm(x, _weights->mlp_norm_w[layer], _meta.epsilon);
        llaisysTensor_t mo = forward_mlp(mn, layer);
        llaisysTensor_t r2 = make_t({total, hs}, _meta.dtype);
        llaisysAdd(r2, x, mo);
        tensorDestroy(x); x = r2;
        // 清理层临时。
        tensorDestroy(x_norm);
        tensorDestroy(q); tensorDestroy(k); tensorDestroy(v);
        tensorDestroy(pos_t); tensorDestroy(q_rope); tensorDestroy(k_rope);
        tensorDestroy(bt_t); tensorDestroy(cl_t); tensorDestroy(ql_t);
        tensorDestroy(attn); tensorDestroy(attn_out);
        tensorDestroy(mn); tensorDestroy(mo);
    }

    // 最终 norm + lm_head。
    llaisysTensor_t x_final = forward_rms_norm(x, _weights->out_norm_w, _meta.epsilon);
    llaisysTensor_t logits = make_t({total, _meta.voc}, _meta.dtype);
    if (_quantized)
        llaisysQuantLinear(logits, x_final, _weights->out_embed, _scale_lm_head, nullptr);
    else
        llaisysLinear(logits, x_final, _weights->out_embed, nullptr);

    // 每序列取最后一个 query 位置采样。
    static std::mt19937 gen(20260101);
    size_t finished_now = 0;
    for (size_t a = 0; a < nseq; ++a) {
        size_t gq = qoff[a + 1] - 1;
        llaisysTensor_t row = tensorSlice(logits, 0, gq, gq + 1);
        std::vector<float> lg = tensor_to_float_vector(row, _meta.voc);
        tensorDestroy(row);
        int64_t nt = engine_sample_one(std::move(lg), top_k, top_p, temperature, gen);
        EngineSeq& s = _eseqs[active[a]];
        s.is_new = false;
        bool hit_eos = (nt == _meta.end_token);
        bool hit_len = (static_cast<int64_t>(s.generated.size()) + 1 >= s.max_new_tokens);
        if (hit_eos || hit_len) {
            if (hit_eos) s.eos_finished = true;
            else s.generated.push_back(nt);  // 长度上限：保留最后一个 token
            s.finished = true;
            finished_now++;
            // 释放该序列在所有层占用的物理块。
            for (size_t layer = 0; layer < _meta.nlayer; ++layer)
                for (int32_t pb : s.block_tables[layer])
                    _paged_pools[layer].free_block(pb);
        } else {
            s.generated.push_back(nt);
        }
    }

    tensorDestroy(x);
    tensorDestroy(x_final);
    tensorDestroy(logits);
    return finished_now;
}

} // namespace models
} // namespace llaisys
