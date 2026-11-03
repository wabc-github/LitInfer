#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    __export int64_t llaisysQwen2ModelInferWithSampling(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, int top_k, float top_p, float temperature);

    // 权重加载接口（从内存加载单张量到权重结构体）
    __export void llaisysQwen2WeightsLoadTensor(
        struct LlaisysQwen2Weights *weights,
        const char *tensor_name,  // 对应safetensors中的张量名（如"model.embed_tokens.weight"）
        llaisysTensor_t tensor    // 已加载数据的张量
    );

    // 初始化KV-Cache（每轮推理前调用）
    __export void llaisysQwen2ModelInitKVCache(struct LlaisysQwen2Model *model);

    // 开启 INT8 weight-only 量化（权重加载完成后调用一次，之后不可再切回 fp16）
    __export void llaisysQwen2ModelQuantizeInt8(struct LlaisysQwen2Model *model);

    // 释放KV-Cache（推理结束后调用）
    __export void llaisysQwen2ModelFreeKVCache(struct LlaisysQwen2Model *model);

    // ---- 连续批处理引擎（P2-2）。前提：已 InitKVCache ----
    // 添加请求，返回 id（<0 表示失败）。
    __export int64_t llaisysEngineAddRequest(struct LlaisysQwen2Model *model,
        int64_t *tokens, size_t n, int64_t max_new_tokens);
    // 推进一次调度，返回本步完成数。
    __export size_t llaisysEngineStep(struct LlaisysQwen2Model *model,
        int top_k, float top_p, float temperature);
    __export int llaisysEngineHasUnfinished(struct LlaisysQwen2Model *model);
    __export size_t llaisysEngineNumSeqs(struct LlaisysQwen2Model *model);
    __export int llaisysEngineIsFinished(struct LlaisysQwen2Model *model, int64_t id);
    // 取已完成请求输出（仅生成部分），返回写入数。
    __export size_t llaisysEngineGetOutput(struct LlaisysQwen2Model *model,
        int64_t id, int64_t *out, size_t cap);
}
#endif // LLAISYS_MODELS_QWEN2_H
