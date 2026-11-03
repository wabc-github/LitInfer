#include "llaisys/models/qwen2.h"
#include "models/qwen2.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
using namespace llaisys::models;

struct LlaisysQwen2Model {
	LlaisysQwen2Meta meta{};
	LlaisysQwen2Weights weights{};
	llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
	std::vector<int> device_ids;
	std::unique_ptr<llaisys::models::Qwen2Model> impl;
};

static void init_layer_arrays(LlaisysQwen2Weights &w, size_t nlayer) {
	w.attn_norm_w = new llaisysTensor_t[nlayer]();
	w.attn_q_w = new llaisysTensor_t[nlayer]();
	w.attn_q_b = new llaisysTensor_t[nlayer]();
	w.attn_k_w = new llaisysTensor_t[nlayer]();
	w.attn_k_b = new llaisysTensor_t[nlayer]();
	w.attn_v_w = new llaisysTensor_t[nlayer]();
	w.attn_v_b = new llaisysTensor_t[nlayer]();
	w.attn_o_w = new llaisysTensor_t[nlayer]();
	w.mlp_norm_w = new llaisysTensor_t[nlayer]();
	w.mlp_gate_w = new llaisysTensor_t[nlayer]();
	w.mlp_up_w = new llaisysTensor_t[nlayer]();
	w.mlp_down_w = new llaisysTensor_t[nlayer]();
}

static void destroy_layer_arrays(LlaisysQwen2Weights &w, size_t nlayer) {
	auto destroy_array = [nlayer](llaisysTensor_t *arr) {
		if (!arr) return;
		for (size_t i = 0; i < nlayer; ++i) {
			if (arr[i]) {
				tensorDestroy(arr[i]);
				arr[i] = nullptr;
			}
		}
		delete[] arr;
	};

	destroy_array(w.attn_norm_w);
	destroy_array(w.attn_q_w);
	destroy_array(w.attn_q_b);
	destroy_array(w.attn_k_w);
	destroy_array(w.attn_k_b);
	destroy_array(w.attn_v_w);
	destroy_array(w.attn_v_b);
	destroy_array(w.attn_o_w);
	destroy_array(w.mlp_norm_w);
	destroy_array(w.mlp_gate_w);
	destroy_array(w.mlp_up_w);
	destroy_array(w.mlp_down_w);

	w.attn_norm_w = nullptr;
	w.attn_q_w = nullptr;
	w.attn_q_b = nullptr;
	w.attn_k_w = nullptr;
	w.attn_k_b = nullptr;
	w.attn_v_w = nullptr;
	w.attn_v_b = nullptr;
	w.attn_o_w = nullptr;
	w.mlp_norm_w = nullptr;
	w.mlp_gate_w = nullptr;
	w.mlp_up_w = nullptr;
	w.mlp_down_w = nullptr;
}


__C {
    // 模型创建
	__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
		const LlaisysQwen2Meta *meta,
		llaisysDeviceType_t device,
		int *device_ids,
		int ndevice) {
		if (!meta || ndevice <= 0) return nullptr;

		auto *model = new LlaisysQwen2Model();
		model->meta = *meta;
		model->device = device;
		model->device_ids.assign(device_ids, device_ids + ndevice);

		init_layer_arrays(model->weights, model->meta.nlayer);
		model->impl = std::make_unique<llaisys::models::Qwen2Model>(
			model->meta,
			model->weights,
			model->device,
			model->device_ids);

		return model;
	}

    //销毁千问2模型实例
	__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
		if (!model) return;

		if (model->weights.in_embed) {
			tensorDestroy(model->weights.in_embed);
			if (model->weights.out_embed == model->weights.in_embed) {
				model->weights.out_embed = nullptr;
			}
			model->weights.in_embed = nullptr;
		}
		if (model->weights.out_embed) {
			tensorDestroy(model->weights.out_embed);
			model->weights.out_embed = nullptr;
		}
		if (model->weights.out_norm_w) {
			tensorDestroy(model->weights.out_norm_w);
			model->weights.out_norm_w = nullptr;
		}

		destroy_layer_arrays(model->weights, model->meta.nlayer);

		model->impl.reset();
		delete model;
	}


    // 获取权重结构体
	__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
		if (!model) return nullptr;
		return &model->weights;
	}

    // 推理
	__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
		if (!model || !model->impl) return -1;
		try {
			return model->impl->infer(token_ids, ntoken);
		} catch (const std::exception &e) {
			std::cerr << "[ERROR] Qwen2 infer failed: " << e.what() << std::endl;
			return -1;
		} catch (...) {
			std::cerr << "[ERROR] Qwen2 infer failed: unknown exception" << std::endl;
			return -1;
		}
    }

    // 修改推理函数，增加采样参数
    __export int64_t llaisysQwen2ModelInferWithSampling(
        struct LlaisysQwen2Model * model, 
        int64_t * token_ids, 
        size_t ntoken,
        int top_k,
        float top_p,
        float temperature
    ) {
        if (!model || !model->impl) return -1;
        try {
            return model->impl->infer_with_sampling(token_ids, ntoken, top_k, top_p, temperature);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Qwen2 infer_with_sampling failed: " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cerr << "[ERROR] Qwen2 infer_with_sampling failed: unknown exception" << std::endl;
            return -1;
        }
    }

    
    // 初始化KV-Cache
    __export void llaisysQwen2ModelInitKVCache(struct LlaisysQwen2Model *model) {
        if (!model || !model->impl) return ;
        model->impl->init_kv_cache();
    }

    // 释放KV-Cache
    __export void llaisysQwen2ModelFreeKVCache(struct LlaisysQwen2Model *model) {
        if (!model || !model->impl) return;
        model->impl->free_kv_cache();
    }

    // 开启 INT8 weight-only 量化
    __export void llaisysQwen2ModelQuantizeInt8(struct LlaisysQwen2Model *model) {
        if (!model || !model->impl) return;
        try {
            model->impl->quantize_int8();
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Qwen2 quantize failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[ERROR] Qwen2 quantize failed: unknown exception" << std::endl;
        }
    }

    // ---- 连续批处理引擎 ----
    __export int64_t llaisysEngineAddRequest(struct LlaisysQwen2Model *model,
        int64_t *tokens, size_t n, int64_t max_new_tokens) {
        if (!model || !model->impl) return -1;
        try {
            return model->impl->engine_add_request(tokens, n, max_new_tokens);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Engine add_request failed: " << e.what() << std::endl;
            return -1;
        } catch (...) { return -1; }
    }

    __export size_t llaisysEngineStep(struct LlaisysQwen2Model *model,
        int top_k, float top_p, float temperature) {
        if (!model || !model->impl) return 0;
        try {
            return model->impl->engine_step(top_k, top_p, temperature);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Engine step failed: " << e.what() << std::endl;
            return 0;
        } catch (...) { return 0; }
    }

    __export int llaisysEngineHasUnfinished(struct LlaisysQwen2Model *model) {
        if (!model || !model->impl) return 0;
        return model->impl->engine_has_unfinished() ? 1 : 0;
    }
    __export size_t llaisysEngineNumSeqs(struct LlaisysQwen2Model *model) {
        if (!model || !model->impl) return 0;
        return model->impl->engine_num_seqs();
    }
    __export int llaisysEngineIsFinished(struct LlaisysQwen2Model *model, int64_t id) {
        if (!model || !model->impl) return 1;
        return model->impl->engine_is_finished(id) ? 1 : 0;
    }
    __export size_t llaisysEngineGetOutput(struct LlaisysQwen2Model *model,
        int64_t id, int64_t *out, size_t cap) {
        if (!model || !model->impl) return 0;
        return model->impl->engine_get_output(id, out, cap);
    }
}
