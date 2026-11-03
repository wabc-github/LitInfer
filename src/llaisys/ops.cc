#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/paged_attention/op.hpp"
#include "../ops/quantize/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias ? bias->tensor : nullptr);
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
    void llaisysQuantizeWeight(llaisysTensor_t q, llaisysTensor_t scale, llaisysTensor_t weight) {
        llaisys::ops::quantize_weight(q->tensor, scale->tensor, weight->tensor);
    }
    void llaisysQuantLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t q_weight, llaisysTensor_t scale, llaisysTensor_t bias) {
        llaisys::ops::quant_linear(out->tensor, in->tensor, q_weight->tensor,
                                   scale->tensor, bias ? bias->tensor : nullptr);
    }
    void llaisysPagedSelfAttention(llaisysTensor_t out, llaisysTensor_t q,
        llaisysTensor_t k_pool, llaisysTensor_t v_pool, llaisysTensor_t block_table,
        size_t context_len, float scale) {
        llaisys::ops::paged_self_attention(out->tensor, q->tensor,
            k_pool->tensor, v_pool->tensor, block_table->tensor,
            context_len, scale);
    }
    void llaisysBatchedPagedSelfAttention(llaisysTensor_t out, llaisysTensor_t q,
        llaisysTensor_t k_pool, llaisysTensor_t v_pool, llaisysTensor_t block_table,
        llaisysTensor_t context_lens, llaisysTensor_t query_lens, float scale) {
        llaisys::ops::batched_paged_self_attention(out->tensor, q->tensor,
            k_pool->tensor, v_pool->tensor, block_table->tensor,
            context_lens->tensor, query_lens->tensor, scale);
    }
}
