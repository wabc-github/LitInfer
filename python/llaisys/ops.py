from .libllaisys import LIB_LLAISYS
from .tensor import Tensor
from ctypes import c_float, c_int, c_size_t


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        LIB_LLAISYS.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        LIB_LLAISYS.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        LIB_LLAISYS.llaisysEmbedding(
            out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
        )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        # bias 可选：None 时向 C 层传 NULL 指针
        bias_ref = bias.lib_tensor() if bias is not None else None
        LIB_LLAISYS.llaisysLinear(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), bias_ref
        )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        LIB_LLAISYS.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        LIB_LLAISYS.llaisysRmsNorm(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
        )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        LIB_LLAISYS.llaisysROPE(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
        )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float):
        LIB_LLAISYS.llaisysSelfAttention(
            attn_val.lib_tensor(),
            q.lib_tensor(),
            k.lib_tensor(),
            v.lib_tensor(),
            c_float(scale),
        )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        LIB_LLAISYS.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())

    # ---- INT8 weight-only 量化 ----
    @staticmethod
    def quantize_weight(q: Tensor, scale: Tensor, weight: Tensor):
        LIB_LLAISYS.llaisysQuantizeWeight(
            q.lib_tensor(), scale.lib_tensor(), weight.lib_tensor()
        )

    @staticmethod
    def quant_linear(out: Tensor, inp: Tensor, q_weight: Tensor,
                     scale: Tensor, bias: Tensor):
        # bias 可选：None 时传 NULL
        bias_ref = bias.lib_tensor() if bias is not None else None
        LIB_LLAISYS.llaisysQuantLinear(
            out.lib_tensor(), inp.lib_tensor(), q_weight.lib_tensor(),
            scale.lib_tensor(), bias_ref
        )

    # ---- PagedAttention ----
    @staticmethod
    def paged_self_attention(attn_val: Tensor, q: Tensor,
                             k_pool: Tensor, v_pool: Tensor,
                             block_table: Tensor, context_len: int, scale: float):
        LIB_LLAISYS.llaisysPagedSelfAttention(
            attn_val.lib_tensor(), q.lib_tensor(),
            k_pool.lib_tensor(), v_pool.lib_tensor(),
            block_table.lib_tensor(), c_size_t(context_len), c_float(scale)
        )

    @staticmethod
    def batched_paged_self_attention(attn_val: Tensor, q: Tensor,
                                     k_pool: Tensor, v_pool: Tensor,
                                     block_table: Tensor,
                                     context_lens: Tensor, query_lens: Tensor,
                                     scale: float):
        LIB_LLAISYS.llaisysBatchedPagedSelfAttention(
            attn_val.lib_tensor(), q.lib_tensor(),
            k_pool.lib_tensor(), v_pool.lib_tensor(),
            block_table.lib_tensor(),
            context_lens.lib_tensor(), query_lens.lib_tensor(), c_float(scale)
        )
