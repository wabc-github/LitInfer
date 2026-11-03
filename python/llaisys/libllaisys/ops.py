from .tensor import llaisysTensor_t
from ctypes import c_float, c_size_t

def load_ops(lib):
    lib.llaisysAdd.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysAdd.restype = None

    lib.llaisysArgmax.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysArgmax.restype = None

    lib.llaisysEmbedding.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysEmbedding.restype = None

    lib.llaisysLinear.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysLinear.restype = None

    lib.llaisysRearrange.argtypes = [llaisysTensor_t, llaisysTensor_t]
    lib.llaisysRearrange.restype = None

    lib.llaisysRmsNorm.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysRmsNorm.restype = None

    lib.llaisysROPE.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysROPE.restype = None

    lib.llaisysSelfAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k
        llaisysTensor_t,  # v
        c_float    # scale
    ]
    lib.llaisysSelfAttention.restype = None

    lib.llaisysSwiGLU.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysSwiGLU.restype = None

    # ---- INT8 weight-only 量化 ----
    lib.llaisysQuantizeWeight.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysQuantizeWeight.restype = None

    lib.llaisysQuantLinear.argtypes = [
        llaisysTensor_t,  # out
        llaisysTensor_t,  # in
        llaisysTensor_t,  # q_weight (int8)
        llaisysTensor_t,  # scale
        llaisysTensor_t,  # bias (可 NULL)
    ]
    lib.llaisysQuantLinear.restype = None

    # ---- PagedAttention ----
    lib.llaisysPagedSelfAttention.argtypes = [
        llaisysTensor_t,  # out
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k_pool
        llaisysTensor_t,  # v_pool
        llaisysTensor_t,  # block_table (int32)
        c_size_t,         # context_len
        c_float,          # scale
    ]
    lib.llaisysPagedSelfAttention.restype = None

    # ---- batched PagedAttention ----
    lib.llaisysBatchedPagedSelfAttention.argtypes = [
        llaisysTensor_t,  # out
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k_pool
        llaisysTensor_t,  # v_pool
        llaisysTensor_t,  # block_table (2D int32)
        llaisysTensor_t,  # context_lens (int32)
        llaisysTensor_t,  # query_lens (int32)
        c_float,          # scale
    ]
    lib.llaisysBatchedPagedSelfAttention.restype = None
