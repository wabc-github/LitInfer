from ctypes import Structure, POINTER, c_size_t, c_int, c_float, c_int64, c_uint32, c_void_p

from .llaisys_types import llaisysDeviceType_t, llaisysDataType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]



LlaisysQwen2Model = c_void_p


def load_models(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    lib.llaisysQwen2ModelInferWithSampling.argtypes = [
        LlaisysQwen2Model,
        POINTER(c_int64),
        c_size_t,
        c_int,
        c_float,
        c_float,
    ]
    lib.llaisysQwen2ModelInferWithSampling.restype = c_int64


    lib.llaisysQwen2ModelInitKVCache.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelInitKVCache.restype = None

    lib.llaisysQwen2ModelFreeKVCache.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelFreeKVCache.restype = None

    lib.llaisysQwen2ModelQuantizeInt8.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelQuantizeInt8.restype = None

    # ---- 连续批处理引擎 ----
    lib.llaisysEngineAddRequest.argtypes = [
        LlaisysQwen2Model, POINTER(c_int64), c_size_t, c_int64]
    lib.llaisysEngineAddRequest.restype = c_int64

    lib.llaisysEngineStep.argtypes = [LlaisysQwen2Model, c_int, c_float, c_float]
    lib.llaisysEngineStep.restype = c_size_t

    lib.llaisysEngineHasUnfinished.argtypes = [LlaisysQwen2Model]
    lib.llaisysEngineHasUnfinished.restype = c_int

    lib.llaisysEngineNumSeqs.argtypes = [LlaisysQwen2Model]
    lib.llaisysEngineNumSeqs.restype = c_size_t

    lib.llaisysEngineIsFinished.argtypes = [LlaisysQwen2Model, c_int64]
    lib.llaisysEngineIsFinished.restype = c_int

    lib.llaisysEngineGetOutput.argtypes = [
        LlaisysQwen2Model, c_int64, POINTER(c_int64), c_size_t]
    lib.llaisysEngineGetOutput.restype = c_size_t


__all__ = [
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "LlaisysQwen2Model",
    "load_models",
]
