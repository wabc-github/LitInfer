from __future__ import annotations

from typing import Sequence
from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysDeviceType_t,
    llaisysDataType_t,
    LlaisysQwen2Meta,
)
import ctypes
import os
from ctypes import byref, c_int, c_size_t, c_float, c_int64, c_void_p
from pathlib import Path
import json
import mmap
import struct
import time



class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU,
                 quantize: str = None):  # quantize: None 或 "int8"（weight-only）
        self.model_path = Path(model_path)
        self.device = device
        self.quantization = quantize
        
        # 实例化模型元信息
        print("Loading model config...\n", flush=True)
        self.meta, _ = self._load_config()
        print("Model config loaded.\n", flush=True)
        
        # 创建模型
        device_ids = (c_int * 1)(0)
        print("Creating Qwen2 model...\n", flush=True)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta),
            llaisysDeviceType_t(device),
            device_ids,
            1,
        )
        
        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model")
            
        # 加载权重
        print("Loading model weights...\n", flush=True)
        import numpy as np

        self._model_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        def _dtype_to_llaisys(dtype: np.dtype) -> DataType:
            name = getattr(dtype, "name", str(dtype)).lower()
            if name in {"float32", "f4"}:
                return DataType.F32
            if name in {"float16", "f2"}:
                return DataType.F16
            if name in {"bfloat16", "bf16"}:
                return DataType.BF16
            if name in {"int64", "i8"}:
                return DataType.I64
            if name in {"int32", "i4"}:
                return DataType.I32
            if name in {"int16", "i2"}:
                return DataType.I16
            if name in {"int8", "i1"}:
                return DataType.I8
            if name in {"uint8", "u1"}:
                return DataType.U8
            raise ValueError(f"Unsupported dtype: {dtype}")

        def _create_tensor_from_numpy(arr: np.ndarray):
            arr = np.ascontiguousarray(arr)
            _shape = (c_size_t * arr.ndim)(*arr.shape)
            _dtype = _dtype_to_llaisys(arr.dtype)
            tensor = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(arr.ndim),
                llaisysDataType_t(_dtype),
                llaisysDeviceType_t(device),
                c_int(0),
            )
            LIB_LLAISYS.tensorLoad(tensor, c_void_p(arr.ctypes.data))
            return tensor

        def _array_from_safetensors_buffer(mm, offset, dtype_name, shape):
            count = int(np.prod(shape, dtype=np.int64))
            dtype_map = {
                "F64": "<f8",
                "F32": "<f4",
                "F16": "<f2",
                "I64": "<i8",
                "I32": "<i4",
                "I16": "<i2",
                "I8": "i1",
                "U64": "<u8",
                "U32": "<u4",
                "U16": "<u2",
                "U8": "u1",
                "BOOL": "?",
            }
            if dtype_name == "BF16":
                raw = np.frombuffer(mm, dtype="<u2", count=count, offset=offset)
                bits = raw.astype(np.uint32) << 16
                return bits.view(np.float32).astype(np.float16).reshape(shape)
            if dtype_name not in dtype_map:
                raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")
            return np.frombuffer(mm, dtype=dtype_map[dtype_name], count=count, offset=offset).reshape(shape)

        def _iter_safetensors(file: Path):
            with open(file, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
                data_start = 8 + header_len
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for name, info in header.items():
                        if name == "__metadata__":
                            continue
                        begin, _end = info["data_offsets"]
                        shape = tuple(info["shape"])
                        arr = _array_from_safetensors_buffer(
                            mm,
                            data_start + begin,
                            info["dtype"],
                            shape,
                        )
                        yield name, arr

        # 加载模型权重
        loaded = set()
        for file in sorted(self.model_path.glob("*.safetensors")):
            print(f"Reading safetensors file: {file.name}", flush=True)
            for name_, arr in _iter_safetensors(file):
                tensor = _create_tensor_from_numpy(arr)
                w = self._model_weights.contents

                if name_ in {"model.embed_tokens.weight", "transformer.wte.weight"}:
                    w.in_embed = tensor
                    loaded.add("in_embed")
                    continue
                if name_ in {"lm_head.weight", "model.lm_head.weight"}:
                    w.out_embed = tensor
                    loaded.add("out_embed")
                    continue
                if name_ in {"model.norm.weight", "transformer.ln_f.weight"}:
                    w.out_norm_w = tensor
                    loaded.add("out_norm_w")
                    continue

                if name_.startswith("model.layers."):
                    parts = name_.split(".")
                    if len(parts) < 4:
                        continue
                    layer = int(parts[2])
                    sub = ".".join(parts[3:])

                    if sub == "input_layernorm.weight":
                        w.attn_norm_w[layer] = tensor
                        loaded.add(f"attn_norm_w.{layer}")
                    elif sub == "self_attn.q_proj.weight":
                        w.attn_q_w[layer] = tensor
                        loaded.add(f"attn_q_w.{layer}")
                    elif sub == "self_attn.q_proj.bias":
                        w.attn_q_b[layer] = tensor
                        loaded.add(f"attn_q_b.{layer}")
                    elif sub == "self_attn.k_proj.weight":
                        w.attn_k_w[layer] = tensor
                        loaded.add(f"attn_k_w.{layer}")
                    elif sub == "self_attn.k_proj.bias":
                        w.attn_k_b[layer] = tensor
                        loaded.add(f"attn_k_b.{layer}")
                    elif sub == "self_attn.v_proj.weight":
                        w.attn_v_w[layer] = tensor
                        loaded.add(f"attn_v_w.{layer}")
                    elif sub == "self_attn.v_proj.bias":
                        w.attn_v_b[layer] = tensor
                        loaded.add(f"attn_v_b.{layer}")
                    elif sub == "self_attn.o_proj.weight":
                        w.attn_o_w[layer] = tensor
                        loaded.add(f"attn_o_w.{layer}")
                    elif sub == "post_attention_layernorm.weight":
                        w.mlp_norm_w[layer] = tensor
                        loaded.add(f"mlp_norm_w.{layer}")
                    elif sub == "mlp.gate_proj.weight":
                        w.mlp_gate_w[layer] = tensor
                        loaded.add(f"mlp_gate_w.{layer}")
                    elif sub == "mlp.up_proj.weight":
                        w.mlp_up_w[layer] = tensor
                        loaded.add(f"mlp_up_w.{layer}")
                    elif sub == "mlp.down_proj.weight":
                        w.mlp_down_w[layer] = tensor
                        loaded.add(f"mlp_down_w.{layer}")

        w = self._model_weights.contents
        if not w.out_embed and w.in_embed:
            w.out_embed = w.in_embed
            loaded.add("out_embed(tied)")
        self._validate_loaded_weights(loaded)
        print("Model weights loaded.\n", flush=True)

        # 可选 INT8 weight-only 量化（必须在权重加载完成后、推理前调用一次）。
        # 量化会销毁原 fp16 权重，之后不可再切回 fp16。
        if quantize == "int8":
            print("Quantizing weights to INT8 (weight-only)...\n", flush=True)
            LIB_LLAISYS.llaisysQwen2ModelQuantizeInt8(self.model)
            print("INT8 quantization done.\n", flush=True)

    def _validate_loaded_weights(self, loaded: set[str]):
        required = {"in_embed", "out_embed", "out_embed(tied)", "out_norm_w"}
        missing = []
        if "in_embed" not in loaded:
            missing.append("model.embed_tokens.weight")
        if "out_embed" not in loaded and "out_embed(tied)" not in loaded:
            missing.append("lm_head.weight or tied model.embed_tokens.weight")
        if "out_norm_w" not in loaded:
            missing.append("model.norm.weight")

        layer_required = [
            ("attn_norm_w", "input_layernorm.weight"),
            ("attn_q_w", "self_attn.q_proj.weight"),
            ("attn_k_w", "self_attn.k_proj.weight"),
            ("attn_v_w", "self_attn.v_proj.weight"),
            ("attn_o_w", "self_attn.o_proj.weight"),
            ("mlp_norm_w", "post_attention_layernorm.weight"),
            ("mlp_gate_w", "mlp.gate_proj.weight"),
            ("mlp_up_w", "mlp.up_proj.weight"),
            ("mlp_down_w", "mlp.down_proj.weight"),
        ]
        for layer in range(self.meta.nlayer):
            for field, name in layer_required:
                if f"{field}.{layer}" not in loaded:
                    missing.append(f"model.layers.{layer}.{name}")

        if missing:
            preview = "\n  - ".join(missing[:20])
            suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
            raise RuntimeError(f"Missing required Qwen2 weights:\n  - {preview}{suffix}")

        print(f"Loaded required Qwen2 weights for {self.meta.nlayer} layers.", flush=True)

    def _load_config(self):

        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        torch_dtype = str(cfg.get("torch_dtype", "bfloat16")).lower()
        if "float32" in torch_dtype or torch_dtype in {"fp32", "f32"}:
            dtype = DataType.F32
        elif "float16" in torch_dtype or torch_dtype in {"fp16", "f16"}:
            dtype = DataType.F16
        else:
            dtype = DataType.BF16
        # 统一用 torch 读取 bfloat16，并降级为 float16，避免 numpy bfloat16 兼容问题
        use_torch_loader = False
        if dtype == DataType.BF16:
            dtype = DataType.F16
            use_torch_loader = True
        # 解析模型参数
        nlayer = int(cfg.get("num_hidden_layers", 0))
        hs = int(cfg.get("hidden_size", 0))
        nh = int(cfg.get("num_attention_heads", 0))
        nkvh = int(cfg.get("num_key_value_heads", nh))
        di = int(cfg.get("intermediate_size", 0))
        maxseq = int(cfg.get("max_position_embeddings", 0))
        voc = int(cfg.get("vocab_size", 0))
        epsilon = float(cfg.get("rms_norm_eps", 1e-6))
        theta = float(cfg.get("rope_theta", 10000.0))
        eos = cfg.get("eos_token_id", -1)
        # 解析结束token
        if isinstance(eos, list):
            end_token = int(eos[0]) if eos else -1
        else:
            end_token = int(eos)
        # 解析head_dim
        dh = int(cfg.get("head_dim", hs // nh if nh else 0))
        # 创建模型元信息结构体
        model_meta = LlaisysQwen2Meta(
            llaisysDataType_t(dtype),
            c_size_t(nlayer),
            c_size_t(hs),
            c_size_t(nh),
            c_size_t(nkvh),
            c_size_t(dh),
            c_size_t(di),
            c_size_t(maxseq),
            c_size_t(voc),
            c_float(epsilon),
            c_float(theta),
            c_int64(end_token),
        )

        return model_meta, use_torch_loader





    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        tokens, _ = self.generate_profile(inputs, max_new_tokens, top_k, top_p, temperature)
        return tokens

    def generate_profile(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        max_new_tokens = 1 if max_new_tokens is None else max_new_tokens
        if max_new_tokens <= 0:
            return [], {
                "prefill_seconds": 0.0,
                "first_token_seconds": 0.0,
                "decode_seconds": 0.0,
                "decode_tokens": 0,
                "decode_tokens_per_second": 0.0,
                "time_per_decode_token_seconds": 0.0,
                "generated_tokens": 0,
            }

        LIB_LLAISYS.llaisysQwen2ModelInitKVCache(self.model)
        try:
            input_tokens = list(inputs)
            if not input_tokens:
                raise ValueError("generate: inputs must not be empty")
            max_seq_len = int(self.meta.maxseq)
            if len(input_tokens) + max_new_tokens - 1 > max_seq_len:
                raise ValueError(
                    "generate: requested prompt and output exceed model max sequence length "
                    f"({max_seq_len})"
                )

            generated_tokens = []

            c_tokens = (ctypes.c_int64 * len(input_tokens))(*input_tokens)
            trace_value = os.environ.get("LLAISYS_TRACE", "").lower()
            if trace_value in {"1", "true", "on"}:
                print(
                    f"[LLAISYS] Calling infer model={self.model} "
                    f"ntokens={len(input_tokens)}",
                    flush=True,
                )
            prefill_start = time.perf_counter()
            first_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, c_tokens, len(input_tokens))
            if first_token < 0:
                raise RuntimeError("LLAISYS prefill failed; check model inputs and max sequence length")
            prefill_seconds = time.perf_counter() - prefill_start
            if first_token == self.meta.end_token:
                return generated_tokens, {
                    "prefill_seconds": prefill_seconds,
                    "first_token_seconds": prefill_seconds,
                    "decode_seconds": 0.0,
                    "decode_tokens": 0,
                    "decode_tokens_per_second": 0.0,
                    "time_per_decode_token_seconds": 0.0,
                    "generated_tokens": 0,
                }
            generated_tokens.append(first_token)

            decode_start = time.perf_counter()
            for _ in range(1, max_new_tokens):
                c_inference_token = (ctypes.c_int64 * 1)(generated_tokens[-1])
                if top_k == 1 and top_p >= 1.0 and temperature == 1.0:
                    next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, c_inference_token, 1)
                else:
                    next_token = LIB_LLAISYS.llaisysQwen2ModelInferWithSampling(
                        self.model,
                        c_inference_token,
                        1,
                        top_k,
                        top_p,
                        temperature,
                    )

                if next_token < 0:
                    raise RuntimeError("LLAISYS decode failed")

                if next_token == self.meta.end_token:
                    break
                generated_tokens.append(next_token)

            decode_seconds = time.perf_counter() - decode_start
            decode_tokens = max(len(generated_tokens) - 1, 0)
            return generated_tokens, {
                "prefill_seconds": prefill_seconds,
                "first_token_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "decode_tokens": decode_tokens,
                "decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
                "time_per_decode_token_seconds": decode_seconds / decode_tokens if decode_tokens > 0 else 0.0,
                "generated_tokens": len(generated_tokens),
            }
        finally:
            LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(self.model)

    def generate_stream(self, inputs, max_new_tokens: int = 128,
                        top_k: int = 1, top_p: float = 1.0,
                        temperature: float = 1.0):
        """流式生成：逐 token yield token id（生成器）。

        与 generate_profile 的区别：每个 token 一生成就立即 yield，
        供服务端做 SSE 实时推送。遍历结束、break 或调用方提前关闭
        （generator.close）时，finally 都会释放 KV-Cache。
        """
        max_new_tokens = 1 if max_new_tokens is None else max_new_tokens
        LIB_LLAISYS.llaisysQwen2ModelInitKVCache(self.model)
        try:
            input_tokens = list(inputs)
            if not input_tokens:
                raise ValueError("generate_stream: inputs must not be empty")
            if len(input_tokens) + max_new_tokens - 1 > int(self.meta.maxseq):
                raise ValueError("generate_stream: prompt and output exceed max sequence length")

            # prefill：一次性处理全部 prompt，得到第一个 token。
            c_tokens = (ctypes.c_int64 * len(input_tokens))(*input_tokens)
            first = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, c_tokens, len(input_tokens))
            if first < 0:
                raise RuntimeError("LLAISYS prefill failed")
            if first == self.meta.end_token:
                return  # 直接结束，不 yield
            yield first

            # decode：逐 token 生成，贪心走 infer，否则 infer_with_sampling。
            prev = first
            for _ in range(1, max_new_tokens):
                c_in = (ctypes.c_int64 * 1)(prev)
                if top_k == 1 and top_p >= 1.0 and temperature == 1.0:
                    nxt = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model, c_in, 1)
                else:
                    nxt = LIB_LLAISYS.llaisysQwen2ModelInferWithSampling(
                        self.model, c_in, 1, top_k, top_p, temperature)
                if nxt < 0:
                    raise RuntimeError("LLAISYS decode failed")
                if nxt == self.meta.end_token:
                    break
                yield nxt
                prev = nxt
        finally:
            LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(self.model)

    # ---- 连续批处理引擎（P2-2）高层接口 ----
    def engine_start(self):
        """初始化共享分页池（开始批量服务前调用一次）。"""
        LIB_LLAISYS.llaisysQwen2ModelInitKVCache(self.model)

    def engine_add_request(self, inputs: Sequence[int], max_new_tokens: int = 128):
        """提交一条请求（prompt token），返回该请求 id。"""
        import numpy as np
        arr = np.ascontiguousarray(list(inputs), dtype=np.int64)
        cptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        return LIB_LLAISYS.llaisysEngineAddRequest(
            self.model, cptr, c_size_t(arr.size), c_int64(max_new_tokens))

    def engine_step(self, top_k: int = 1, top_p: float = 1.0,
                    temperature: float = 1.0):
        """推进一次调度：所有未完成请求同时做一次前向，返回本步完成数。"""
        return LIB_LLAISYS.llaisysEngineStep(
            self.model, c_int(top_k), c_float(top_p), c_float(temperature))

    def engine_has_unfinished(self) -> bool:
        return bool(LIB_LLAISYS.llaisysEngineHasUnfinished(self.model))

    def engine_num_seqs(self) -> int:
        return LIB_LLAISYS.llaisysEngineNumSeqs(self.model)

    def engine_is_finished(self, req_id: int) -> bool:
        return bool(LIB_LLAISYS.llaisysEngineIsFinished(
            self.model, c_int64(req_id)))

    def engine_get_output(self, req_id: int):
        """取已完成请求的生成 token（不含 prompt）。未完成返回空列表。"""
        cap = int(self.meta.maxseq)
        buf = (c_int64 * cap)()
        n = LIB_LLAISYS.llaisysEngineGetOutput(
            self.model, c_int64(req_id), buf, c_size_t(cap))
        return list(buf[:n])

    def engine_stop(self):
        """释放共享分页池。"""
        LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(self.model)

    def __del__(self):
        # 销毁模型
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
