"""冒烟测试：不依赖 PyTorch，用 numpy 作为参考，验证新 DLL 的算子正确性。
用法（在仓库根目录）：
    # 普通运行
    python scripts/smoke_test.py
    # 同时开启 Profiler（进程退出时打印各算子耗时）
    set LLAISYS_PROFILE=1   && python scripts/smoke_test.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

# 直接使用仓库内的 python 包，无需 pip install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from llaisys import Tensor, DataType, DeviceType
from llaisys import Ops


def load_numpy(arr: np.ndarray, dtype: DataType):
    """创建与 arr 同形状的 llaisys 张量并载入数据。"""
    t = Tensor(shape=list(arr.shape), dtype=dtype, device=DeviceType.CPU)
    t.load(np.ascontiguousarray(arr).ctypes.data)
    return t


def check(name, got, ref, atol=1e-4, rtol=1e-4):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ok = np.allclose(got, ref, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        print("  got:", got.ravel()[:8])
        print("  ref:", ref.ravel()[:8])
    return ok


def to_numpy(t: Tensor) -> np.ndarray:
    shape = t.shape()
    ptr = t.data_ptr()
    dt = t.dtype()
    if dt == DataType.F32:
        buf = np.empty(shape, dtype=np.float32)
    else:
        raise RuntimeError("smoke test only uses F32")
    ctypes_data = buf.ctypes.data
    # 从设备/库内存拷贝到 numpy
    import ctypes
    ctypes.memmove(buf.ctypes.data, ptr, buf.nbytes)
    return buf


def main():
    all_ok = True
    rng = np.random.default_rng(0)

    # ---- linear: Y = X W^T + b  (M=2, K=16, N=8) ----
    M, K, N = 2, 16, 8
    x = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
    b = (rng.standard_normal((N,)) * 0.1).astype(np.float32)
    y_ref = x @ w.T + b

    xt = load_numpy(x, DataType.F32)
    wt = load_numpy(w, DataType.F32)
    bt = load_numpy(b, DataType.F32)
    yt = Tensor(shape=[M, N], dtype=DataType.F32)
    Ops.linear(yt, xt, wt, bt)
    all_ok &= check("linear (M=2, with bias)", to_numpy(yt), y_ref)

    # 无 bias
    y_ref2 = x @ w.T
    yt2 = Tensor(shape=[M, N], dtype=DataType.F32)
    Ops.linear(yt2, xt, wt, None)
    all_ok &= check("linear (no bias)", to_numpy(yt2), y_ref2)

    # GEMV（M=1）并计时
    x1 = x[:1]
    y1_ref = x1 @ w.T + b
    x1t = load_numpy(x1, DataType.F32)
    y1t = Tensor(shape=[1, N], dtype=DataType.F32)
    t0 = time.perf_counter()
    for _ in range(100):
        Ops.linear(y1t, x1t, wt, bt)
    dt = (time.perf_counter() - t0) / 100
    all_ok &= check("linear GEMV (M=1)", to_numpy(y1t), y1_ref)
    print(f"   small GEMV avg wall time: {dt*1e6:.1f} us (tiny shape, not representative)")

    # ---- rms_norm ----
    rows, d = 2, 16
    xn = (rng.standard_normal((rows, d)) * 0.1).astype(np.float32)
    wn = (rng.standard_normal((d,)) * 0.1).astype(np.float32)
    eps = 1e-6
    rms = np.sqrt(np.mean(xn ** 2, axis=1, keepdims=True) + eps)
    n_ref = wn * xn / rms

    xnt = load_numpy(xn, DataType.F32)
    wnt = load_numpy(wn, DataType.F32)
    out_n = Tensor(shape=[rows, d], dtype=DataType.F32)
    Ops.rms_norm(out_n, xnt, wnt, eps)
    all_ok &= check("rms_norm", to_numpy(out_n), n_ref)

    # ---- add ----
    a = rng.standard_normal((4,)).astype(np.float32)
    bb = rng.standard_normal((4,)).astype(np.float32)
    at = load_numpy(a, DataType.F32)
    bbt = load_numpy(bb, DataType.F32)
    ct = Tensor(shape=[4], dtype=DataType.F32)
    Ops.add(ct, at, bbt)
    all_ok &= check("add", to_numpy(ct), a + bb)

    # ---- INT8 量化（quantize_weight + quant_linear）----
    import ctypes
    Nq, Kq = 8, 64
    wq_fp = (rng.standard_normal((Nq, Kq)) * 0.1).astype(np.float32)

    # quantize_weight：fp32 权重 -> int8 + scale
    q_t = Tensor(shape=[Nq, Kq], dtype=DataType.I8)
    s_t = Tensor(shape=[Nq], dtype=DataType.F32)
    Ops.quantize_weight(q_t, s_t, load_numpy(wq_fp, DataType.F32))
    # 读回 int8 与 scale
    q_np = np.empty((Nq, Kq), dtype=np.int8)
    ctypes.memmove(q_np.ctypes.data, q_t.data_ptr(), q_np.nbytes)
    s_np = np.empty((Nq), dtype=np.float32)
    ctypes.memmove(s_np.ctypes.data, s_t.data_ptr(), s_np.nbytes)
    # 反量化权重应接近原 fp32 权重（INT8 精度有限，容差放大）
    deq = q_np.astype(np.float32) * s_np[:, None]
    all_ok &= check("quantize_weight dequant ~= fp32", deq, wq_fp, atol=2e-2, rtol=2e-2)

    # quant_linear：fp32 激活（通用内核）
    Mq = 2
    xq = (rng.standard_normal((Mq, Kq)) * 0.1).astype(np.float32)
    bq = (rng.standard_normal((Nq)) * 0.1).astype(np.float32)
    ref_q = xq @ wq_fp.T + bq
    out_q = Tensor(shape=[Mq, Nq], dtype=DataType.F32)
    Ops.quant_linear(out_q, load_numpy(xq, DataType.F32), q_t, s_t,
                     load_numpy(bq, DataType.F32))
    all_ok &= check("quant_linear (fp32 act)", to_numpy(out_q), ref_q, atol=2e-2, rtol=2e-2)

    # quant_linear：fp16 激活（走 AVX2/F16C 在线反量化内核）
    w16 = wq_fp.astype(np.float16)
    x16 = xq.astype(np.float16)
    b16 = bq.astype(np.float16)
    q16 = Tensor(shape=[Nq, Kq], dtype=DataType.I8)
    s16 = Tensor(shape=[Nq], dtype=DataType.F32)
    Ops.quantize_weight(q16, s16, load_numpy(w16, DataType.F16))
    out16 = Tensor(shape=[Mq, Nq], dtype=DataType.F16)
    Ops.quant_linear(out16, load_numpy(x16, DataType.F16), q16, s16,
                     load_numpy(b16, DataType.F16))
    ref16 = (x16 @ w16.T + b16).astype(np.float16)  # fp16 参考
    got16 = np.empty((Mq, Nq), dtype=np.float16)
    ctypes.memmove(got16.ctypes.data, out16.data_ptr(), got16.nbytes)
    all_ok &= check("quant_linear (fp16 act, AVX2)", got16.astype(np.float32),
                    ref16.astype(np.float32), atol=3e-2, rtol=3e-2)

    # GEMV fp16（M=1，decode 关键路径）
    g16 = Tensor(shape=[1, Nq], dtype=DataType.F16)
    Ops.quant_linear(g16, load_numpy(x16[:1], DataType.F16), q16, s16,
                     load_numpy(b16, DataType.F16))
    gref = (x16[:1] @ w16.T + b16).astype(np.float16)
    ggot = np.empty((1, Nq), dtype=np.float16)
    ctypes.memmove(ggot.ctypes.data, g16.data_ptr(), ggot.nbytes)
    all_ok &= check("quant_linear GEMV (fp16, AVX2)", ggot.astype(np.float32),
                    gref.astype(np.float32), atol=3e-2, rtol=3e-2)

    # ---- PagedAttention（分页 KV，GQA + 因果 + 未填满最后块）----
    BLOCK = 16
    nh_p, nkvh_p, dh_p = 4, 2, 8
    ctx = 20                 # 跨 2 块，最后一块仅 4 个 token
    nphys = 4               # 物理块数（多于需求，用于验证乱序寻址）
    l2p = [3, 1]             # 逻辑块 0->物理块 3，逻辑块 1->物理块 1（故意乱序）
    scale_p = 1.0 / np.sqrt(dh_p)

    q_p = (rng.standard_normal((ctx, nh_p, dh_p)) * 0.2).astype(np.float32)
    k_log = (rng.standard_normal((ctx, nkvh_p, dh_p)) * 0.2).astype(np.float32)
    v_log = (rng.standard_normal((ctx, nkvh_p, dh_p)) * 0.2).astype(np.float32)

    # 按乱序映射把逻辑 KV 写入物理池（最后块只写 4 个 slot）。
    kp_np = np.zeros((nphys, BLOCK, nkvh_p, dh_p), dtype=np.float32)
    vp_np = np.zeros((nphys, BLOCK, nkvh_p, dh_p), dtype=np.float32)
    for lb in range((ctx + BLOCK - 1) // BLOCK):
        pb = l2p[lb]; s = lb * BLOCK; e = min(s + BLOCK, ctx)
        kp_np[pb, 0:e - s] = k_log[s:e]
        vp_np[pb, 0:e - s] = v_log[s:e]
    bt_np = np.zeros(nphys, dtype=np.int32); bt_np[:len(l2p)] = l2p

    # numpy 参考：GQA 因果 attention（用逻辑顺序的 K/V）。
    def numpy_gqa(q, k, v, scale, causal):
        ql, nh = q.shape[0], q.shape[1]; nkvh = k.shape[1]
        group = nh // nkvh; out = np.zeros_like(q)
        for h in range(nh):
            kh = h // group
            sc = np.einsum("qd,jd->qj", q[:, h], k[:, kh]) * scale
            if causal:
                sc[np.arange(ql)[None, :] > np.arange(ql)[:, None]] = -np.inf
            p = np.exp(sc - sc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            out[:, h] = p @ v[:, kh]
        return out

    # prefill：q_len=ctx（因果），PagedAttention 按 block table 寻址。
    out_p = Tensor(shape=[ctx, nh_p, dh_p], dtype=DataType.F32)
    Ops.paged_self_attention(
        out_p, load_numpy(q_p, DataType.F32),
        load_numpy(kp_np, DataType.F32), load_numpy(vp_np, DataType.F32),
        load_numpy(bt_np, DataType.I32), ctx, scale_p)
    ref_p = numpy_gqa(q_p, k_log, v_log, scale_p, causal=True)
    all_ok &= check("paged_attention prefill (乱序块表, 因果)",
                    to_numpy(out_p), ref_p, atol=1e-3, rtol=1e-3)

    # decode：q_len=1，query 看全部 ctx 个 KV（非因果）。
    q_d = (rng.standard_normal((1, nh_p, dh_p)) * 0.2).astype(np.float32)
    out_d = Tensor(shape=[1, nh_p, dh_p], dtype=DataType.F32)
    Ops.paged_self_attention(
        out_d, load_numpy(q_d, DataType.F32),
        load_numpy(kp_np, DataType.F32), load_numpy(vp_np, DataType.F32),
        load_numpy(bt_np, DataType.I32), ctx, scale_p)
    ref_d = numpy_gqa(q_d, k_log, v_log, scale_p, causal=False)
    all_ok &= check("paged_attention decode (q_len=1)",
                    to_numpy(out_d), ref_d, atol=1e-3, rtol=1e-3)

    # ---- batched PagedAttention（2 序列同时 decode，块表隔离）----
    ctx0, ctx1 = 20, 28
    nphys_b = 4
    # seq0 块 0,1 -> 物理块 1,3；seq1 块 0,1 -> 物理块 0,2（全部错开，验证隔离）
    bt2 = np.array([[1, 3], [0, 2]], dtype=np.int32)
    k0 = (rng.standard_normal((ctx0, nkvh_p, dh_p)) * 0.2).astype(np.float32)
    v0 = (rng.standard_normal((ctx0, nkvh_p, dh_p)) * 0.2).astype(np.float32)
    k1 = (rng.standard_normal((ctx1, nkvh_p, dh_p)) * 0.2).astype(np.float32)
    v1 = (rng.standard_normal((ctx1, nkvh_p, dh_p)) * 0.2).astype(np.float32)
    qb2 = (rng.standard_normal((2, nh_p, dh_p)) * 0.2).astype(np.float32)

    kpb = np.zeros((nphys_b, BLOCK, nkvh_p, dh_p), dtype=np.float32)
    vpb = np.zeros((nphys_b, BLOCK, nkvh_p, dh_p), dtype=np.float32)
    def fill_pool(pool, logical, ctx, table):
        for lb in range((ctx + BLOCK - 1) // BLOCK):
            pb = table[lb]; s = lb * BLOCK; e = min(s + BLOCK, ctx)
            pool[pb, 0:e - s] = logical[s:e]
    fill_pool(kpb, k0, ctx0, bt2[0])
    fill_pool(kpb, k1, ctx1, bt2[1])
    fill_pool(vpb, v0, ctx0, bt2[0])
    fill_pool(vpb, v1, ctx1, bt2[1])

    cl2 = np.array([ctx0, ctx1], dtype=np.int32)
    ql2 = np.array([1, 1], dtype=np.int32)
    out_b2 = Tensor(shape=[2, nh_p, dh_p], dtype=DataType.F32)
    Ops.batched_paged_self_attention(
        out_b2, load_numpy(qb2, DataType.F32),
        load_numpy(kpb, DataType.F32), load_numpy(vpb, DataType.F32),
        load_numpy(bt2, DataType.I32),
        load_numpy(cl2, DataType.I32), load_numpy(ql2, DataType.I32),
        scale_p)
    # numpy 参考：每序列 decode 单 query，看全 context。
    def decode_ref(q, k, v, scale):
        nh = q.shape[0]; nkvh = k.shape[1]; group = nh // nkvh
        out = np.zeros_like(q)
        for h in range(nh):
            kvh = h // group
            sc = q[h] @ k[:, kvh].T * scale
            p = np.exp(sc - sc.max()); p /= p.sum()
            out[h] = p @ v[:, kvh]
        return out
    ref_b2 = np.stack([
        decode_ref(qb2[0], k0, v0, scale_p),
        decode_ref(qb2[1], k1, v1, scale_p),
    ])
    all_ok &= check("batched_paged_attention (2 seq decode)",
                    to_numpy(out_b2), ref_b2, atol=1e-3, rtol=1e-3)

    print("\n" + ("ALL SMOKE TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
