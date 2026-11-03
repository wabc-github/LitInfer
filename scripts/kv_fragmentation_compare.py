"""分页 KV vs 连续 KV：内存利用率 / 最大并发 / 碎片对比（Monte Carlo）。
用于简历与面试：同一 KV 内存预算下，两种分配策略能同时服务的请求数。

运行：
    python scripts/kv_fragmentation_compare.py
"""
import numpy as np

# ---- Qwen2-1.5B（DeepSeek-R1-Distill-Qwen-1.5B）配置 ----
NLAYER = 28
NKVH = 2          # num_key_value_heads
DH = 128          # head_dim
BYTES = 2         # fp16
BLOCK_TOKENS = 16

# 每 token 的 KV 字节数（K 和 V 两份，全部层）。
PER_TOKEN_BYTES = 2 * NLAYER * NKVH * DH * BYTES
BLOCK_BYTES = BLOCK_TOKENS * PER_TOKEN_BYTES

# 连续方案：每条序列固定预分配 maxseq 槽位。
MAXSEQ = 2048
PER_SEQ_CONTIGUOUS = MAXSEQ * PER_TOKEN_BYTES

# 请求长度分布（聊天场景：多数短，少数长），均值约 300。
def sample_lengths(rng, n):
    # 截断的指数分布 + 下限
    lengths = rng.exponential(scale=300.0, size=n) + 16
    return np.minimum(lengths, 1800).astype(np.int64)


def run(budget_mb, seed=0, trials=200):
    rng = np.random.default_rng(seed)
    budget = budget_mb * 1024 * 1024

    # 连续方案最大并发（与长度无关，每请求占 maxseq）。
    contiguous_concurrency = int(budget // PER_SEQ_CONTIGUOUS)

    paged_conc = []
    paged_util = []
    for _ in range(trials):
        lengths = sample_lengths(rng, 10000)
        used_bytes = 0
        count = 0
        useful_bytes = 0
        for L in lengths:
            blocks = (L + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            need = blocks * BLOCK_BYTES
            if used_bytes + need > budget:
                break
            used_bytes += need
            useful_bytes += L * PER_TOKEN_BYTES
            count += 1
        paged_conc.append(count)
        paged_util.append(useful_bytes / used_bytes if used_bytes else 0)

    return {
        "budget_mb": budget_mb,
        "contiguous_concurrency": contiguous_concurrency,
        "paged_concurrency_mean": float(np.mean(paged_conc)),
        "paged_concurrency_p50": float(np.percentile(paged_conc, 50)),
        "paged_memory_utilization": float(np.mean(paged_util)),
        "speedup_concurrency": float(np.mean(paged_conc)) / max(contiguous_concurrency, 1),
    }


def main():
    print(f"每 token KV: {PER_TOKEN_BYTES/1024:.1f} KB | "
          f"连续方案每请求预分配: {PER_SEQ_CONTIGUOUS/1024/1024:.1f} MB (maxseq={MAXSEQ})")
    print("-" * 78)
    print(f"{'KV预算':>7} | {'连续并发':>8} | {'分页并发(均)':>12} | "
          f"{'分页内存利用率':>14} | {'并发提升':>8}")
    for mb in [256, 512, 1024, 2048]:
        r = run(mb)
        print(f"{mb:>6}MB | {r['contiguous_concurrency']:>8} | "
              f"{r['paged_concurrency_mean']:>12.1f} | "
              f"{r['paged_memory_utilization']*100:>13.1f}% | "
              f"{r['speedup_concurrency']:>7.1f}x")
    print("-" * 78)
    print("说明：连续方案每请求固定占 maxseq，短请求浪费大量槽位（内部碎片）；")
    print("分页方案按 16-token 块在共享池中按需分配，浪费仅限块内，故并发显著提升。")


if __name__ == "__main__":
    main()
