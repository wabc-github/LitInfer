"""静态批处理 vs 连续批处理（continuous batching）吞吐对比（离散事件模拟）。

核心思想（面试主线）：
- 一次 decode 前向的耗时与批大小"近似无关"（GPU/CPU 一次算一批）。
- 静态批处理：一组请求必须等最慢的（max L）才一起结束，短请求完成后"空转"占槽。
- 连续批处理：请求一完成立即释放槽位，立刻塞入等待请求，槽位不空转。

运行：
    python scripts/continuous_batching_compare.py
"""
import numpy as np


def static_schedule(lengths, B):
    """静态批：每 B 个一组，组耗时 = 组内 max L；逐组串行。返回总 step 数。"""
    total_steps = 0
    groups = 0
    for i in range(0, len(lengths), B):
        group = lengths[i:i + B]
        total_steps += int(np.max(group))
        groups += 1
    return total_steps, groups


def continuous_schedule(lengths, B):
    """连续批：B 个槽位，完成即换新。返回总 step 数。"""
    remaining = list(lengths)  # 等待队列
    slots = []                 # 槽位中请求的剩余长度
    steps = 0
    idx = 0
    # 初始填满
    while len(slots) < B and idx < len(remaining):
        slots.append(remaining[idx]); idx += 1
    while slots:
        steps += 1
        # 每个槽位推进一步。
        new_slots = []
        for r in slots:
            if r - 1 > 0:
                new_slots.append(r - 1)
        slots = new_slots
        # 空出的槽位立刻塞入等待请求。
        while len(slots) < B and idx < len(remaining):
            slots.append(remaining[idx]); idx += 1
    return steps


def run(n_requests, B, seed=0):
    rng = np.random.default_rng(seed)
    # 输出长度分布：多数短、少数长（均值约 80，最长 ~400）。
    lengths = rng.exponential(scale=80.0, size=n_requests) + 10
    lengths = np.minimum(lengths, 400).astype(int).tolist()
    total_tokens = int(sum(lengths))

    s_steps, _ = static_schedule(lengths, B)
    c_steps = continuous_schedule(lengths, B)
    return {
        "n": n_requests, "B": B,
        "total_tokens": total_tokens,
        "static_steps": s_steps,
        "continuous_steps": c_steps,
        "static_tput": total_tokens / s_steps,
        "continuous_tput": total_tokens / c_steps,
        "speedup": s_steps / c_steps,
    }


def main():
    print("假设：一次 decode 前向耗时与批大小近似无关；每 step 处理一个 decode token。")
    print("-" * 80)
    print(f"{'请求数':>6} {'并发B':>6} | {'静态步数':>8} {'连续步数':>8} | "
          f"{'静态吞吐':>8} {'连续吞吐':>8} | {'加速':>6}")
    for n in [32, 64, 128]:
        for B in [4, 8, 16]:
            r = run(n, B, seed=n + B)
            print(f"{n:>6} {B:>6} | {r['static_steps']:>8} {r['continuous_steps']:>8} | "
                  f"{r['static_tput']:>8.2f} {r['continuous_tput']:>8.2f} | "
                  f"{r['speedup']:>5.2f}x")
    print("-" * 80)
    print("连续批处理避免了短请求空转，且请求长度方差越大、并发越高，收益越明显。")
    print("（实际引擎中还需配合分页 KV：新请求进入时按需分配块，退出时归还。）")


if __name__ == "__main__":
    main()
