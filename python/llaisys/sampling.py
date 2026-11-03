"""采样算子：temperature / top-k / top-p（nucleus）。
用 numpy 实现，可独立用于对 logits 采样，也可作为采样逻辑的参考实现。
注意：Qwen2 模型的 C++ 后端已内置等价采样（infer_with_sampling），
本模块用于独立测试、自定义采样与讲解。
"""
import numpy as np


def sample(logits, temperature: float = 1.0, top_k: int = 0,
           top_p: float = 1.0, greedy: bool = False,
           rng: np.random.Generator = None):
    """从 logits 中采样一个 token。

    参数：
        logits:      1-D 数组，未归一化的对数概率（词表大小）。
        temperature: 温度；越小越确定，<1 更集中，>1 更发散。
        top_k:       >0 时只保留概率最高的 k 个候选；0 表示不启用。
        top_p:       <1 时只保留累计概率达到 p 的最小候选集（nucleus）。
        greedy:      True 时直接取 argmax（贪心）。
        rng:         可选 numpy 随机数生成器（便于复现）。
    返回：
        采样到的 token id（int）。
    """
    logits = np.asarray(logits, dtype=np.float64)
    if greedy:
        return int(np.argmax(logits))

    # 1) 温度缩放：logits / temperature。
    logits = logits / max(temperature, 1e-6)

    # 2) Top-K：只保留最大的 k 个 logits，其余置 -inf。
    if top_k > 0:
        top_k = min(top_k, logits.shape[0])
        idx = np.argpartition(logits, -top_k)[-top_k:]  # 未排序的 top-k 索引
        logits_full = np.full_like(logits, -np.inf)
        logits_full[idx] = logits[idx]
        logits = logits_full

    # 3) softmax（减去最大值，保证数值稳定）。
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    # 4) Top-P（nucleus）：按概率降序，保留累计概率达到 top_p 的最小集合。
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]          # 概率降序的索引
        cum = np.cumsum(probs[order])            # 累计概率
        # 保留累计概率（不含当前项）<= top_p 的候选
        keep = order[cum - probs[order] <= top_p]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()                # 重新归一化

    # 5) 按概率分布随机采样。
    rng = rng or np.random.default_rng()
    return int(rng.choice(len(probs), p=probs))
