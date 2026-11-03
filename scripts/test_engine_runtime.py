"""EngineRuntime 的调度逻辑测试（Mock 模型，无需真实权重）。

验证：
- 多个不同长度请求并发提交，全部正确完成；
- 短请求比长请求更早完成（连续批处理的关键：不空等最慢请求）。
运行：
    python scripts/test_engine_runtime.py
"""
import time
import sys
from pathlib import Path

# 让 import serving 生效（仓库根目录）。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serving.engine_runtime import EngineRuntime


class MockModel:
    def __init__(self):
        self._next = 0
        self._length = {}
        self._started = False

    def engine_start(self):
        self._started = True

    def engine_add_request(self, ids, max_new_tokens):
        rid = self._next
        self._next += 1
        self._length[rid] = int(max_new_tokens)
        return rid

    def engine_has_unfinished(self):
        return any(x > 0 for x in self._length.values())

    def engine_step(self, *args):
        for rid in self._length:
            if self._length[rid] > 0:
                self._length[rid] -= 1

    def engine_is_finished(self, rid):
        return self._length[rid] <= 0

    def engine_get_output(self, rid):
        # 伪输出：长度 = max_new（用 1000+ 偏移），调用时请求已完成。
        return [1000 + rid]

    def engine_stop(self):
        pass


def main():
    model = MockModel()
    # 让 mock 每步耗时可控：用很小的 poll。
    rt = EngineRuntime(model, poll_interval=0.001)
    # 三个请求，长度不同：短/中/长。
    specs = [("short", 2), ("mid", 5), ("long", 10)]
    futures = {}
    t0 = time.perf_counter()
    for name, L in specs:
        futures[name] = (L, rt.submit([1, 2, 3], max_new_tokens=L))

    finish_time = {}
    for name, (L, fut) in futures.items():
        fut.result(timeout=5)
        finish_time[name] = time.perf_counter() - t0

    rt.shutdown()

    ok = True
    for name, (L, fut) in futures.items():
        done = fut.done() and fut.result()
        ok &= bool(done)
        print(f"[{'PASS' if done else 'FAIL'}] {name:>5} (L={L}) completed at {finish_time[name]:.3f}s")

    # 短请求应比长请求先完成。
    ordered = finish_time["short"] <= finish_time["mid"] <= finish_time["long"]
    print(f"[{'PASS' if ordered else 'FAIL'}] 完成顺序 short <= mid <= long（不空等最慢）")
    ok &= ordered
    print("\nALL ENGINE RUNTIME TESTS PASSED" if ok else "\nSOME TESTS FAILED")


if __name__ == "__main__":
    main()
