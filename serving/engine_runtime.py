"""连续批处理引擎的运行时封装（后台调度线程 + 请求 Future）。

职责：
- 持有一个已加载权重的 Qwen2 模型（共享一份权重）。
- 后台线程循环调用 engine_step：所有未完成请求同时做一次前向。
- 每提交一条请求返回一个 Future，请求完成时 set_result(生成 token)。

这是"非流式"连续批处理入口；逐 token 流式 SSE 与引擎的深度集成
（需要每步快照，见 dev_process_log 的待办）。
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Sequence


class EngineRuntime:
    def __init__(self, model, poll_interval: float = 0.01,
                 top_k: int = 1, top_p: float = 1.0, temperature: float = 1.0):
        self._model = model
        self._poll = poll_interval
        self._top_k, self._top_p, self._temperature = top_k, top_p, temperature
        self._cv = threading.Condition()
        self._futures: dict[int, Future] = {}
        self._stop = False
        self._model.engine_start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, prompt_ids: Sequence[int], max_new_tokens: int = 128) -> Future:
        rid = self._model.engine_add_request(prompt_ids, max_new_tokens)
        if rid < 0:
            raise RuntimeError("engine_add_request failed")
        fut: Future = Future()
        with self._cv:
            self._futures[int(rid)] = fut
        with self._cv:
            self._cv.notify_all()
        return fut

    def _loop(self):
        while True:
            with self._cv:
                if self._stop:
                    return
                has_work = self._model.engine_has_unfinished()
                if not has_work:
                    self._cv.wait(self._poll)
                    continue
            # 推进一次批量前向（所有未完成请求共享一次前向）。
            self._model.engine_step(self._top_k, self._top_p, self._temperature)
            # 收集本步完成的请求。
            with self._cv:
                done = [rid for rid in self._futures
                        if self._model.engine_is_finished(rid)]
            for rid in done:
                out = self._model.engine_get_output(rid)
                with self._cv:
                    fut = self._futures.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(out)

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        self._model.engine_stop()
