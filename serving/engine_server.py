"""连续批处理版 OpenAI 兼容服务（非流式）。

与 server.py 的区别：
- server.py 每个请求独立用 generate_stream（教学/流式，请求间不共享一次前向）。
- engine_server.py 所有请求进入同一个 EngineRuntime，后台线程把它们
  拼成一批一起做前向；请求完成即释放槽位、塞入新请求（连续批处理）。

并发压测（示例）：
    for i in 1 2 3 4 5 6 7 8; do
        curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
          -d '{"model":"qwen2","messages":[{"role":"user","content":"..."}]}' &
    done
它们会被自动批处理，而不是逐个串行。

启动：
    python -m serving.engine_server --model-path <本地模型路径>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from llaisys.models.qwen2 import Qwen2
from serving.engine_runtime import EngineRuntime

app = FastAPI(title="LitInfer Continuous-Batching Engine")

STATE = {"model": None, "tokenizer": None, "runtime": None, "model_name": "qwen2"}


def _build_input(messages):
    """把对话套 chat 模板并编码为 token id。"""
    text = STATE["tokenizer"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return STATE["tokenizer"].encode(text)


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": STATE["model_name"], "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: dict):
    messages = req.get("messages", [])
    max_new = int(req.get("max_tokens", 128))
    if not messages:
        return JSONResponse({"error": "messages is required"}, status_code=400)

    prompt_ids = _build_input(messages)
    t0 = time.perf_counter()
    fut = STATE["runtime"].submit(prompt_ids, max_new_tokens=max_new)
    output_ids = fut.result()  # 等该请求完成
    latency = time.perf_counter() - t0

    text = STATE["tokenizer"].decode(output_ids, skip_special_tokens=True)
    return {
        "id": "chatcmpl-batch",
        "object": "chat.completion",
        "model": STATE["model_name"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt_ids),
                  "completion_tokens": len(output_ids),
                  "total_tokens": len(prompt_ids) + len(output_ids),
                  "latency_seconds": latency},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--quantize", default=None, help="可选 int8")
    args = ap.parse_args()

    print("Loading model for continuous batching...", flush=True)
    model = Qwen2(args.model_path, quantize=args.quantize)
    STATE["model"] = model
    STATE["tokenizer"] = AutoTokenizer.from_pretrained(args.model_path)
    STATE["model_name"] = Path(args.model_path).name
    STATE["runtime"] = EngineRuntime(model)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
