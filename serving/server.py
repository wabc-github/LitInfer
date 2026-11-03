"""OpenAI 兼容的推理服务：FastAPI + SSE。

启动（在仓库根目录，建议用含 transformers 的 conda 环境，如 mine）：
    python -m serving.server --model ./models/deepseek-r1-distill-qwen-1.5b --host 0.0.0.0 --port 8000
    # INT8 量化：
    python -m serving.server --model ./models/xxx --quantize int8

调用：
    # 非流式
    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"你好"}],"stream":false}'
    # 流式
    curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
"""
import argparse
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llaisys.models.qwen2 import Qwen2
from transformers import AutoTokenizer

# 服务内部的模型名（OpenAI 接口的 model 字段）
SERVED_MODEL_NAME = "litinfer-qwen2"

# 全局对象（在启动时初始化）
STATE = {"model": None, "tokenizer": None}

app = FastAPI(title="LitInfer OpenAI-compatible server")


# ---------------- 请求/响应模型 ----------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = Field(default=128, alias="max_tokens")
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 1

    class Config:
        populate_by_name = True


# ---------------- 工具函数 ----------------

def build_input_ids(messages: list[ChatMessage], tokenizer):
    """把 OpenAI 风格 messages 转成 token ids（应用 chat template）。"""
    conversation = [{"role": m.role, "content": m.content} for m in messages]
    prompt_text = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(prompt_text)


def sse_chunk(content: str, finish_reason=None, created=0, chunk_id=""):
    """构造一个 OpenAI chat.completion.chunk 的 SSE 数据帧。"""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------- 路由 ----------------

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "litinfer"}
    ]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    model: Qwen2 = STATE["model"]
    tokenizer = STATE["tokenizer"]

    input_ids = build_input_ids(req.messages, tokenizer)
    created = int(time.time())
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # ---- 流式 ----
    if req.stream:
        def event_generator():
            output_ids = []
            seen_text = ""
            # 逐 token 生成，token id 由模型生成器 yield。
            gen = model.generate_stream(
                input_ids,
                max_new_tokens=req.max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
            )
            for nid in gen:
                output_ids.append(nid)
                # 累计 decode 取增量，保证多字节字符边界正确。
                full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                delta = full_text[len(seen_text):]
                seen_text = full_text
                if delta:
                    yield sse_chunk(delta, created=created, chunk_id=chunk_id)
            # 结束帧：finish_reason=stop，再发 [DONE]。
            yield sse_chunk("", finish_reason="stop", created=created, chunk_id=chunk_id)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---- 非流式 ----
    output_ids = list(model.generate_stream(
        input_ids,
        max_new_tokens=req.max_tokens,
        top_k=req.top_k,
        top_p=req.top_p,
        temperature=req.temperature,
    ))
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return JSONResponse({
        "id": chunk_id,
        "object": "chat.completion",
        "created": created,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids),
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型目录（含 config/tokenizer/safetensors）")
    parser.add_argument("--quantize", default=None, choices=[None, "int8"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}", flush=True)
    STATE["tokenizer"] = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading LitInfer model (quantize={args.quantize}) ...", flush=True)
    STATE["model"] = Qwen2(args.model, quantize=args.quantize)
    print("Server ready.", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
