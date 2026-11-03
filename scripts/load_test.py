"""并发压测：对 OpenAI 兼容服务发 N 个并发请求，测聚合吞吐与延迟分位。

用于对比「静态批处理（serving/server.py，端口 8000）」与「连续批处理
（serving/engine_server.py，端口 8001）」在相同并发、相同请求下的吞吐 / 延迟。

用法（先起服务，再跑本脚本）：
    python -m serving.engine_server --model-path ../DeepSeek-R1-Distill-Qwen-1.5B --port 8001
    python scripts/load_test.py --url http://127.0.0.1:8001/v1/chat/completions `
        --concurrency 8 --requests 32 --output benchmarks/load_cb.json

只用标准库（http.client + ThreadPoolExecutor），无需额外依赖。
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request
from urllib.error import URLError


def post_chat(url: str, message: str, max_tokens: int, timeout: float):
    body = json.dumps({
        "model": "qwen2",
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": repr(exc),
                "latency_s": time.perf_counter() - t0}
    latency = time.perf_counter() - t0
    usage = payload.get("usage", {})
    content = ""
    choices = payload.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    return {
        "ok": "error" not in payload,
        "latency_s": latency,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "content_chars": len(content),
    }


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    # 最近秩百分位（样本少时是近似值）。
    i = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", default=8, type=int)
    ap.add_argument("--requests", default=32, type=int)
    ap.add_argument("--message",
                    default="Write a short C++ function that adds two integers.")
    ap.add_argument("--max_tokens", default=64, type=int)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    wall_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(post_chat, args.url, args.message, args.max_tokens,
                        timeout=600)
            for _ in range(args.requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - wall_start

    oks = [r for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]
    total_tokens = sum(r.get("completion_tokens", 0) for r in oks)
    latencies = sorted(r["latency_s"] for r in oks)

    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "ok": len(oks),
        "errors": len(errors),
        "wall_seconds": round(wall, 3),
        "aggregate_tokens_per_second": round(total_tokens / wall, 3) if wall else None,
        "total_completion_tokens": total_tokens,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p99_s": percentile(latencies, 0.99),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "results": results},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
