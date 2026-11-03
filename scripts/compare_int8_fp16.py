"""FP16 vs INT8 生成 token 一致性对比（weight-only 量化质量指标）。

同一 prompt 分别用 FP16 权重和 INT8 weight-only 量化权重做 greedy 生成，
逐 token 比较，得到 INT8 相对 FP16 的一致率。一致率越高，说明量化对
生成质量影响越小（省一半权重内存，代价是少量 token 翻转）。

运行（需含 transformers 的环境，如 conda `mine`）：
    python scripts/compare_int8_fp16.py `
      --model ../DeepSeek-R1-Distill-Qwen-1.5B `
      --max_steps 32 --repeats 1 --output benchmarks/int8_vs_fp16.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encode(tokenizer, prompt: str):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False)
    return list(tokenizer.encode(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_steps", default=32, type=int)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--output", default="benchmarks/int8_vs_fp16.json")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "python"))
    from transformers import AutoTokenizer
    import llaisys

    prompts = args.prompts or [
        "Who are you?",
        "Explain KV Cache in one paragraph.",
        "Write a short C++ function that adds two integers.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 两个模型不常驻（权重各占一份内存）：逐个 prompt、逐个 repeat 分别
    # 建 FP16 跑完销毁，再建 INT8 跑完销毁。
    samples = []
    for prompt in prompts:
        input_ids = encode(tokenizer, prompt)
        for repeat in range(args.repeats):
            model_fp16 = llaisys.models.Qwen2(args.model, llaisys.DeviceType.CPU)
            fp16_tokens, _ = model_fp16.generate_profile(
                input_ids, max_new_tokens=args.max_steps,
                top_k=1, top_p=1.0, temperature=1.0)
            del model_fp16

            model_int8 = llaisys.models.Qwen2(
                args.model, llaisys.DeviceType.CPU, quantize="int8")
            int8_tokens, _ = model_int8.generate_profile(
                input_ids, max_new_tokens=args.max_steps,
                top_k=1, top_p=1.0, temperature=1.0)
            del model_int8

            n = min(len(fp16_tokens), len(int8_tokens))
            matched = sum(
                1 for i in range(n) if fp16_tokens[i] == int8_tokens[i])
            samples.append({
                "prompt": prompt,
                "repeat": repeat,
                "input_tokens": len(input_ids),
                "fp16_tokens": fp16_tokens,
                "int8_tokens": int8_tokens,
                "fp16_len": len(fp16_tokens),
                "int8_len": len(int8_tokens),
                "matched": matched,
                "compared": n,
                "match_rate": (matched / n) if n else None,
            })

    total_matched = sum(s["matched"] for s in samples)
    total_compared = sum(s["compared"] for s in samples)
    result = {
        "schema_version": 1,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(Path(args.model).resolve()),
            "max_steps": args.max_steps,
            "repeats": args.repeats,
            "sample_strategy": "greedy (top_k=1, top_p=1.0, temperature=1.0)",
        },
        "overall": {
            "matched_tokens": total_matched,
            "compared_tokens": total_compared,
            "match_rate": (total_matched / total_compared) if total_compared else None,
        },
        "samples": samples,
    }

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Wrote {out}")
    if total_compared:
        rate = 100.0 * total_matched / total_compared
        print(f"INT8 vs FP16 token 一致率: {total_matched}/{total_compared} "
              f"= {rate:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
