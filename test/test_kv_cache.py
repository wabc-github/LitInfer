"""Model-level KV Cache regression checks.

Run with a local model directory:
    python test/test_kv_cache.py --model ../DeepSeek-R1-Distill-Qwen-1.5B

The checks intentionally use short prompts. They verify cache reset between
requests and the prefill/decode path without requiring a second model copy.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain KV Cache briefly.")
    parser.add_argument("--max_steps", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    import llaisys
    from llaisys.libllaisys import LIB_LLAISYS

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_tokens = list(tokenizer.encode(prompt_text))
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.CPU)

    def infer_with_cache(tokens):
        values = (ctypes.c_int64 * len(tokens))(*tokens)
        token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            model.model, values, len(tokens)
        )
        assert token >= 0, "native inference failed during KV regression"
        return int(token)

    # A multi-token prefill must produce the same last-position logits as
    # feeding the same prefix one token at a time into an empty cache.
    short_prefix = input_tokens[: min(3, len(input_tokens))]
    LIB_LLAISYS.llaisysQwen2ModelInitKVCache(model.model)
    try:
        prefill_next = infer_with_cache(short_prefix)
    finally:
        LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(model.model)

    LIB_LLAISYS.llaisysQwen2ModelInitKVCache(model.model)
    try:
        sequential_next = None
        for token in short_prefix:
            sequential_next = infer_with_cache([token])
        assert sequential_next == prefill_next, (
            "multi-token prefill and token-by-token cache disagree"
        )
        # Decode accepts exactly the previous generated token and advances the
        # cache by one position without requiring a second prefill.
        decode_next = infer_with_cache([prefill_next])
        assert decode_next >= 0
    finally:
        LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(model.model)

    first, first_metrics = model.generate_profile(
        input_tokens, max_new_tokens=args.max_steps, top_k=1, top_p=1.0, temperature=1.0
    )
    second, second_metrics = model.generate_profile(
        input_tokens, max_new_tokens=args.max_steps, top_k=1, top_p=1.0, temperature=1.0
    )

    assert first == second, "cache reset changed deterministic output"
    assert first_metrics["generated_tokens"] == len(first)
    assert second_metrics["generated_tokens"] == len(second)
    assert first_metrics["first_token_seconds"] >= 0.0
    assert second_metrics["first_token_seconds"] >= 0.0

    overflow_tokens = [input_tokens[0]] * (int(model.meta.maxseq) + 1)
    try:
        model.generate_profile(overflow_tokens, max_new_tokens=1)
    except ValueError as exc:
        assert "max sequence length" in str(exc)
    else:
        raise AssertionError("overlong prompt was not rejected")

    print("KV cache reset regression passed")
    print(f"generated tokens: {len(first)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
