"""Reproducible CPU benchmark for the local LLAISYS Qwen2 backend.

Example (PowerShell):
    python scripts/benchmark_cpu.py `
      --model ../DeepSeek-R1-Distill-Qwen-1.5B `
      --device cpu --max_steps 8 --repeats 3 --threads 1 `
      --output benchmarks/cpu_baseline.json

The default run measures LLAISYS only. Add --align to run a separate greedy
PyTorch reference pass and record token alignment without mixing its time into
the LLAISYS measurements.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


def current_rss_mb() -> Optional[float]:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]
        get_info.restype = ctypes.c_int
        if get_info(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1024.0 / 1024.0
        return None

    if sys.platform.startswith("linux"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * page_size / 1024.0 / 1024.0
        except (OSError, IndexError, ValueError):
            pass

    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; this is a fallback for platforms without a
        # current-RSS API above.
        return value / 1024.0 if value < 10**9 else value / 1024.0 / 1024.0
    except Exception:
        return None


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=ROOT,
            check=False,
        )
        return result.returncode != 0
    except OSError:
        return None


def compiler_description() -> str:
    configured = os.environ.get("CXX") or os.environ.get("CC")
    if configured:
        return configured
    if os.name == "nt":
        return "MSVC via xmake"
    return "C++ compiler via xmake"


def cpu_description() -> str:
    value = platform.processor() or platform.uname().processor
    return value or platform.machine()


def build_prompt_tokens(tokenizer: Any, prompt: str) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return list(tokenizer.encode(text))
    return list(tokenizer.encode(prompt))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_reference(
    model_path: str, tokenizer: Any, prompt: str, max_steps: int
) -> List[int]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(torch.device("cpu"))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_steps,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            do_sample=False,
        )
    input_len = inputs.shape[1]
    result = outputs[0].tolist()[input_len:]
    del model
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--max_steps", default=8, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--threads", default=None, type=int)
    parser.add_argument("--quantize", default=None, choices=[None, "int8"],
                        help="weight-only 量化：'int8'（与模型 dtype 兼容）")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--output", default="benchmarks/cpu_benchmark.json")
    args = parser.parse_args()

    if args.max_steps <= 0 or args.repeats <= 0:
        parser.error("--max_steps and --repeats must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")

    if args.threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    sys.path.insert(0, str(ROOT / "python"))
    from transformers import AutoTokenizer
    import llaisys
    from llaisys import DataType

    prompts = args.prompts or [
        "Who are you?",
        "Explain KV Cache in one paragraph.",
        "Write a short C++ function that adds two integers.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.CPU,
                                 quantize=args.quantize)

    encoded = [build_prompt_tokens(tokenizer, prompt) for prompt in prompts]
    # Warm up the shared library and allocator once. Warm-up is not included
    # in samples, and every measured call resets the model KV cache itself.
    model.generate_profile(encoded[0], max_new_tokens=args.max_steps)

    samples: List[Dict[str, Any]] = []
    for prompt_index, (prompt, input_tokens) in enumerate(zip(prompts, encoded)):
        for repeat in range(args.repeats):
            rss_before = current_rss_mb()
            start = time.perf_counter()
            generated, metrics = model.generate_profile(
                input_tokens,
                max_new_tokens=args.max_steps,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
            )
            elapsed = time.perf_counter() - start
            rss_after = current_rss_mb()
            samples.append(
                {
                    "prompt_index": prompt_index,
                    "repeat": repeat,
                    "prompt": prompt,
                    "input_tokens": len(input_tokens),
                    "generated_tokens": list(generated),
                    "elapsed_seconds": elapsed,
                    "metrics": metrics,
                    "rss_before_mb": rss_before,
                    "rss_after_mb": rss_after,
                    "rss_delta_mb": (
                        rss_after - rss_before
                        if rss_before is not None and rss_after is not None
                        else None
                    ),
                    "ttft_ms": metrics["first_token_seconds"] * 1000.0,
                    "decode_elapsed_seconds": metrics["decode_seconds"],
                    "decode_token_per_second": metrics["decode_tokens_per_second"],
                    "tpot_ms": metrics["time_per_decode_token_seconds"] * 1000.0,
                    "aligned": None,
                }
            )

    alignment: Dict[str, Any] = {"status": "not_requested"}
    if args.align:
        try:
            alignment["status"] = "passed"
            alignment["prompts"] = []
            for prompt_index, prompt in enumerate(prompts):
                reference = run_reference(args.model, tokenizer, prompt, args.max_steps)
                actual = samples[prompt_index * args.repeats]["generated_tokens"]
                matched = actual == reference
                if not matched:
                    alignment["status"] = "failed"
                for sample in samples:
                    if sample["prompt_index"] == prompt_index:
                        sample["aligned"] = matched
                alignment["prompts"].append(
                    {
                        "prompt_index": prompt_index,
                        "pytorch_new_tokens": reference,
                        "llaisys_new_tokens": actual,
                        "matched": matched,
                    }
                )
        except Exception as exc:  # reference is optional for CPU timing
            alignment = {"status": "error", "error": repr(exc)}

    prompt_summaries = []
    for prompt_index, prompt in enumerate(prompts):
        rows = [s for s in samples if s["prompt_index"] == prompt_index]
        prompt_summaries.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "ttft_ms": summarize(
                    [r["metrics"]["first_token_seconds"] * 1000 for r in rows]
                ),
                "decode_token_per_second": summarize(
                    [r["metrics"]["decode_tokens_per_second"] for r in rows]
                ),
                "decode_seconds": summarize(
                    [r["metrics"]["decode_seconds"] for r in rows]
                ),
                "tpot_ms": summarize(
                    [r["metrics"]["time_per_decode_token_seconds"] * 1000.0
                     for r in rows
                     if r["metrics"]["decode_tokens"] > 0]
                ),
                "rss_delta_mb": summarize(
                    [r["rss_delta_mb"] for r in rows if r["rss_delta_mb"] is not None]
                ),
            }
        )

    result = {
        "schema_version": 1,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "os": platform.platform(),
            "python": sys.version,
            "python_compiler": platform.python_compiler(),
            "compiler": compiler_description(),
            "processor": cpu_description(),
            "cpu_count": os.cpu_count(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "device": args.device,
            "model": str(Path(args.model).resolve()),
            "dtype": DataType(int(model.meta.dtype)).name,
            "quantize": args.quantize,
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "max_steps": args.max_steps,
            "repeats": args.repeats,
            "alignment_requested": args.align,
        },
        "alignment": alignment,
        "summary": prompt_summaries,
        "samples": samples,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote benchmark: {output}")
    for summary in prompt_summaries:
        tpot = summary["tpot_ms"].get("median")
        tpot_text = f"TPOT median={tpot:.2f} ms" if tpot is not None else "TPOT n/a"
        print(
            f"prompt[{summary['prompt_index']}] "
            f"TTFT median={summary['ttft_ms']['median']:.2f} ms, "
            f"decode median={summary['decode_token_per_second']['median']:.4f} token/s, "
            f"{tpot_text}"
        )
    print(f"alignment: {alignment['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
