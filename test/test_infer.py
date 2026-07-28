import faulthandler, signal
faulthandler.enable()
# Some platforms or Python builds disallow registering SIGSEGV/SIGABRT with faulthandler.
# Wrap in try/except and fall back to enable() only if register is not allowed.
try:
    faulthandler.register(signal.SIGSEGV, chain=True, all_threads=True)
except (RuntimeError, ValueError, AttributeError):
    pass
try:
    faulthandler.register(signal.SIGABRT, chain=True, all_threads=True)
except (RuntimeError, ValueError, AttributeError):
    pass
import gc

import argparse
import torch
import os
import time
import sys
import io
import json
import ctypes

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    raise ValueError(f"Unsupported device name: {device_name}")


def llaisys_device(device_name: str):
    import llaisys

    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device name: {device_name}")


def load_hf_model(model_path=None, device_name="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    print("Loading model...\n", flush=True)
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if device_name == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(torch_device(device_name))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=torch_device(device_name),
            trust_remote_code=True,
        )

    print("Model loaded.\n", flush=True)

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    do_sample = not (top_k == 1 and top_p >= 1.0 and temperature == 1.0)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
        )
    elapsed = time.perf_counter() - start
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_tokens = outputs[0].tolist()
    input_tokens = inputs[0].tolist()
    new_tokens = full_tokens[len(input_tokens):]
    return input_tokens, new_tokens, full_tokens, result, elapsed


def load_llaisys_model(model_path, device_name):
    import llaisys

    print("Loading llaisys model...\n",flush=True)
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    print("llaisys model loaded.\n",flush=True)
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs, metrics = model.generate_profile(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return inputs, outputs, tokenizer.decode(outputs, skip_special_tokens=True), metrics


def current_rss_mb():
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
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        if not get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.WorkingSetSize / 1024 / 1024

    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--align_log", default=None, type=str)
    parser.add_argument("--skip_hf", action="store_true", help="Skip PyTorch reference and run LLAISYS only.")

    args = parser.parse_args()
    if args.skip_hf and args.test:
        parser.error("--test requires PyTorch reference; remove --skip_hf or remove --test.")

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    if args.skip_hf:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model_path = args.model
        input_tokens = hf_new_tokens = hf_output = hf_elapsed = None
    else:
        tokenizer, model, model_path = load_hf_model(args.model, args.device)

        input_tokens, hf_new_tokens, hf_full_tokens, hf_output, hf_elapsed = hf_infer(
            args.prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )

        del model
        gc.collect()

        print("\n=== PyTorch Reference ===\n")
        print("Input tokens:")
        print(input_tokens)
        print("Generated tokens:")
        print(hf_new_tokens)
        print("\nContents:")
        print(hf_output)
        print("\n")
        print(f"PyTorch elapsed: {hf_elapsed:.2f}s\n")

    model = load_llaisys_model(model_path, args.device)
    rss_before = current_rss_mb()
    start_time = time.perf_counter()
    llaisys_input_tokens, llaisys_tokens, llaisys_output, metrics = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    end_time = time.perf_counter()
    rss_after = current_rss_mb()

    print("\n=== Your Result ===\n")
    print("Input tokens:")
    print(llaisys_input_tokens)
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")
    print("=== LLAISYS Benchmark ===")
    print(f"First token latency: {metrics['first_token_seconds'] * 1000:.2f} ms")
    print(f"Decode elapsed: {metrics['decode_seconds']:.2f} s")
    print(f"Decode token/s: {metrics['decode_tokens_per_second']:.4f}")
    print(f"Generated tokens: {metrics['generated_tokens']}")
    if rss_before is not None and rss_after is not None:
        print(f"RSS before inference: {rss_before:.2f} MB")
        print(f"RSS after inference: {rss_after:.2f} MB")
        print(f"RSS delta: {rss_after - rss_before:.2f} MB")

    if args.align_log:
        os.makedirs(os.path.dirname(args.align_log) or ".", exist_ok=True)
        record = {
            "prompt": args.prompt,
            "input_tokens": input_tokens,
            "pytorch_new_tokens": hf_new_tokens,
            "llaisys_new_tokens": llaisys_tokens,
            "matched": None if args.skip_hf else llaisys_tokens == hf_new_tokens,
            "llaisys_metrics": metrics,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
        }
        with open(args.align_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.test:
        if args.skip_hf:
            raise RuntimeError("--test requires PyTorch reference; remove --skip_hf.")
        assert llaisys_input_tokens == input_tokens
        assert llaisys_tokens == hf_new_tokens
        print("\033[92mTest passed!\033[0m\n")
