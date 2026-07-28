import argparse
import subprocess
import sys
from pathlib import Path


OPS = [
    "add.py",
    "argmax.py",
    "embedding.py",
    "linear.py",
    "rms_norm.py",
    "rope.py",
    "self_attention.py",
    "swiglu.py",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    ops_dir = Path(__file__).parent / "ops"
    for op in OPS:
        cmd = [sys.executable, str(ops_dir / op), "--device", args.device]
        if args.profile:
            cmd.append("--profile")
        print(f"\n=== Running {op} ===", flush=True)
        subprocess.run(cmd, check=True)

    print("\n\033[92mAll operator tests passed!\033[0m\n")
