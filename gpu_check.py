"""Display available GPU info: devices, driver, CUDA, memory (total/used/free)."""
from __future__ import annotations

import shutil
import subprocess
import sys


def run_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"nvidia-smi failed: {e}"


def print_nvidia_smi_table(raw: str) -> None:
    print("=" * 90)
    print("NVIDIA-SMI")
    print("=" * 90)
    header = f"{'idx':<4}{'name':<30}{'driver':<12}{'total(MiB)':>12}{'used(MiB)':>11}{'free(MiB)':>11}{'util%':>7}{'temp°C':>8}"
    print(header)
    print("-" * 90)
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            print(line)
            continue
        idx, name, drv, total, used, free, util, temp = parts
        print(f"{idx:<4}{name[:29]:<30}{drv:<12}{total:>12}{used:>11}{free:>11}{util:>7}{temp:>8}")


def print_torch_info() -> None:
    print()
    print("=" * 90)
    print("PyTorch CUDA")
    print("=" * 90)
    try:
        import torch
    except ImportError:
        print("torch not installed")
        return

    print(f"torch version       : {torch.__version__}")
    print(f"cuda available      : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(f"cuda compiled ver   : {torch.version.cuda}")
        return

    print(f"cuda runtime version: {torch.version.cuda}")
    print(f"cudnn version       : {torch.backends.cudnn.version()}")
    print(f"device count        : {torch.cuda.device_count()}")
    print("-" * 90)
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        free_mb = free / (1024 ** 2)
        total_mb = total / (1024 ** 2)
        used_mb = total_mb - free_mb
        print(
            f"[{i}] {props.name}  | capability sm_{props.major}{props.minor}  | "
            f"total={total_mb:,.0f} MiB  used={used_mb:,.0f} MiB  free={free_mb:,.0f} MiB"
        )


def main() -> int:
    raw = run_nvidia_smi()
    if raw is None:
        print("nvidia-smi not found on PATH — NVIDIA driver may not be installed.")
    elif raw.startswith("nvidia-smi failed"):
        print(raw)
    else:
        print_nvidia_smi_table(raw)

    print_torch_info()
    return 0


if __name__ == "__main__":
    sys.exit(main())
