"""Print the local runtime used for reproducible benchmark records."""

from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / (1024**3)
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {memory_gib:.1f} GiB")
    else:
        print("WARNING: CUDA is unavailable; GPU experiments must not be benchmarked.")


if __name__ == "__main__":
    main()
