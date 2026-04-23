"""
Environment bootstrap for every notebook.

Imported once per notebook via:

    from utils.env import bootstrap
    bootstrap()

Works identically in Google Colab and locally. It:
  1. Detects whether we're running in Colab.
  2. Reports torch / CUDA / device info.
  3. Seeds Python, NumPy, and torch RNGs for reproducibility.
  4. Returns the torch device so notebooks can use it immediately.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Repo location (used by the Colab setup cell in every notebook).
# ---------------------------------------------------------------------------
GITHUB_REPO = "https://github.com/numberonewastefellow/my_learning.git"
GITHUB_REPO_DIR = "my_learning"  # the folder name `git clone` creates

# For a private repo, use:
#   "https://<TOKEN>@github.com/numberonewastefellow/my_learning.git"


@dataclass
class EnvInfo:
    in_colab: bool
    torch_version: str
    cuda_available: bool
    device_name: str
    device: "object"  # torch.device — avoid importing torch at module top-level
    seed: int


def _is_colab() -> bool:
    return "google.colab" in sys.modules


def bootstrap(seed: int = 42, verbose: bool = True) -> EnvInfo:
    """Set up torch, seed RNGs, and print a short env summary.

    Call this as the first thing in every notebook (after the clone/install
    setup cell). Returns an EnvInfo with the device you should use:

        info = bootstrap()
        model.to(info.device)
    """
    import numpy as np
    import torch

    in_colab = _is_colab()

    # ------------------------------------------------------------------
    # Seed everything for reproducibility. We pick a single seed and feed
    # it to all random sources. Not a full determinism guarantee (cudnn
    # nondeterminism, DataLoader worker seeding, etc.) but enough for
    # classroom-level reproducibility.
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    info = EnvInfo(
        in_colab=in_colab,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        device_name=device_name,
        device=device,
        seed=seed,
    )

    if verbose:
        env = "Colab" if in_colab else "Local"
        print(f"[env]     {env}")
        print(f"[torch]   {info.torch_version}")
        print(f"[cuda]    available={cuda_available}  device={device_name}")
        print(f"[seed]    {seed}  (python, numpy, torch, cuda)")
        if not cuda_available:
            print("[warn]    No GPU detected. Notebooks 01-03 are CPU-friendly; "
                  "later ones will be slow without CUDA.")

    return info


def data_dir() -> str:
    """Canonical data directory — same path in Colab and locally."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(path, exist_ok=True)
    return path


def checkpoints_dir() -> str:
    """Canonical checkpoints directory — same path in Colab and locally."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(path, exist_ok=True)
    return path


__all__ = [
    "bootstrap",
    "EnvInfo",
    "data_dir",
    "checkpoints_dir",
    "GITHUB_REPO",
    "GITHUB_REPO_DIR",
]
