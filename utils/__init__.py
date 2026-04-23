"""Reusable helpers for the EfficientNetV2 learning notebooks.

Modules in this package are created progressively as the curriculum introduces
the concepts they encapsulate:

    env.py       -- Batch 0  (this batch)
    plotting.py  -- Notebook 03
    metrics.py   -- Notebook 03
    training.py  -- Notebook 03
    heads.py     -- Notebook 06
    gradcam.py   -- Notebook 10
"""

from .env import bootstrap, EnvInfo, data_dir, checkpoints_dir, GITHUB_REPO, GITHUB_REPO_DIR

__all__ = [
    "bootstrap",
    "EnvInfo",
    "data_dir",
    "checkpoints_dir",
    "GITHUB_REPO",
    "GITHUB_REPO_DIR",
]
