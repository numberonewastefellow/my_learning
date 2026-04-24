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
from .plotting import (
    plot_curves,
    plot_confusion_matrix,
    show_image_grid,
    plot_roc_pr,
    plot_grad_norms,
)
from .metrics import MetricTracker, classification_report_dict
from .training import train_one_epoch, evaluate, fit
from .heads import (
    CustomHead,
    DualHead,
    MultiHead,
    ChannelAttention,
    SpatialAttention,
    SelfAttentionHead,
)
from .gradcam import (
    GradCAM,
    overlay_cam,
    find_efficientnet_target_layer,
)

__all__ = [
    # env
    "bootstrap",
    "EnvInfo",
    "data_dir",
    "checkpoints_dir",
    "GITHUB_REPO",
    "GITHUB_REPO_DIR",
    # plotting
    "plot_curves",
    "plot_confusion_matrix",
    "show_image_grid",
    "plot_roc_pr",
    "plot_grad_norms",
    # metrics
    "MetricTracker",
    "classification_report_dict",
    # training
    "train_one_epoch",
    "evaluate",
    "fit",
    # heads
    "CustomHead",
    "DualHead",
    "MultiHead",
    "ChannelAttention",
    "SpatialAttention",
    "SelfAttentionHead",
    # gradcam
    "GradCAM",
    "overlay_cam",
    "find_efficientnet_target_layer",
]
