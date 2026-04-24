"""
Plotting helpers used across the EfficientNetV2 learning notebooks.

All functions are thin wrappers over matplotlib (+ seaborn for heatmaps).
They are intentionally small and opinionated so notebooks can stay focused
on the concept being taught rather than on plotting boilerplate.

Design notes for beginners:
- Every function here just calls standard matplotlib/seaborn under the hood.
  You could always drop down to raw matplotlib if you want more control.
- We never call `plt.show()` inside the helpers; Jupyter shows figures
  automatically. This also plays nicely with `save_path`.
- `save_path` is optional everywhere. If provided, we call `plt.savefig`
  with a sensible dpi so plots are shareable.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------
def plot_curves(history: dict, save_path: Optional[str] = None) -> None:
    """Plot side-by-side loss and accuracy curves from a training history.

    Parameters
    ----------
    history : dict
        Must contain the keys ``train_loss``, ``val_loss``, ``train_acc``,
        ``val_acc``. Each value is a list-like of floats whose length equals
        the number of epochs actually run.
    save_path : str, optional
        If given, the figure is written to this path (PNG inferred from
        extension). Parent directory must already exist.

    Notes
    -----
    The function is deliberately lenient: if one of the four series is
    missing it is silently skipped so partially populated histories still
    render.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Loss subplot -----------------------------------------------------
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train", marker="o")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val", marker="o")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Accuracy subplot -------------------------------------------------
    ax = axes[1]
    if "train_acc" in history:
        ax.plot(history["train_acc"], label="train", marker="o")
    if "val_acc" in history:
        ax.plot(history["val_acc"], label="val", marker="o")
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 2. Confusion matrix (seaborn heatmap)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm,
    class_names: Sequence[str],
    normalize: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Render a confusion matrix as an annotated seaborn heatmap.

    Parameters
    ----------
    cm : array-like, shape (n_classes, n_classes)
        Confusion matrix, as returned by ``sklearn.metrics.confusion_matrix``.
    class_names : sequence of str
        Human-readable labels for each row/column (length ``n_classes``).
    normalize : bool, default False
        If True, rows are normalized so each row sums to 1 and annotations
        are displayed as ratios with two decimals. Useful when class sizes
        are imbalanced.
    save_path : str, optional
        If given, saves the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Guard against division by zero for empty rows.
        row_sums[row_sums == 0] = 1.0
        cm_to_plot = cm / row_sums
        fmt = ".2f"
    else:
        cm_to_plot = cm.astype(int)
        fmt = "d"

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n), max(5, 0.5 * n)))
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "Confusion matrix" + (" (row-normalized)" if normalize else "")
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 3. Image grid
# ---------------------------------------------------------------------------
def show_image_grid(
    images,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    figsize=None,
    denormalize: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> None:
    """Display a grid of images stored as tensors or numpy arrays.

    Parameters
    ----------
    images : Tensor of shape (N, C, H, W) or a list/iterable of images
        Each image can be a torch.Tensor (C,H,W) or a HxW / HxWxC numpy array.
    titles : sequence of str, optional
        Per-image titles.
    ncols : int, default 4
        Number of columns in the grid.
    figsize : tuple, optional
        Custom figure size. Defaults to a proportional size.
    denormalize : bool, default False
        If True, ``mean`` and ``std`` must be provided and the inverse
        normalization ``x * std + mean`` is applied before display. Use this
        when your images have been run through ``torchvision.transforms.Normalize``.
    mean, std : sequence of float, optional
        Per-channel normalization statistics.
    """
    import matplotlib.pyplot as plt
    import torch

    # Convert any tensor/list input into a plain Python list of HxWxC arrays
    # in [0,1] suitable for matplotlib.imshow.
    def _to_display_array(img):
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu().float()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                # (C,H,W) -> (H,W,C)
                if denormalize and mean is not None and std is not None:
                    m = torch.tensor(mean).view(-1, 1, 1)
                    s = torch.tensor(std).view(-1, 1, 1)
                    x = x * s + m
                x = x.permute(1, 2, 0).numpy()
            else:
                x = x.numpy()
        else:
            x = np.asarray(img)
        # Squeeze single-channel to 2D so imshow uses a grayscale map.
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        # Clip into the valid display range. Images coming out of denorm
        # can drift slightly outside [0,1] due to numerical error.
        x = np.clip(x, 0.0, 1.0) if x.dtype.kind == "f" else x
        return x

    # Normalize the input into an iterable of images.
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        imgs = [images[i] for i in range(images.shape[0])]
    else:
        imgs = list(images)

    n = len(imgs)
    if n == 0:
        raise ValueError("show_image_grid: got an empty image collection")
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # `axes` comes back as a scalar/1D/2D array depending on shape.
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n:
            arr = _to_display_array(imgs[i])
            cmap = "gray" if arr.ndim == 2 else None
            ax.imshow(arr, cmap=cmap)
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]))
        ax.axis("off")

    fig.tight_layout()


# ---------------------------------------------------------------------------
# 4. ROC / PR curves
# ---------------------------------------------------------------------------
def plot_roc_pr(
    y_true,
    y_score,
    class_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot side-by-side ROC and Precision-Recall curves.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground-truth integer class indices. For binary problems these are
        0/1; for multiclass they are 0..C-1.
    y_score : array-like of shape (N,) or (N, C)
        Predicted scores. For binary problems pass a 1-D array of the
        positive-class probability. For multiclass pass a (N, C) matrix of
        per-class probabilities; we compute one-vs-rest curves.
    class_names : sequence of str, optional
        Names to use in the legend for multiclass.
    save_path : str, optional
        If given, saves the figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc, ax_pr = axes

    if y_score.ndim == 1:
        # Binary case: single curve.
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax_pr.plot(rec, prec, label=f"AP = {ap:.3f}")
    else:
        # Multiclass: one-vs-rest, one curve per class.
        n_classes = y_score.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_score[:, c])
            ax_roc.plot(fpr, tpr, label=f"{class_names[c]} (AUC {auc(fpr, tpr):.2f})")
            prec, rec, _ = precision_recall_curve(y_bin, y_score[:, c])
            ap = average_precision_score(y_bin, y_score[:, c])
            ax_pr.plot(rec, prec, label=f"{class_names[c]} (AP {ap:.2f})")

    # Chance line for ROC.
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curve")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right", fontsize=8)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 5. Gradient norms per layer
# ---------------------------------------------------------------------------
def plot_grad_norms(grad_norms_by_layer: dict) -> None:
    """Horizontal bar chart of per-layer gradient L2 norms.

    Parameters
    ----------
    grad_norms_by_layer : dict[str, float]
        Mapping from layer (parameter) name to its gradient L2 norm.
        Typically built with something like::

            {name: p.grad.detach().norm().item()
             for name, p in model.named_parameters() if p.grad is not None}

    Why this is useful
    ------------------
    The plot makes vanishing/exploding-gradient issues immediately obvious:
    a long flat sequence of near-zero bars at the top of the network means
    the early layers are barely learning.
    """
    import matplotlib.pyplot as plt

    if not grad_norms_by_layer:
        raise ValueError("plot_grad_norms: received an empty dict")

    names = list(grad_norms_by_layer.keys())
    values = [float(grad_norms_by_layer[k]) for k in names]

    # Layers are typically registered input-to-output; flipping makes the
    # input layer appear at the top, which is easier to scan.
    names = names[::-1]
    values = values[::-1]

    height = max(3, 0.3 * len(names))
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Gradient L2 norm")
    ax.set_title("Per-layer gradient norms")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()


__all__ = [
    "plot_curves",
    "plot_confusion_matrix",
    "show_image_grid",
    "plot_roc_pr",
    "plot_grad_norms",
]
