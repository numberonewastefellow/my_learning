"""
Metric helpers used across the EfficientNetV2 learning notebooks.

We deliberately layer on top of ``torchmetrics`` rather than reimplementing
running averages ourselves:

* torchmetrics handles GPU accumulation correctly (no sneaky host/device
  round-trips per batch).
* Its API (``update`` / ``compute`` / ``reset``) matches exactly how a
  training loop wants to consume a metric.
* Multiclass vs multilabel vs binary is a single constructor kwarg.

The :class:`MetricTracker` below is a tiny facade that bundles the four
classification metrics we care about for teaching -- accuracy, precision,
recall, F1 -- behind a uniform interface. If the notebook wants anything
beyond that, the underlying torchmetrics objects are exposed as attributes.
"""

from __future__ import annotations

from typing import Optional, Sequence


class MetricTracker:
    """Accumulate classification metrics across mini-batches.

    This wraps ``torchmetrics`` ``Accuracy``, ``Precision``, ``Recall`` and
    ``F1Score`` (all configured with the same ``task`` + ``num_classes``
    kwargs) so notebooks can write::

        tracker = MetricTracker(num_classes=10, device=device)
        for logits, y in loader:
            tracker.update(logits, y)
        print(tracker.compute())
        tracker.reset()

    Parameters
    ----------
    num_classes : int
        Number of output classes. Required for multiclass/multilabel tasks.
    task : str, default "multiclass"
        Forwarded to torchmetrics -- one of ``"binary"``, ``"multiclass"``
        or ``"multilabel"``.
    device : str, default "cpu"
        Device on which the internal torchmetrics state lives.
    """

    def __init__(self, num_classes: int, task: str = "multiclass", device: str = "cpu"):
        import torchmetrics

        self.num_classes = num_classes
        self.task = task
        self.device = device

        # Use macro averaging for precision/recall/F1 so a minority class
        # cannot be hidden by the majority; accuracy stays micro (the
        # default), which matches the standard "fraction correct" intuition.
        common = dict(task=task, num_classes=num_classes)
        self.accuracy = torchmetrics.Accuracy(**common).to(device)
        self.precision = torchmetrics.Precision(average="macro", **common).to(device)
        self.recall = torchmetrics.Recall(average="macro", **common).to(device)
        self.f1 = torchmetrics.F1Score(average="macro", **common).to(device)

    def update(self, logits, targets) -> None:
        """Feed one batch of ``(logits, targets)`` into the tracker.

        ``logits`` may be raw network outputs (shape ``(N, C)``) or
        already-argmaxed integer predictions (shape ``(N,)``); torchmetrics
        detects the difference automatically.
        """
        import torch

        if isinstance(logits, torch.Tensor) and logits.ndim == 2:
            preds = logits.argmax(dim=1)
        else:
            preds = logits
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self) -> dict:
        """Return a dict ``{"accuracy", "precision", "recall", "f1"}`` of floats."""
        return {
            "accuracy": float(self.accuracy.compute().item()),
            "precision": float(self.precision.compute().item()),
            "recall": float(self.recall.compute().item()),
            "f1": float(self.f1.compute().item()),
        }

    def reset(self) -> None:
        """Clear internal state -- call this between epochs."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()


def classification_report_dict(
    y_true,
    y_pred,
    class_names: Optional[Sequence[str]] = None,
) -> dict:
    """Return ``sklearn.metrics.classification_report`` as a Python dict.

    This is purely a thin convenience wrapper so the caller doesn't have to
    remember the ``output_dict=True`` flag.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (N,)
        Ground-truth and predicted class indices.
    class_names : sequence of str, optional
        Human-readable labels. If given, keys in the returned dict will be
        the class names; otherwise they are stringified integer indices.
    """
    from sklearn.metrics import classification_report

    return classification_report(
        y_true,
        y_pred,
        target_names=list(class_names) if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )


__all__ = [
    "MetricTracker",
    "classification_report_dict",
]
