"""
Training-loop helpers used across the EfficientNetV2 learning notebooks.

These functions capture the standard PyTorch "train for an epoch, evaluate,
checkpoint, maybe early-stop" dance so each notebook can focus on what is
*new* to that lesson rather than re-deriving the loop every time.

The public surface is intentionally small:

* :func:`train_one_epoch` -- one pass over a loader with an optimizer.
* :func:`evaluate`        -- one pass with ``torch.no_grad()``.
* :func:`fit`             -- orchestrates ``train_one_epoch`` + ``evaluate``
  for multiple epochs, plus checkpointing and early stopping.

All three work for plain multi-class classification with ``CrossEntropyLoss``
(or anything equivalent). The notebooks never need bigger machinery.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper: detect OneCycle-like schedulers (step per-batch, not per-epoch).
# ---------------------------------------------------------------------------
def _is_per_batch_scheduler(scheduler) -> bool:
    """True for LR schedulers that expect ``.step()`` every batch."""
    if scheduler is None:
        return False
    try:
        from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
        if isinstance(scheduler, (OneCycleLR, CyclicLR)):
            return True
    except Exception:
        pass
    # Fall-back heuristic: class name contains "OneCycle" or "Cyclic".
    name = type(scheduler).__name__.lower()
    return "onecycle" in name or "cyclic" in name


# ---------------------------------------------------------------------------
# 1. Single epoch trainer
# ---------------------------------------------------------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    scheduler=None,
) -> Tuple[float, float]:
    """Run one training epoch and return ``(avg_loss, avg_acc)``.

    This is the innermost loop of a supervised classification pipeline:
    for each batch we run a forward pass, compute the loss, back-propagate,
    step the optimizer, and track running averages.

    Parameters
    ----------
    model : nn.Module
    loader : torch.utils.data.DataLoader
    optimizer : torch.optim.Optimizer
    loss_fn : callable
        Any loss that returns a scalar tensor, e.g. ``nn.CrossEntropyLoss()``.
    device : torch.device or str
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        If given *and* it's a per-batch scheduler (``OneCycleLR``,
        ``CyclicLR``), ``scheduler.step()`` is called after every
        optimizer step. Otherwise the caller is responsible for
        epoch-level scheduler stepping.

    Returns
    -------
    avg_loss : float
        Mean cross-entropy loss across all samples seen.
    avg_acc : float
        Fraction of samples classified correctly.
    """
    import torch

    model.train()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_batch_scheduler = _is_per_batch_scheduler(scheduler)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # --- forward ---
        logits = model(xb)
        loss = loss_fn(logits, yb)

        # --- backward ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if per_batch_scheduler:
            scheduler.step()

        # --- running stats (detach to free graph memory) ---
        batch_size = yb.size(0)
        total_loss += loss.detach().item() * batch_size
        preds = logits.detach().argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model,
    loader,
    loss_fn,
    device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate a model on ``loader`` and return predictions.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    loss_fn : callable
    device : torch.device or str

    Returns
    -------
    avg_loss : float
    avg_acc  : float
    y_true   : 1-D numpy int array of ground-truth labels
    y_pred   : 1-D numpy int array of predicted labels

    Notes
    -----
    Uses ``model.eval()`` + ``torch.no_grad()``. This both disables dropout /
    batchnorm updates and avoids building an autograd graph -- essential
    for fast, memory-light evaluation.
    """
    import torch

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true_chunks = []
    y_pred_chunks = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = loss_fn(logits, yb)
            preds = logits.argmax(dim=1)

            batch_size = yb.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == yb).sum().item()
            total_samples += batch_size

            y_true_chunks.append(yb.cpu().numpy())
            y_pred_chunks.append(preds.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=int)
    return avg_loss, avg_acc, y_true.astype(int), y_pred.astype(int)


# ---------------------------------------------------------------------------
# 3. Full fit loop
# ---------------------------------------------------------------------------
def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs: int,
    device,
    scheduler=None,
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Train a model for up to ``epochs`` epochs with optional checkpointing.

    This is the "nice ergonomics" loop we reach for once the mechanics of
    ``train_one_epoch`` / ``evaluate`` are clear. It:

    * calls ``train_one_epoch`` + ``evaluate`` every epoch;
    * tracks loss and accuracy history for both splits;
    * steps epoch-level schedulers (``StepLR`` / ``CosineAnnealingLR`` / ...);
    * saves a checkpoint whenever val accuracy improves;
    * optionally early-stops after ``early_stopping_patience`` stagnant
      epochs;
    * shows a tqdm progress bar across epochs.

    Parameters
    ----------
    model, train_loader, val_loader, loss_fn, optimizer, device
        Standard pieces of a PyTorch pipeline.
    epochs : int
        Maximum number of epochs.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Per-batch schedulers (``OneCycleLR``, ``CyclicLR``) are stepped
        inside ``train_one_epoch``; anything else is stepped once per epoch
        here.
    early_stopping_patience : int, optional
        If set, training stops once val accuracy hasn't improved for this
        many consecutive epochs.
    checkpoint_path : str, optional
        File path (not directory) to ``torch.save`` the best model weights
        to. Parent directory is created if missing.
    verbose : bool, default True
        If True, print a one-line summary per epoch.

    Returns
    -------
    history : dict
        Dict with keys ``train_loss``, ``val_loss``, ``train_acc``,
        ``val_acc`` (lists of length ``epochs_run``), plus scalars
        ``best_val_acc`` and ``best_epoch`` (0-indexed).
    """
    import torch
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is in requirements.txt
        tqdm = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = -float("inf")
    best_epoch = -1
    stagnant = 0
    per_batch_scheduler = _is_per_batch_scheduler(scheduler)

    # Make sure the checkpoint directory exists before the first save.
    if checkpoint_path is not None:
        parent = os.path.dirname(os.path.abspath(checkpoint_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    iterator = range(epochs)
    if verbose and tqdm is not None:
        iterator = tqdm(iterator, desc="epochs", total=epochs)

    for epoch in iterator:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scheduler=scheduler
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Step epoch-level schedulers *after* validation. Per-batch ones
        # were already stepped inside train_one_epoch.
        if scheduler is not None and not per_batch_scheduler:
            try:
                scheduler.step()
            except TypeError:
                # ReduceLROnPlateau requires the metric as argument.
                scheduler.step(val_loss)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            stagnant = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                    },
                    checkpoint_path,
                )
        else:
            stagnant += 1

        if verbose:
            flag = "  *" if improved else ""
            msg = (
                f"epoch {epoch + 1:3d}/{epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}{flag}"
            )
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)

        if (
            early_stopping_patience is not None
            and stagnant >= early_stopping_patience
        ):
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no val-acc improvement for {stagnant} epochs."
                )
            break

    history["best_val_acc"] = float(best_val_acc) if best_epoch >= 0 else 0.0
    history["best_epoch"] = int(best_epoch)
    return history


__all__ = [
    "train_one_epoch",
    "evaluate",
    "fit",
]
