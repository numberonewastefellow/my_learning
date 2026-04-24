"""
Grad-CAM utilities for visualizing *where* a CNN is looking.

Grad-CAM (Selvaraju et al., 2017 -- https://arxiv.org/abs/1610.02391) is the
standard "class-activation map" technique for CNNs. For a chosen class ``c``:

    1. Run a forward pass and grab activations ``A^k`` of a late conv layer
       (one feature map per channel ``k``).
    2. Run a backward pass on the class-score ``y^c`` and grab the gradient
       ``dy^c / dA^k`` at the same layer.
    3. Spatial-average those gradients to get a per-channel importance
       weight ``alpha^c_k``.
    4. Combine: ``L^c = ReLU( sum_k alpha^c_k * A^k )`` -- a small 2D map
       highlighting pixels that push the score for class ``c`` up.
    5. Upsample to the input resolution and min-max normalize to ``[0, 1]``
       so it can be rendered as a heatmap.

This file exposes:

    GradCAM(model, target_layer)         -- the hook-based CAM generator.
    overlay_cam(image_tensor, cam, ...)  -- composites CAM over the image.
    find_efficientnet_target_layer(m)    -- pick a sensible default layer.

Everything is written to be readable first and fast second. Notebook 10 walks
through the math and uses these three helpers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Default ImageNet normalization stats -- the same ones timm / torchvision
# models are trained with. ``overlay_cam`` needs these to *undo* the
# normalization before it blends the heatmap on top.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GradCAM:
    """Class-activation map via gradient-weighted feature maps.

    Reference
    ---------
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.

    Parameters
    ----------
    model : nn.Module
        A CNN in eval mode (we do not toggle it for you). The model must have
        ``requires_grad`` left at its defaults -- we only need gradients to
        flow into ``target_layer``, so freezing the backbone is fine.
    target_layer : nn.Module
        The layer whose activations + gradients are captured. For
        EfficientNetV2 a good default is the last conv before the classifier
        pool (``model.conv_head`` in timm). Use
        :func:`find_efficientnet_target_layer` if unsure.

    Notes
    -----
    The class registers two hooks. Call :meth:`remove_hooks` when done or
    reuse the same instance for many images (much faster than recreating).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        # These are set from inside the hooks.
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # ``register_forward_hook`` fires on the forward pass and receives
        # ``(module, input, output)``. We keep ``output`` as the activation
        # map A^k of shape (B, C, H, W).
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)

        # ``register_full_backward_hook`` fires during ``.backward()`` and
        # receives ``(module, grad_input, grad_output)``. ``grad_output[0]``
        # is ``dy^c / dA^k`` at this layer -- exactly what we need.
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks -- keep these as simple getters; all math lives in
    # __call__ so that failure modes are easy to read in a stack trace.
    # ------------------------------------------------------------------
    def _save_activation(self, module, inputs, output):
        # Detach? No -- we need the graph so backward() can flow through.
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; index [0] is the tensor we want.
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape ``(1, 3, H, W)`` or ``(3, H, W)``, already normalized with
            the same stats the model was trained on, on the same device as
            the model.
        target_class : int, optional
            Class index to explain. When ``None`` we use ``argmax`` of the
            model's logits -- i.e. "explain the prediction the model made".

        Returns
        -------
        np.ndarray
            Shape ``(H, W)``, dtype float32, values in ``[0, 1]``. The map
            has been upsampled to the input resolution and per-image
            min-max normalized so it is ready to plot.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        assert input_tensor.dim() == 4 and input_tensor.size(0) == 1, (
            f"GradCAM expects a single image (1,3,H,W); got {tuple(input_tensor.shape)}"
        )

        # We need gradients but we do NOT want them to flow into the model
        # parameters -- enabling grad on the input is enough.
        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.requires_grad_(True)

        # 1) Forward pass: fills ``self.activations`` through the hook.
        logits = self.model(input_tensor)                       # (1, num_classes)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # 2) Backward pass on y^c (the target class score).
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        activations = self.activations                           # (1, C, H', W')
        gradients = self.gradients                               # (1, C, H', W')
        assert activations is not None and gradients is not None, (
            "Hooks did not fire -- is target_layer actually on the forward path?"
        )

        # 3) alpha^c_k = mean over spatial dims of the gradient.
        #    Shape: (1, C, H', W') -> (1, C, 1, 1) so broadcasting works.
        weights = gradients.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)

        # 4) Combine: weighted sum of activation maps, then ReLU (we only
        #    care about pixels that *raise* the class score).
        cam = (weights * activations).sum(dim=1, keepdim=True)    # (1, 1, H', W')
        cam = F.relu(cam)

        # 5) Upsample to input resolution. bilinear is the standard choice.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)                           # (H, W)

        # 6) Min-max normalize to [0, 1] so it is plottable.
        cam = cam.detach().cpu().float()
        cam_min = cam.min()
        cam_max = cam.max()
        if float(cam_max - cam_min) < 1e-8:
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.numpy().astype(np.float32)

    def remove_hooks(self):
        """Detach the forward/backward hooks. Safe to call more than once."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def __del__(self):  # best-effort cleanup
        try:
            self.remove_hooks()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------
def overlay_cam(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    alpha: float = 0.5,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Image.Image:
    """Blend a jet-colored CAM on top of an image.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape ``(3, H, W)`` or ``(1, 3, H, W)``, normalized with
        ``(mean, std)``. We *undo* that normalization so the heatmap is
        drawn over the original-looking image.
    cam : np.ndarray
        Shape ``(H, W)``, values in ``[0, 1]``. Typically produced by
        :class:`GradCAM.__call__`.
    alpha : float, default 0.5
        Blending weight. ``0.0`` = pure image, ``1.0`` = pure heatmap.
    mean, std : sequence of 3 floats, optional
        Normalization stats used when preparing ``image_tensor``. Default
        is ImageNet.

    Returns
    -------
    PIL.Image.Image
        An RGB image, same (H, W) as the input, dtype uint8.
    """
    import matplotlib.cm as cm  # local import keeps utils import-light

    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    assert image_tensor.dim() == 3 and image_tensor.size(0) == 3, (
        f"overlay_cam expects (3,H,W) or (1,3,H,W); got {tuple(image_tensor.shape)}"
    )

    # Un-normalize: x * std + mean, clamp to [0, 1].
    mean_t = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    img = (image_tensor.detach().cpu() * std_t.cpu() + mean_t.cpu()).clamp(0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()                          # (H, W, 3) in [0, 1]

    # Colorize the CAM with jet and drop the alpha channel matplotlib adds.
    heatmap = cm.jet(np.clip(cam, 0.0, 1.0))[..., :3]              # (H, W, 3) in [0, 1]

    # If CAM and image have different sizes, resize the heatmap to match.
    if heatmap.shape[:2] != img_np.shape[:2]:
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize(
            (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
        )
        heatmap = np.asarray(heatmap_pil, dtype=np.float32) / 255.0

    blended = (1.0 - alpha) * img_np + alpha * heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


# ---------------------------------------------------------------------------
# Target-layer heuristics
# ---------------------------------------------------------------------------
def find_efficientnet_target_layer(model: nn.Module) -> nn.Module:
    """Pick a sensible "last conv" layer for Grad-CAM on EfficientNet(V2).

    Strategy (first match wins):

    1. timm EfficientNetV2: ``model.conv_head`` is the 1x1 conv right before
       the classifier's global pool. It's the conventional choice.
    2. torchvision EfficientNet: ``model.features[-1]`` is the terminal
       Conv-BN-Act block, semantically equivalent.
    3. Fallback: iterate over ``model.modules()`` and return the last
       ``nn.Conv2d`` encountered.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    nn.Module
        A layer you can pass to ``GradCAM(model, target_layer)``.

    Raises
    ------
    ValueError
        If no ``nn.Conv2d`` is found anywhere in the model.
    """
    # 1) timm EfficientNet / EfficientNetV2
    conv_head = getattr(model, "conv_head", None)
    if isinstance(conv_head, nn.Module):
        return conv_head

    # 2) torchvision EfficientNet
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential) and len(features) > 0:
        return features[-1]

    # 3) Fallback -- last Conv2d anywhere in the graph.
    last_conv: Optional[nn.Module] = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError(
            "Could not locate a target layer: model has no `conv_head`, "
            "no `features` Sequential, and no nn.Conv2d layers."
        )
    return last_conv


__all__ = [
    "GradCAM",
    "overlay_cam",
    "find_efficientnet_target_layer",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
