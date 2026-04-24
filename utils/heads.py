"""
Reusable head modules for the EfficientNetV2 learning notebooks.

A "head" is the small sub-network that sits on top of a pretrained backbone
and turns the backbone's pooled feature vector (or spatial feature map) into
the final predictions required by a specific task. Swapping the head is the
most common first step when adapting a pretrained model to a new problem.

This module collects the heads that Notebook 06 introduces:

    CustomHead          Richer MLP classification head.
    DualHead            Two simultaneous classification outputs (breed + species).
    MultiHead           Generic multi-task head (dict-in / dict-out).
    ChannelAttention    Squeeze-and-Excitation style channel recalibration.
    SpatialAttention    CBAM-style spatial attention mask.
    SelfAttentionHead   Token-based self-attention over a feature map.

Every class has an explicit, beginner-friendly docstring with shape comments
so you can read it and understand exactly what happens at each step.
"""

from __future__ import annotations

from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. CustomHead -- a richer MLP classification head
# ---------------------------------------------------------------------------
class CustomHead(nn.Module):
    """Richer classification head: ``Dropout -> Linear -> BN -> ReLU -> Linear``.

    The default head produced by ``timm.create_model(..., num_classes=N)`` is
    a single ``nn.Linear(in_features, N)``. For many tasks this is all you
    need, but a slightly deeper MLP head with Dropout and BatchNorm can help
    when:

    * the task is fine-grained (many visually similar classes),
    * the pooled feature vector is high-dimensional and you want a bit of
      regularisation before the final logits,
    * you are doing linear-probe style experiments and want to compare a
      single-linear head with a deeper one.

    Architecture
    ------------
    ::

        x  ---(Dropout)---(Linear: in -> hidden)---(BN)---(ReLU)---(Linear: hidden -> num_classes)---> logits

    Parameters
    ----------
    in_features : int
        Dimensionality of the pooled feature vector produced by the backbone.
    num_classes : int
        Number of output classes.
    hidden : int, default 512
        Width of the intermediate hidden layer.
    dropout : float, default 0.3
        Dropout probability applied to the input features.

    Shapes
    ------
    Input : ``(B, in_features)``
    Output: ``(B, num_classes)``
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden = hidden

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc1(x)
        # BatchNorm1d expects (B, C). If a spurious extra dim slips in we
        # flatten it here -- this makes the module robust to upstream shape
        # surprises when users plug it into a Sequential backbone.
        if x.ndim > 2:
            x = x.flatten(1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 2. DualHead -- two classification outputs sharing a backbone
# ---------------------------------------------------------------------------
class DualHead(nn.Module):
    """Two classification outputs that share a backbone.

    The head is literally just two parallel ``nn.Linear`` layers that both
    consume the same pooled feature vector. The forward method returns a
    ``(primary_logits, aux_logits)`` tuple so the caller can mix two losses:

    ::

        primary, aux = head(features)
        loss = ce(primary, y_primary) + lam * ce(aux, y_aux)

    A common pattern on Oxford-IIIT Pet is:

    * ``num_primary = 37``  (37 breeds, multiclass)
    * ``num_aux     =  2``  (cat vs dog, multiclass)

    The auxiliary task acts as a regulariser on the shared backbone.

    Parameters
    ----------
    in_features : int
        Dimensionality of the pooled feature vector.
    num_primary : int
        Number of classes for the primary task.
    num_aux : int
        Number of classes for the auxiliary task.
    """

    def __init__(self, in_features: int, num_primary: int, num_aux: int):
        super().__init__()
        self.primary = nn.Linear(in_features, num_primary)
        self.aux = nn.Linear(in_features, num_aux)

    def forward(self, x: torch.Tensor):
        return self.primary(x), self.aux(x)


# ---------------------------------------------------------------------------
# 3. MultiHead -- generic multi-task head
# ---------------------------------------------------------------------------
class MultiHead(nn.Module):
    """Generic multi-head container for multi-task learning.

    ``head_specs`` is a list of dicts, one per task, e.g.::

        [
            {"name": "species", "out": 37, "type": "multiclass"},
            {"name": "colors",  "out":  8, "type": "multilabel"},
            {"name": "coords",  "out":  2, "type": "regression"},
        ]

    Each dict must provide ``name`` (a unique string) and ``out`` (the output
    dimensionality). ``type`` is stored for downstream bookkeeping (to pick
    the right loss) but the module itself is a plain ``nn.Linear`` per head
    in all three cases: the distinction only matters for how you interpret
    the outputs and which loss function you pair them with.

    ``forward`` returns a ``dict`` mapping each head's ``name`` to its output
    tensor, which makes loss computation in training loops very tidy::

        outputs = head(features)
        loss = (
            F.cross_entropy(outputs["species"], y_species)
            + F.binary_cross_entropy_with_logits(outputs["colors"], y_colors.float())
            + F.mse_loss(outputs["coords"], y_coords.float())
        )

    Parameters
    ----------
    in_features : int
        Dimensionality of the pooled feature vector from the backbone.
    head_specs : list of dict
        One entry per task as described above.
    """

    _VALID_TYPES = {"multiclass", "multilabel", "regression"}

    def __init__(self, in_features: int, head_specs: List[dict]):
        super().__init__()
        if not head_specs:
            raise ValueError("MultiHead: head_specs cannot be empty")

        self.in_features = in_features
        self.specs = head_specs

        heads: Dict[str, nn.Module] = {}
        for spec in head_specs:
            name = spec["name"]
            out = int(spec["out"])
            ttype = spec.get("type", "multiclass")
            if ttype not in self._VALID_TYPES:
                raise ValueError(
                    f"MultiHead: unknown head type {ttype!r} (valid: {self._VALID_TYPES})"
                )
            if name in heads:
                raise ValueError(f"MultiHead: duplicate head name {name!r}")
            heads[name] = nn.Linear(in_features, out)

        # ``nn.ModuleDict`` registers each sub-module so parameters are
        # visible to optimizers and ``.to(device)`` calls.
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# 4. ChannelAttention -- Squeeze-and-Excitation style
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) style channel attention.

    Given a feature map ``x`` of shape ``(B, C, H, W)`` the module learns a
    per-channel gate vector ``s`` of shape ``(B, C)`` in the range ``(0, 1)``
    and multiplies the input channels by it. Channels that are judged useful
    for the current input are amplified; the rest are suppressed.

    Pipeline
    --------
    ::

        x  ---GAP---> (B, C) ---Linear(C -> C/r)---ReLU---Linear(C/r -> C)---Sigmoid---> s

        out = x * s.view(B, C, 1, 1)

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    reduction : int, default 16
        Bottleneck ratio. The hidden size is ``max(channels // reduction, 1)``.

    Shapes
    ------
    Input : ``(B, C, H, W)``
    Output: ``(B, C, H, W)`` with channels rescaled.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channels = channels
        self.reduction = reduction
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pool to (B, C, 1, 1) then flatten.
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation: bottleneck MLP followed by a sigmoid gate.
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        # Scale: broadcast the gate over the spatial dimensions.
        return x * s.view(b, c, 1, 1)


# ---------------------------------------------------------------------------
# 5. SpatialAttention -- CBAM-style
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention.

    The module summarises ``x`` along the channel axis using both mean and
    max pooling, concatenates the two summaries into a 2-channel map, and
    runs a single ``7x7`` convolution followed by a sigmoid to produce a
    single-channel spatial attention mask in ``(0, 1)``. The mask is then
    broadcast across the channel axis to gate ``x`` spatially.

    Pipeline
    --------
    ::

        avg = x.mean(dim=1, keepdim=True)  -> (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)  -> (B, 1, H, W)
        cat = concat([avg, mx], dim=1)     -> (B, 2, H, W)
        mask = sigmoid( Conv2d(2, 1, 7, padding=3)(cat) )  -> (B, 1, H, W)
        out  = x * mask

    Shapes
    ------
    Input : ``(B, C, H, W)``
    Output: ``(B, C, H, W)`` with spatial locations rescaled.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)  # (B, 2, H, W)
        mask = torch.sigmoid(self.conv(cat))  # (B, 1, H, W)
        return x * mask


# ---------------------------------------------------------------------------
# 6. SelfAttentionHead -- tokenize spatial positions and attend
# ---------------------------------------------------------------------------
class SelfAttentionHead(nn.Module):
    """Token-based self-attention over a feature map.

    Treats the ``H * W`` spatial locations of a feature map as a sequence of
    tokens (each of dimension ``C``), adds a learned positional embedding,
    applies one round of multi-head self-attention with a residual connection
    and LayerNorm, mean-pools the tokens into a single vector, and (optionally)
    projects it to ``num_classes`` logits.

    The result is a "vision-transformer-lite" head you can attach on top of
    any CNN backbone that produces a 4-D feature map.

    Pipeline
    --------
    ::

        x: (B, C, H, W) -- spatial features from the backbone
        tokens: (B, H*W, C)
        tokens = tokens + pos_embed[:, :H*W]
        attn_out, _ = MultiheadAttention(tokens, tokens, tokens)
        tokens = LayerNorm(tokens + attn_out)
        pooled = tokens.mean(dim=1)                 # (B, C)
        logits = fc(pooled) if num_classes else pooled

    Parameters
    ----------
    dim : int
        Channel dimension ``C`` of the incoming feature map. Doubles as the
        token embedding dimension for the attention layer.
    num_heads : int, default 4
        Number of attention heads. ``dim`` must be divisible by ``num_heads``.
    num_classes : int or None, default None
        If ``None``, forward returns the pooled feature vector ``(B, dim)``.
        Otherwise a final ``nn.Linear(dim, num_classes)`` is applied.

    Notes
    -----
    * A generous-sized positional embedding (``max_tokens=1024``) is allocated
      once and sliced to the required length each forward pass. This supports
      feature maps up to ``32 x 32`` without reallocation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_classes: Optional[int] = None,
        max_tokens: int = 1024,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"SelfAttentionHead: dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.num_classes = num_classes

        # Learned absolute positional embedding. Kept slightly oversized so
        # the same module can handle feature maps of different spatial sizes.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

        if num_classes is not None:
            self.fc = nn.Linear(dim, num_classes)
        else:
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"SelfAttentionHead expects (B, C, H, W); got shape {tuple(x.shape)}"
            )
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(
                f"SelfAttentionHead: channel dim {c} != configured dim {self.dim}"
            )
        n_tokens = h * w
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"SelfAttentionHead: H*W={n_tokens} exceeds max_tokens={self.max_tokens}"
            )

        # (B, C, H, W) -> (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :n_tokens]

        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attn_out)

        pooled = tokens.mean(dim=1)  # (B, C)
        if self.fc is not None:
            return self.fc(pooled)
        return pooled


__all__ = [
    "CustomHead",
    "DualHead",
    "MultiHead",
    "ChannelAttention",
    "SpatialAttention",
    "SelfAttentionHead",
]
