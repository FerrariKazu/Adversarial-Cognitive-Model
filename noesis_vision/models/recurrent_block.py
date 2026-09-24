"""
Recurrent (tied-weight) transformer block — Agent C.
================================================================================

Part 1.C (LOCKED, Hybrid Option C): the within-glimpse refinement is a
SHARED-WEIGHT transformer block run 2-3 times per glimpse before pooling —
Universal-Transformer-style. Tied weights mean the iteration count changes
COMPUTE, NOT parameter count; the compactness accounting therefore stays
exact regardless of the configured iteration count.

"Same parameter tensors reused, not copies" is enforced structurally:
TiedRecurrence holds ONE TransformerBlock instance and every iteration
calls the same module, so a forward+backward accumulates every iteration's
gradient onto that single parameter set (asserted by
tests/test_recurrent_vision_core.py::test_recurrent_tied_weights — if the
iterations ever held copies, each copy would receive its own separate
grad).

Iteration range 2-3 is LOCKED (Part 1.C; enforced identically by
noesis_vision.core.schema's within_glimpse_iters validation — keep the two
in sync; schema raises on the config side, this module raises on the model
side).

Submodule names (norm1 / attn.qkv / attn.proj / norm2 / mlp.fc1 / mlp.fc2)
mirror DINOv2's block layout so the trunk's warm-start key mapping is 1:1
(see backbone.load_dino_warm_start). The tied refinement block is NEW
machinery — it intentionally has NO DINOv2 counterpart and is reported as
new-by-design by the warm-start loader, not silently treated as matched.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: LOCKED range (Part 1.C). Schema enforces it for the config; we enforce
#: it for the model — a caller bypassing the schema still cannot drift.
WITHIN_GLIMPSE_ITERS_RANGE = (2, 3)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (DINOv2 naming), shape (B, N, D) -> same."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.ModuleDict({
            "qkv": nn.Linear(d_model, 3 * d_model, bias=True),
            "proj": nn.Linear(d_model, d_model, bias=True),
        })
        self.attn_drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.ModuleDict({
            "fc1": nn.Linear(d_model, hidden, bias=True),
            "fc2": nn.Linear(hidden, d_model, bias=True),
        })
        self.mlp_drop = nn.Dropout(drop)
        self.num_heads = num_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D). Self-attention + MLP, pre-LN residuals."""
        B, N, D = x.shape
        qkv = self.attn["qkv"](self.norm1(x))                     # (B, N, 3D)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, D // self.num_heads) \
            .permute(2, 0, 3, 1, 4).unbind(0)                     # 3 x (B, h, N, hd)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                dropout_p=self.attn_drop.p
                                                                if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        x = x + self.attn["proj"](attn)

        h = self.mlp["fc2"](self.mlp_drop(
            torch.nn.functional.gelu(self.mlp["fc1"](self.norm2(x)))))
        return x + self.mlp_drop(h)


class TiedRecurrence(nn.Module):
    """Applies ONE block K times, reusing the same parameter tensors.

    Not a ModuleList of copies: a single block instance called K times, so
    parameter identity across iterations is true by construction and the
    iteration count lives purely in the forward loop (compute, not params).
    """

    def __init__(self, block: TransformerBlock):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, num_iters: int) -> torch.Tensor:
        lo, hi = WITHIN_GLIMPSE_ITERS_RANGE
        if not (lo <= num_iters <= hi):
            raise ValueError(
                f"within-glimpse iterations must be in the LOCKED range "
                f"{lo}-{hi} (Part 1.C); got {num_iters} — changing the range "
                f"is a plan ruling, not a call-site choice")
        for _ in range(num_iters):
            x = self.block(x)
        return x

    def tied_parameters(self):
        """The single parameter set every iteration shares."""
        return list(self.block.parameters())
