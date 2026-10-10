"""Gen-2 Substrate Backbone — CompactViTGen2 with LayerScale and Warm-Start.
================================================================================

Adds LayerScale support (ls1/ls2 gammas matching official DINOv2 naming)
behind a flag (use_layer_scale=True by default for Gen-2), bicubic antialiased
pos_embed interpolation for the 4x4 patch grid (56x56 fovea/gist view), and
strict key matching verification for warm start.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from noesis_vision.models.recurrent_block import (
    WITHIN_GLIMPSE_ITERS_RANGE,
    TiedRecurrence,
)
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE, foveal_sample

D_Z = 384
PATCH_SIZE = 14
TRUNK_DEPTH = 12
TRUNK_WIDTH = 384
TRUNK_HEADS = 6
REFINE_MLP_RATIO = 4.0
WARM_START_MATCH_FRACTION_MIN = 0.95


class LayerScale(nn.Module):
    """LayerScale module matching DINOv2's `ls1.gamma` / `ls2.gamma` convention."""

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class TransformerBlockGen2(nn.Module):
    """Pre-LN transformer block matching DINOv2-small layout with LayerScale."""

    def __init__(
        self,
        d_model: int = TRUNK_WIDTH,
        num_heads: int = TRUNK_HEADS,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        use_layer_scale: bool = True,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.use_layer_scale = use_layer_scale
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.ModuleDict({
            "qkv": nn.Linear(d_model, 3 * d_model, bias=True),
            "proj": nn.Linear(d_model, d_model, bias=True),
        })
        self.attn_drop = nn.Dropout(drop)
        if use_layer_scale:
            self.ls1 = LayerScale(d_model, init_values=init_values)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.ModuleDict({
            "fc1": nn.Linear(d_model, hidden, bias=True),
            "fc2": nn.Linear(hidden, d_model, bias=True),
        })
        self.mlp_drop = nn.Dropout(drop)
        if use_layer_scale:
            self.ls2 = LayerScale(d_model, init_values=init_values)

        self.num_heads = num_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # Attention branch
        qkv = self.attn["qkv"](self.norm1(x))
        q, k, v = (
            qkv.reshape(B, N, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(B, N, D)
        attn = self.attn["proj"](attn)
        if self.use_layer_scale:
            attn = self.ls1(attn)
        x = x + attn

        # MLP branch
        h = self.mlp["fc2"](
            self.mlp_drop(F.gelu(self.mlp["fc1"](self.norm2(x))))
        )
        h = self.mlp_drop(h)
        if self.use_layer_scale:
            h = self.ls2(h)
        return x + h


class CompactViTGen2(nn.Module):
    """DINOv2-small-shaped trunk with LayerScale + tied within-glimpse refinement."""

    def __init__(
        self,
        img_size: int = DEFAULT_FOVEA_SIZE,
        num_refine_iters: int = 2,
        refine_mlp_ratio: float = REFINE_MLP_RATIO,
        use_layer_scale: bool = True,
    ) -> None:
        super().__init__()
        lo, hi = WITHIN_GLIMPSE_ITERS_RANGE
        if not (lo <= num_refine_iters <= hi):
            raise ValueError(f"num_refine_iters must be in range {lo}-{hi}")
        if img_size % PATCH_SIZE != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch size {PATCH_SIZE}")

        self.img_size = img_size
        self.num_refine_iters = num_refine_iters
        self.use_layer_scale = use_layer_scale
        self.d_z = D_Z

        self.patch_embed = nn.ModuleDict({
            "proj": nn.Conv2d(3, TRUNK_WIDTH, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=True),
        })
        num_patches = (img_size // PATCH_SIZE) ** 2
        self.num_prefix_tokens = 2  # cls + register (DINOv2 layout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, TRUNK_WIDTH))
        self.register_token = nn.Parameter(torch.zeros(1, 1, TRUNK_WIDTH))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, TRUNK_WIDTH)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_token, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlockGen2(TRUNK_WIDTH, TRUNK_HEADS, use_layer_scale=use_layer_scale)
            for _ in range(TRUNK_DEPTH)
        )
        self.norm = nn.LayerNorm(TRUNK_WIDTH)

        self.refinement = TiedRecurrence(
            TransformerBlockGen2(
                TRUNK_WIDTH,
                TRUNK_HEADS,
                mlp_ratio=refine_mlp_ratio,
                use_layer_scale=use_layer_scale,
            )
        )

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed["proj"](x)
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.register_token.expand(B, -1, -1)
        x = torch.cat([cls, reg, tokens], dim=1)
        return x + self.pos_embed

    def _trunk_forward(self, x_prep: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x_prep = blk(x_prep)
        return self.norm(x_prep)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._prep(x)
        x = self._trunk_forward(x)
        x = self.refinement(x, self.num_refine_iters)
        pooled_z = x[:, 0]
        tokens = x[:, self.num_prefix_tokens:]
        return pooled_z, tokens

    def load_dino_warm_start(
        self,
        path: str,
        strict_band: float = WARM_START_MATCH_FRACTION_MIN,
        interpolate_pos_embed: bool = True,
    ) -> Dict[str, Any]:
        """Load official DINOv2-small weights with LayerScale and pos_embed interpolation.

        Trunk keys (blocks, patch_embed, norm, cls_token) map 1:1.
        pos_embed is interpolated bicubically (antialias=True) to (img_size // 14, img_size // 14).
        Register token and tied refinement block are freshly initialized new-by-design keys.
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        dino = {k: v for k, v in sd.items()}

        own = dict(self.state_dict())

        def _new_by_design(k: str) -> bool:
            return k.startswith("refinement.") or k in ("register_token",)

        trunk_keys = [k for k in own if not _new_by_design(k)]

        matched: Dict[str, torch.Tensor] = {}
        skipped: Dict[str, Any] = {}

        # 1. Handle pos_embed interpolation if requested
        if interpolate_pos_embed and "pos_embed" in dino:
            dino_pos = dino["pos_embed"]  # (1, 1 + N_src, D)
            cls_pos = dino_pos[:, :1, :]
            patch_pos = dino_pos[:, 1:, :]
            dim = patch_pos.shape[-1]
            src_grid = int(math.isqrt(patch_pos.shape[1]))
            if src_grid * src_grid == patch_pos.shape[1]:
                target_grid = self.img_size // PATCH_SIZE
                patch_pos_2d = patch_pos.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
                interp_patch = F.interpolate(
                    patch_pos_2d,
                    size=(target_grid, target_grid),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                interp_patch = interp_patch.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)
                # Keep register token position as zeros or existing own
                reg_pos = own["pos_embed"][:, 1:2, :]
                new_pos_embed = torch.cat([cls_pos, reg_pos, interp_patch], dim=1)
                if new_pos_embed.shape == own["pos_embed"].shape:
                    matched["pos_embed"] = new_pos_embed
                else:
                    skipped["pos_embed"] = {
                        "reason": "interpolated shape mismatch",
                        "own": tuple(own["pos_embed"].shape),
                        "interp": tuple(new_pos_embed.shape),
                    }
            else:
                skipped["pos_embed"] = {"reason": "source pos_embed is not square grid"}

        # 2. Match all other keys
        for k, v in dino.items():
            if k == "pos_embed":
                continue  # already handled above
            if k in own:
                if own[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped[k] = {
                        "reason": "shape mismatch",
                        "own": tuple(own[k].shape),
                        "dino": tuple(v.shape),
                    }
            else:
                skipped[k] = {"reason": "no counterpart in model"}

        matched_trunk = [k for k in matched if not _new_by_design(k)]
        matched_fraction = len(matched_trunk) / max(len(trunk_keys), 1)
        accepted = matched_fraction >= strict_band

        if accepted:
            super().load_state_dict({**own, **matched}, strict=True)

        return {
            "matched_keys": len(matched),
            "trunk_keys": len(trunk_keys),
            "matched_trunk_keys": len(matched_trunk),
            "matched_fraction": matched_fraction,
            "threshold": strict_band,
            "accepted": accepted,
            "skipped": skipped,
            "missing_new_by_design": [k for k in own if _new_by_design(k)],
        }
