"""
CompactViT + encode_glimpse — Agent C. The RHAN-NXA substrate.
================================================================================

Part 1.A: `z_t` is the "pooled CLS-token-equivalent output of the
within-glimpse recurrent transformer (see 1.C)"; D_z = backbone embed dim,
"locked once the substrate (Agent C) sets it". THIS module sets it:
**d_z = 384** — the trunk is DINOv2-small-shaped (patch 14, depth 12,
width 384) so the warm-start maps cleanly (the Agent C contract's STOP
condition), and d_z = 384 is the only width compatible with that mapping.
RECORD d_z = 384 in the run's RHANNXAConfig (the schema field stays None
until a run constructs this backbone; the substrate decision propagates
through the config hash, not by magic).

Parameter budget (Agent C contract: 20-25M excluding pillar heads):
  * trunk: 12 DINOv2-small-shaped blocks @ width 384, head dim 48:
    ~21.29M blocks + ~0.23M patch embed (14x14x384) + pos/cls/register
    tokens + final norm ~= 21.53M — in band on its own;
  * tied refinement block: ONE standard block (mlp_ratio 4), applied 2-3
    times (tied => params independent of the iteration count): ~1.77M.
    New-by-design machinery (no warm-start counterpart — reported as such
    by the loader, see recurrent_block.py);
  * total: ~23.3M, asserted into [20M, 25M] by test_backbone_param_count.

Glimpse count T=4 is schema-locked (num_glimpses); encode_glimpse is
ONE glimpse — the T-loop belongs to the integration layer (Agent J), not
the substrate.

encode_glimpse(image, gaze_coords) -> (pooled_z, tokens):
    THE load-bearing signature consumed by Agent E (prediction target)
    and Agent F (candidate scoring) — Agent 0's Part 1.B predictor
    interface is specified against these shapes ((B, D_z) pooled,
    (B, N, D_feat) token features). Do not change this signature without
    flagging to Agent 0 (the contract's own rule).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from noesis_vision.models.recurrent_block import (
    WITHIN_GLIMPSE_ITERS_RANGE,
    TiedRecurrence,
    TransformerBlock,
)
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE, foveal_sample

#: The substrate's width — Part 1.A's D_z, set HERE per the plan's
#: "locked once the substrate (Agent C) sets it".
D_Z = 384

#: DINOv2-small architecture constants (the clean-mapping anchors).
PATCH_SIZE = 14
TRUNK_DEPTH = 12
TRUNK_WIDTH = 384
TRUNK_HEADS = 6

#: Refinement block: standard mlp_ratio 4. (An mlp_ratio-8 widening was
#: considered for budget margin but is unnecessary: trunk ~21.5M + a
#: standard block ~1.8M = ~23.3M, inside the band with margin both sides.)
REFINE_MLP_RATIO = 4.0

#: Contract band for the parameter count (excluding pillar heads).
PARAM_COUNT_BAND = (20_000_000, 25_000_000)

#: Warm-start threshold: below this matched fraction the load is NOT
#: accepted silently — the caller decides (STOP condition, Agent C contract).
WARM_START_MATCH_FRACTION_MIN = 0.95


class CompactViT(nn.Module):
    """DINOv2-small-shaped trunk + tied within-glimpse refinement.

    Trunk: patch embed (14, 14) + cls token + register token + 12 blocks +
    final norm — submodule naming mirrors DINOv2 (patch_embed.proj,
    blocks.N.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}, norm) so
    load_dino_warm_start maps keys 1:1.

    Refinement: ONE TiedRecurrence block at the same width, applied
    num_refine_iters times per glimpse (LOCKED range 2-3).
    """

    def __init__(self, img_size: int = DEFAULT_FOVEA_SIZE,
                 num_refine_iters: int = 2,
                 refine_mlp_ratio: float = REFINE_MLP_RATIO):
        super().__init__()
        lo, hi = WITHIN_GLIMPSE_ITERS_RANGE
        if not (lo <= num_refine_iters <= hi):
            raise ValueError(
                f"num_refine_iters must be in the LOCKED range {lo}-{hi} "
                f"(Part 1.C); got {num_refine_iters}")
        if img_size % PATCH_SIZE != 0:
            raise ValueError(
                f"img_size {img_size} not divisible by patch size "
                f"{PATCH_SIZE} — a clean patch-embed mapping is required "
                f"(Agent C STOP condition)")
        self.img_size = img_size
        self.num_refine_iters = num_refine_iters
        self.d_z = D_Z

        self.patch_embed = nn.ModuleDict({
            "proj": nn.Conv2d(3, TRUNK_WIDTH, kernel_size=PATCH_SIZE,
                              stride=PATCH_SIZE, bias=True),
        })
        num_patches = (img_size // PATCH_SIZE) ** 2
        self.num_prefix_tokens = 2  # cls + register (DINOv2 layout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, TRUNK_WIDTH))
        self.register_token = nn.Parameter(torch.zeros(1, 1, TRUNK_WIDTH))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, TRUNK_WIDTH))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_token, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(TRUNK_WIDTH, TRUNK_HEADS) for _ in range(TRUNK_DEPTH))
        self.norm = nn.LayerNorm(TRUNK_WIDTH)

        self.refinement = TiedRecurrence(
            TransformerBlock(TRUNK_WIDTH, TRUNK_HEADS,
                             mlp_ratio=refine_mlp_ratio))

    # ── forward pieces ──────────────────────────────────────────────────────
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) -> (B, 1+1+N, D): patchify + cls/register + pos."""
        B = x.shape[0]
        tokens = self.patch_embed["proj"](x)                    # (B, D, n, n)
        tokens = tokens.flatten(2).transpose(1, 2)              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.register_token.expand(B, -1, -1)
        x = torch.cat([cls, reg, tokens], dim=1)                # (B, 2+N, D)
        return x + self.pos_embed

    def _trunk_forward(self, x_prep: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x_prep = blk(x_prep)
        return self.norm(x_prep)

    # ── the load-bearing interface ──────────────────────────────────────────
    def encode_glimpse(self, image: torch.Tensor,
                       gaze_coords: torch.Tensor,
                       fovea_size: int = DEFAULT_FOVEA_SIZE
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One glimpse: foveate, encode, refine, pool.

        Args:
            image: (B, 3, H, W) full image (H = W).
            gaze_coords: (B, 2) normalized [-1, +1] (x, y); gradients flow
                through the crop into this argument.
            fovea_size: crop size; must be divisible by PATCH_SIZE.

        Returns:
            (pooled_z, tokens):
                pooled_z: (B, D_z) — the CLS-token-equivalent pooled
                    content (Part 1.A's z_t), after tied refinement;
                tokens: (B, N, D_feat) — patch-token features at the
                    fixation (N = (fovea_size/14)^2), the predictor's
                    target/score space (Part 1.B). Gradient-carrying.
        """
        crop = foveal_sample(image, gaze_coords, fovea_size=fovea_size)
        x = self._prep(crop)
        x = self._trunk_forward(x)
        x = self.refinement(x, self.num_refine_iters)           # tied 2-3 iters
        pooled_z = x[:, 0]                                      # cls position
        tokens = x[:, self.num_prefix_tokens:]                  # (B, N, D)
        return pooled_z, tokens

    # ── warm start ──────────────────────────────────────────────────────────
    def load_dino_warm_start(self, path: str,
                             strict_band: float = WARM_START_MATCH_FRACTION_MIN
                             ) -> Dict[str, object]:
        """Load DINOv2-small backbone weights with an exact match report.

        The trunk's keys map 1:1 by name (naming mirrors DINOv2). The
        threshold metric is the matched fraction over the MAPPABLE trunk
        keys — new-by-design keys (the refinement block, pos/cls/register
        machinery) are excluded from the denominator because they are not
        supposed to have a warm-start counterpart; counting them would
        self-refuse a perfectly clean mapping. Both the trunk fraction and
        the full key accounting are reported.

        Returns a report dict: {matched_keys, skipped, missing_new_by_design,
        matched_fraction, threshold, accepted}. Nothing is loaded partially
        and silently: if matched_fraction < strict_band, accepted=False and
        NO weights are applied (the caller decides — STOP condition, Agent C
        contract; report the mismatch, never force a degraded load).
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        dino = {k: v for k, v in sd.items()}

        own = dict(self.state_dict())

        # New-by-design keys: no warm-start counterpart is expected.
        def _new_by_design(k: str) -> bool:
            return k.startswith("refinement.") or k in (
                "pos_embed", "cls_token", "register_token")

        trunk_keys = [k for k in own if not _new_by_design(k)]

        matched, skipped = {}, {}
        for k, v in dino.items():
            if k in own:
                if own[k].shape == v.shape:
                    matched[k] = v
                else:
                    skipped[k] = {"reason": "shape mismatch",
                                  "own": tuple(own[k].shape),
                                  "dino": tuple(v.shape)}
            else:
                skipped[k] = {"reason": "no counterpart"}

        matched_trunk = [k for k in matched if not _new_by_design(k)]
        matched_fraction = len(matched_trunk) / max(len(trunk_keys), 1)
        accepted = matched_fraction >= strict_band
        if accepted:
            # Warm-start the trunk; new-by-design keys (refinement block,
            # pos/cls/register naming) keep their fresh init — applied
            # explicitly, never partially-and-silently.
            super().load_state_dict({**own, **matched}, strict=True)
        return {
            "matched_keys": len(matched),
            "trunk_keys": len(trunk_keys),
            "matched_trunk_keys": len(matched_trunk),
            "skipped": skipped,
            "missing_new_by_design": [k for k in own if _new_by_design(k)],
            "matched_fraction": matched_fraction,
            "threshold": strict_band,
            "accepted": accepted,
        }
