"""
GlimpseFeaturePredictor — Agent E. The ONE predictor, two consumers.
================================================================================

THE most load-bearing implementation in Tier 2 (Part 4 Adjustment 1): this
class concretely implements Agent 0's shared interface
(noesis_vision.predictive_coding.interfaces.GlimpseFeaturePredictor) and is
consumed identically by Agent F (candidate scoring) and the belief update
(realized E_t). Neither consumer may reimplement its own copy.

Part 1.B's LOCKED DEFAULT, implemented: given `B_t` and a candidate/chosen
gaze location `a`, predict what the ENCODER will produce when it actually
looks there — LOCAL token/patch-level features (B, N, D_feat), NOT a global
z' (a pooled-vector self-prediction has little genuine signal, per the plan).

Conditioning (Agent 0's ABC docstring, satisfied exactly): the belief's
z_t AND U_t condition the prediction — inputs are [z_t (B, D_z), the
uncertainty scalar (B, 1), Fourier features of the location (B, 32)].

CHEAPNESS is a design constraint, enforced by test (Agent E contract):
predict_features' FLOPs must stay under 10% of ONE full backbone forward
pass. Architecture: a narrow residual MLP on the conditioning vector plus a
per-token head — explicitly NOT a second backbone (no patch embed, no
depth-12 self-attention stack). FLOPs scale with the hidden width, not the
trunk; test_predictor_is_cheap.py documents the multiply-accumulate
counting so the threshold is auditable.

Gradient rules (Part 1.A/1.B): predictions are NEVER detached here — the
observed target is detached at the consumer's call site; E_t is never
detached before the update. score_candidates routes through the SAME
predict_features (one module, two consumers — no separate scoring head).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from noesis_vision.predictive_coding.interfaces import GlimpseFeaturePredictor


class _LocationFourier(nn.Module):
    """Fourier features of the (x, y) gaze location — cheap spatial signal."""

    def __init__(self, n_freq: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """(B, 2) -> (B, 4 * n_freq): sin/cos of each coord at each freq."""
        ang = xy.unsqueeze(-1) * self.freqs          # (B, 2, F)
        return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)


class ConcreteGlimpseFeaturePredictor(GlimpseFeaturePredictor, nn.Module):
    """(z_t, U_t, location) -> predicted LOCAL token features, (B, N, D_feat).

    Narrow by construction: cond MLP (D_z + 1 + 32 -> hidden -> D_feat),
    one residual block, then a per-token head (D_feat -> N * D_feat). The
    head dominates the parameter count (~2.4M of ~2.8M) but runs ONCE per
    candidate — a tiny fraction of one trunk pass (measured, not asserted
    by hope: tests/test_predictor_is_cheap.py).
    """

    def __init__(self, d_z: int = 384, d_feat: int = 384, n_tokens: int = 16,
                 hidden: int = 256, n_freq: int = 8):
        super().__init__()
        self.d_z = d_z
        self.d_feat = d_feat
        self.n_tokens = n_tokens
        self.loc = _LocationFourier(n_freq=n_freq)
        n_in = d_z + 1 + 4 * n_freq
        self.cond = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(),
            nn.Linear(hidden, d_feat))
        self.res = nn.Sequential(
            nn.Linear(d_feat, hidden), nn.GELU(),
            nn.Linear(hidden, d_feat))
        self.token_head = nn.Linear(d_feat, n_tokens * d_feat)

    def _validate(self, belief, gaze_location: torch.Tensor) -> None:
        if gaze_location.dim() != 2 or gaze_location.shape[1] != 2:
            got = tuple(gaze_location.shape)
            raise ValueError(f"gaze_location must be (B, 2); got {got}")
        if bool((gaze_location.abs() > 1.0 + 1e-6).any()):
            raise ValueError(
                "gaze_location must lie in [-1, +1] (the A_t convention, "
                "Part 1.A) — a convention mismatch corrupts the prediction")
        if belief.z.shape[0] != gaze_location.shape[0]:
            raise ValueError(
                f"batch mismatch: belief.z {tuple(belief.z.shape)} vs gaze "
                f"{tuple(gaze_location.shape)}")
        if belief.z.shape[-1] != self.d_z:
            raise ValueError(
                f"belief.z dim {belief.z.shape[-1]} != predictor's d_z "
                f"{self.d_z} — conditioning must match the substrate")

    def predict_features(self, belief, gaze_location: torch.Tensor
                         ) -> torch.Tensor:
        """Predicted token features at a fixation, (B, N, D_feat).

        NOT detached — the predicted path carries gradient (the observed
        target is detached at the consumer's call site; E_t is never
        detached before the update, Part 1.A).
        """
        self._validate(belief, gaze_location)
        B = gaze_location.shape[0]
        u = belief.uncertainty.unsqueeze(-1).to(belief.z.dtype)   # (B, 1)
        cond_in = torch.cat([belief.z, u, self.loc(gaze_location)], dim=-1)
        cond = self.cond(cond_in)                                 # (B, D_feat)
        h = cond + self.res(cond)                                 # residual
        return self.token_head(h).reshape(B, self.n_tokens, self.d_feat)

    def forward(self, belief, gaze_location: torch.Tensor) -> torch.Tensor:
        """nn.Module callable form — routes through the interface method
        (the Module IS the predictor; no second entry point)."""
        return self.predict_features(belief, gaze_location)

    def score_candidates(self, belief, candidate_locations: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score K candidates through the SAME predict_features.

        Returns (predicted_features (B, K, N, D_feat), scores (B, K)).
        Per-candidate scores here are the predictor's cheap feature-based
        magnitude proxy; the UNCERTAINTY-REDUCTION scoring (Dirichlet-
        entropy deltas per Part 1.E) is Agent F's consumer wiring applied
        on top of these SAME predicted features — one predictor, two
        consumers: this module never grows a second scoring head.

        Compute constraint (interface-enforced): K cheap predictor passes,
        NEVER K backbone passes — this method runs the predictor K times,
        nothing more.
        """
        if candidate_locations.dim() != 3 or candidate_locations.shape[2] != 2:
            got = tuple(candidate_locations.shape)
            raise ValueError(
                f"candidate_locations must be (B, K, 2); got {got}")
        K = candidate_locations.shape[1]
        preds, scores = [], []
        for k in range(K):
            pk = self.predict_features(belief, candidate_locations[:, k, :])
            preds.append(pk)
            scores.append(pk.mean(dim=(1, 2)))   # cheap per-candidate scalar
        return torch.stack(preds, dim=1), torch.stack(scores, dim=-1)
