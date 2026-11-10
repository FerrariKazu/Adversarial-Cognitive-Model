"""The learned and the measured parts of the sensor.

EvidenceHead         a per-token linear readout trunk-token -> class score. It is the only new
                     trainable module of RHAN-IM-Sensor (385 * C parameters); the null class ("absent")
                     is not a head output because the Bayes state represents it as H_0.
EvidenceCalibration  measured, never trained: the null mean / std of each class score (so z is N(0, 1)
                     under H_0) and the detectability d' of every token position and of the gist look.
fit_calibration      turns held-out positive / null scores into those numbers.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from rhan_im.sdt import dprime_from_samples, zroc_slope


class EvidenceHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class EvidenceCalibration(nn.Module):
    """Measured, never trained. look type 0 = native-window token, 1 = gist token.

    mu0, sd0 are the null mean / std of every (look type, token position, class) score, so that the
    standardised score is N(0, 1) under H_0 at every position (edge tokens of a window, which see less
    context, are NOT assumed to share the centre tokens' null statistics). d_tok / d_gist are the measured
    detectabilities that go into the Bayes update.

    vif_tok / vif_gist (default 1 = tokens of one look are independent evidence) are the measured
    within-look variance-inflation factors (gate1.block_vif). For equicorrelated tokens the exact
    log-likelihood ratio equals the independent-token sum divided by VIF, which is what scaling both the
    standardised scores and the detectabilities by 1/sqrt(VIF) produces; `standardize` and `d_tok_eff` /
    `d_gist_eff` apply it, so the Bayes state needs no change."""

    def __init__(self, n_classes: int, tok: int) -> None:
        super().__init__()
        self.register_buffer("mu0", torch.zeros(2, tok * tok, n_classes))
        self.register_buffer("sd0", torch.ones(2, tok * tok, n_classes))
        self.register_buffer("d_tok", torch.zeros(tok, tok))
        self.register_buffer("d_gist", torch.zeros(()))
        self.register_buffer("vif_tok", torch.ones(()))
        self.register_buffer("vif_gist", torch.ones(()))

    def standardize(self, scores: torch.Tensor, look_type: int) -> torch.Tensor:
        """scores (b, T, C) -> standardised (and VIF-corrected) scores of the same shape."""
        vif = self.vif_tok if look_type == 0 else self.vif_gist
        return (scores - self.mu0[look_type]) / self.sd0[look_type] / vif.sqrt()

    @property
    def d_tok_eff(self) -> torch.Tensor:
        return self.d_tok / self.vif_tok.sqrt()

    @property
    def d_gist_eff(self) -> torch.Tensor:
        return self.d_gist / self.vif_gist.sqrt()


def fit_calibration(pos: torch.Tensor, null: torch.Tensor) -> Dict[str, float]:
    """pos: scores of the target class at units that contain it; null: scores of the same class at
    units that do not. Returns the null standardiser and the measured d' and z-ROC slope."""
    mu0, sd0 = null.mean().item(), null.std().item()
    zp, zn = (pos - mu0) / sd0, (null - mu0) / sd0
    slope, _ = zroc_slope(zp, zn)
    return {"mu0": mu0, "sd0": sd0, "d_prime": dprime_from_samples(zp, zn), "zroc_slope": slope}
