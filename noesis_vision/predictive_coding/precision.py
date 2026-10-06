"""
Precision g(U_t) -> Pi_t — Agent E.
================================================================================

Part 1.B/1.F: precision is a SMALL LEARNED FUNCTION of the uncertainty
representation — not a fixed scalar. One uncertainty representation
(Part 1.F): g reads Agent D's canonical DirichletParams (the U_t carrier)
through its own scalars — the uncertainty scalar C/sum(alpha) and the
Dirichlet entropy — and maps them through a tiny MLP to a positive
per-sample precision.

  * POSITIVE by construction: softplus output (+ tiny floor). Precision
    scales how strongly the update is applied (Part 1.B); a non-positive
    Pi would INVERT the correction term.
  * LEARNED, own optimizer group (Agent E contract): this module carries
    the parameters; registration under group name "precision" happens in
    Agent A's registry at wiring time — test_precision_own_optimizer_group
    introspects the registry to confirm the isolation (never lumped with
    the backbone — the Stage-2 starvation lesson).
  * Gradient rules: g's input is the belief's evidence — NON-detached, so
    gradient reaches the EvidentialHead through g when the loss wants it
    (one uncertainty representation serving three consumers). A detached
    variant is available for diagnostics ONLY (detached=True argument,
    never the default) — the default path is the training path.
  * The monotonic DIRECTION (high uncertainty -> low precision) is the
    intended behavior for the update scaling; it is NOT hard-coded via a
    sign flip — the learned function can express it, and Agent H's staged
    diagnostics will verify it empirically. Hard-coding a direction would
    be inventing a mechanism the plan does not specify.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: Softplus floor so Pi > 0 strictly even at extreme negative pre-acts.
PI_FLOOR = 1e-4


class PrecisionFunction(nn.Module):
    """g: DirichletParams -> Pi_t, (B,), positive, learned.

    Inputs per sample (2 scalars): the Part 1.A uncertainty
    C/sum(alpha_t) and the Dirichlet entropy — both derived from the ONE
    canonical evidence representation (no second uncertainty mechanism).
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, U, detached: bool = False) -> torch.Tensor:
        """DirichletParams -> Pi_t, (B,), strictly positive.

        Args:
            U: the canonical DirichletParams (Agent D) — anything exposing
                .uncertainty (C/sum(alpha)) and .entropy().
            detached: diagnostics-only switch (NEVER default — the
                training path must stay non-detached).
        """
        u_scalar = U.uncertainty                    # (B,)
        u_entropy = U.entropy()                     # (B,)
        feats = torch.stack([u_scalar, u_entropy], dim=-1)   # (B, 2)
        if detached:
            feats = feats.detach()
        raw = self.net(feats).squeeze(-1)           # (B,)
        return torch.nn.functional.softplus(raw) + PI_FLOOR
