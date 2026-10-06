"""
UpdateNet + the belief-update equation — Agent E.
================================================================================

Part 1.B's REVISED UPDATE EQUATION, implemented exactly:

    z_{t+1} = z_t + Pi_t * UpdateNet(z_t, E_t)

  * UpdateNet is a small learned MLP — NOT a raw linear addition (the
    `z_t + lambda*Pi*E_t` draft is superseded/REJECTED). The bridge from
    E_t's LOCAL token-feature space to z_t's GLOBAL space happens HERE:
    mean-pool the token error to (B, D_z) BEFORE the MLP — pooling is
    dimensionality reduction, not a learned shortcut, so the "not assumed
    compatible without a learned mapping" rule is honored (the mapping IS
    the MLP).
  * Pi_t scales OUTSIDE UpdateNet (Part 1.B): UpdateNet outputs the
    update term; the caller applies precision.
  * The output is BOUNDED (bounded_delta, Agent E contract): tanh-scaled
    to [-BOUND, +BOUND] per component, so a single glimpse's update can
    never blow the belief away — a runaway z_{t+1} would be the same
    instability class the Gen-0 drift incidents exhibited.

t = 0 handling — LOCKED, implemented exactly (MASTER_PLAN Part 1.B):
    E_0 := 0 (zero tensor, correct shape, no gradient contribution);
    z_1 = z_0 + Pi_0 * UpdateNet(z_0, 0)  — pure observation, no
    correction term; prediction and genuine E_t begin at t = 1.
    belief_update() implements the update for BOTH cases from one code
    path; test_t0_produces_exact_zero_error pins the value-exact behavior.

Engineering contract (Part 1.B): UpdateNet is small, new, and
independently gradient-isolated — its own optimizer group + pre-flight
|dW| check BEFORE any smoke test involving real gradient steps
(scripts/measure_group_dw.py's pattern, ported as the pre-flight test
tests/test_preflight_dw.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from noesis_vision.predictive_coding.interfaces import UpdateNet

#: Bounded delta: per-component update magnitude cap (contract's
#: "bounded_delta"). 0.1 of the feature scale — a per-glimpse update is a
#: correction, not a rewrite.
DELTA_BOUND = 0.1


class ConcreteUpdateNet(UpdateNet, nn.Module):
    """(z_t, E_t) -> bounded update term, (B, D_z).

    Small MLP (hidden 2x d_z, GELU) with a tanh bound. Own optimizer group
    (update_net) per the gradient-isolation rule — registration is Agent
    A's registry at wiring time; this class only carries the parameters.
    """

    def __init__(self, d_z: int = 384, hidden: int = 768,
                 delta_bound: float = DELTA_BOUND):
        super().__init__()
        self.d_z = d_z
        self.delta_bound = float(delta_bound)
        self.net = nn.Sequential(
            nn.Linear(2 * d_z, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_z))

    def _pool_error(self, E: torch.Tensor) -> torch.Tensor:
        """Token-feature error (B, N, D_feat) -> pooled (B, D_z).

        D_feat == D_z for the substrate (both 384); a mismatch raises —
        no silent projection (Part 1.B: the spaces are comparable BY
        CONSTRUCTION via the same encoder; if a future substrate splits
        them, that is an explicit design change, not a cast).
        """
        if E.dim() == 1:
            E = E.unsqueeze(0)
        if E.dim() == 2:
            pooled = E
        elif E.dim() == 3:
            pooled = E.mean(dim=1)
        else:
            got = tuple(E.shape)
            raise ValueError(f"E_t must be (B, D_z) or (B, N, D_feat); got {got}")
        if pooled.shape[-1] != self.d_z:
            raise ValueError(
                f"E_t's feature dim {pooled.shape[-1]} != d_z {self.d_z} — "
                f"the predictor's output space must match z's (same encoder, "
                f"Part 1.B); introducing a projection is a design change, "
                f"not a silent cast")
        return pooled

    def forward(self, z: torch.Tensor, prediction_error: torch.Tensor
                ) -> torch.Tensor:
        """The update term, (B, D_z), bounded to [-BOUND, +BOUND].

        NOT detached: the predicted-error path must carry gradient (the
        single most repeated Gen-0 failure mode — checked from day one).
        """
        if z.dim() != 2 or z.shape[-1] != self.d_z:
            got = tuple(z.shape)
            raise ValueError(f"z must be (B, {self.d_z}); got {got}")
        pooled = self._pool_error(prediction_error)
        if pooled.shape[0] != z.shape[0]:
            raise ValueError(
                f"batch mismatch: z {tuple(z.shape)} vs E_t "
                f"{tuple(pooled.shape)}")
        h = self.net(torch.cat([z, pooled], dim=-1))
        return self.delta_bound * torch.tanh(h)


def belief_update(z_t: torch.Tensor, E_t: torch.Tensor, Pi_t: torch.Tensor,
                  update_net: ConcreteUpdateNet) -> torch.Tensor:
    """z_{t+1} = z_t + Pi_t * UpdateNet(z_t, E_t) — exact.

    Args:
        z_t: (B, D_z) current content.
        E_t: (B, D_z) or (B, N, D_feat) prediction error. At t = 0 pass
            the ZERO tensor (LOCKED E_0 := 0) — the update reduces to
            pure observation; see test_t0_produces_exact_zero_error.
        Pi_t: (B,) or broadcastable precision (Agent E's precision module;
            Agent D's uncertainty feeds it).
        update_net: the (own-optimizer-group) update network.

    Returns:
        z_{t+1}, (B, D_z). Gradient flows into z_t, E_t's predicted path,
        Pi_t, and UpdateNet's parameters.
    """
    if not torch.is_tensor(Pi_t):
        raise ValueError(
            f"Pi_t must be a tensor (scalar-like (B,) or (B, 1)); got "
            f"{type(Pi_t).__name__} — a python float would silently drop "
            f"the precision gradient path")
    if Pi_t.dim() == 1:
        Pi_t = Pi_t.unsqueeze(-1)   # (B,) -> (B, 1): scale per-SAMPLE, not
        # per-feature (a bare (B,) against (B, D_z) would broadcast against
        # the trailing dim — caught by test_exact_equation_value_equality).
    delta = update_net(z_t, E_t)
    return z_t + Pi_t * delta
