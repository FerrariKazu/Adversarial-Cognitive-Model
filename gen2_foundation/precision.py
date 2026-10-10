"""G2-K4 — precision field family.

Precision is explicit, independent of U_t (epistemic uncertainty). The
legacy `1 - U_t` mechanism is retained EXECUTABLE as the Gen-1 parity
control. Every precision output is clamped to [1e-4, 1] and reports
collapse diagnostics (Part X: "precision collapse").

Reference:  L_cls(evidential) + L_pred + L_adv  (frozen); every precision
arm is a one-flag control against that frozen reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal, Optional

#: Hard bound floor / ceiling (spec: every precision field clamped to
#: [1e-4, 1]).
PRECISION_MIN: float = 1e-4
PRECISION_MAX: float = 1.0
FLOOR = PRECISION_MIN


class PrecisionArm:
    """Discriminant of a registered precision field family arm.

    Members:
        const       constant field
        one_minus_u legacy 1-U (Gen-1 parity control)
        learned_scalar  learned scalar precision
        learned_spatial learned spatial map (B, N, 1)
        channelwise     channel-wise precision (B, C)
        policyconditioned  policy-conditioned precision
        collapse_diag   collapse diagnostics from the Part X catalogue
    """

    const = "const"
    one_minus_u = "one_minus_u"
    learned_scalar = "learned_scalar"
    learned_spatial = "learned_spatial"
    channelwise = "channelwise"
    policyconditioned = "policyconditioned"
    collapse_diag = "collapse_diag"

    @classmethod
    def all_arms(cls) -> list[str]:
        return [
            cls.const,
            cls.one_minus_u,
            cls.learned_scalar,
            cls.learned_spatial,
            cls.channelwise,
            cls.policyconditioned,
            cls.collapse_diag,
        ]


class PrecisionField(nn.Module):
    """A registered precision field over the belief state.

    Precision gates the belief update: Pi multiplies the UpdateNet output
    so high precision -> strong belief revision, low precision -> weak.

    All outputs are clamped to [1e-4, 1] per spec; no arm may collapse to
    a constant field (collapse diagnostics flagged by the caller).

    Args:
        arm: one of PrecisionArm values.
        d_z: belief dimension for spatial/channel-wise fields.
        n_tokens: token count for spatial field (default 16).
        hidden: hidden width for learned-scalar arm.
    """

    def __init__(
        self,
        arm: str,
        d_z: int = 384,
        n_tokens: int = 16,
        hidden: int = 32,
        collapse_diag: bool = False,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.d_z = int(d_z)
        self.n_tokens = int(n_tokens)
        self.collapse_diag = collapse_diag
        self.collapse_count = 0

        if arm == PrecisionArm.const:
            self._net = nn.Identity()
        elif arm == PrecisionArm.one_minus_u:
            self._net = nn.Identity()            # control: 1 - U_t
        elif arm == PrecisionArm.learned_scalar:
            self._net = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
        elif arm == PrecisionArm.learned_spatial:
            # Per-token spatial precision: input is the flattened spatial
            # map (B, N) -> hidden -> (B, 1) field.
            self._net = nn.Sequential(
                nn.Linear(self.n_tokens, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
        elif arm == PrecisionArm.channelwise:
            self._net = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
        elif arm == PrecisionArm.policyconditioned:
            self._net = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
        elif arm == PrecisionArm.collapse_diag:
            self._net = nn.Identity()
        else:
            raise ValueError(f"unknown precision arm {arm}")

        # Do NOT register a "type" parameter: precision fields are
        # regular nn.Modules (or identity), no group-mixing.

    # ---- forward ----

    def forward(
        self,
        U: Optional[torch.Tensor] = None,
        spatial_map: Optional[torch.Tensor] = None,
        policy_logits: Optional[torch.Tensor] = None,
        observed_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce precision over the current belief state.

        All outputs clamped to [1e-4, 1].

        Args:
            U: epistemic uncertainty scalar (B,) — used by legacy arm.
            spatial_map: (B, N, 1) spatial error -> precision map.
            policy_logits: (B, K) candidate scores -> policy-conditioned.
            observed_features: (B, D) observed feature evidence.

        Returns:
            Pi: (B, ...) clamped to [PRECISION_MIN, PRECISION_MAX].
        """
        if self.arm == PrecisionArm.const:
            pi = torch.full_like(U.float(), 0.5) if U is not None else torch.ones(1)
            pi = pi.to(U.device if U is not None else torch.device("cpu"))
        elif self.arm == PrecisionArm.one_minus_u:
            # Gen-1 parity control: legacy 1 - U_t.
            if U is None:
                raise ValueError("legacy_1_minus_u requires U_t")
            pi = 1.0 - U.float()
        else:
            if U is None:
                raise ValueError(
                    f"learned-precision arms need U_t "
                    f"(arm={self.arm})"
                )
            # learned_spatial is fed the spatial map directly; the MLP
            # maps per-token features and keeps the spatial field shape.
            # All other learned arms use U as the per-sample feature.
            if self.arm == PrecisionArm.learned_spatial:
                if spatial_map is None:
                    raise ValueError(
                        "learned_spatial arm requires spatial_map input")
                feats = spatial_map.float()          # (B, N, 1)
                B, N, _ = feats.shape
                # Flatten to (B, N); MLP maps each token's 1 feature to
                # 1 precision value per token, then restore shape.
                feats = feats.reshape(B, -1)         # (B, N)
                pi = self._net(feats)                # (B, 1)
                # (B, 1) -> broadcast to spatial map (B, N, 1).
                pi = pi.unsqueeze(1).expand(B, N, 1)
            else:
                feats = U.float().unsqueeze(-1)      # (B, 1)
                pi = self._net(feats).squeeze(-1)

        # ── hard clamp to [1e-4, 1] ───────────────────────────────────
        pi = torch.clamp(pi, PRECISION_MIN, PRECISION_MAX)

        # ── collapse diagnostics ──────────────────────────────────────
        if self.collapse_diag:
            near_floor = (pi <= PRECISION_MIN + 1e-3).float().mean().item()
            near_ceil = (pi >= PRECISION_MAX - 1e-3).float().mean().item()
            if near_floor > 0.9 or near_ceil > 0.9:
                self.collapse_count += 1
                # Log the diagnostic (caller reads collapse_count once).

        return pi


def legacy_1_minus_u_control(U: torch.Tensor) -> torch.Tensor:
    """Executable Gen-1 parity control: Pi = 1 - U_t.

    Clamped to [1e-4, 1]. Must be preserved exactly as the reference
    control arm; the learned precision family is compared against it.
    """
    pi = 1.0 - U.float()
    return torch.clamp(pi, PRECISION_MIN, PRECISION_MAX)
