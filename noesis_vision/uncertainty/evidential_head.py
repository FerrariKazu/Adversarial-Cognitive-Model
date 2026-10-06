"""
EvidentialHead + the canonical DirichletParams — Agent D.
================================================================================

THE ONE UNCERTAINTY REPRESENTATION (Part 1.F, LOCKED): a single
class-readout Dirichlet formulation feeding all three consumers — Pi_t
(precision, Part 1.B), AIS-v2 candidate scoring (Part 1.E), and L_stab's
drift metric (Part 1.G). No second uncertainty mechanism may be
introduced; Agent B's placeholder DirichletParams is superseded by THIS
definition (see "Supersession" below).

PORT-TABLE DISCREPANCY — FLAGGED, NOT SILENT (Part 5; non-improvisation
rule): the port table grades EvidentialHead PORT VERBATIM ("already
exists, already validated in isolation"), but no class by that name (nor
any class-readout Dirichlet/evidential head) exists anywhere in the Gen-0
codebase — the closest machinery is SBR's slot-attention ENTROPY
(rhan_core/beliefs/structured_belief.py), a different mechanism. This
module is therefore built to the Agent D contract's explicit spec
(softplus evidence, alpha = evidence + 1, C / sum(alpha) uncertainty,
entropy, two-sided clamp) rather than pretended to be a verbatim port.
Raised to the plan's author in the Agent D handoff; until reconciled,
treat "PORT VERBATIM" for this row as unverified.

Supersession (Agent B contract): DirichletParams HERE is canonical the
moment this file exists. Agent B's placeholder must be DELETED and
replaced by importing this — Agent J's integration checklist must include
exactly that; the two definitions must not coexist past integration.
(Beliefs that construct via noesis_vision.beliefs continue to work: the
field name `evidence` and the alpha/uncertainty formulas are identical.)

Numerical stability (contract: clamped and TESTED, not hoped for):
evidence = clamp(softplus(raw), EVIDENCE_CLAMP_MIN, EVIDENCE_CLAMP_MAX).
  * high end: raw -> +inf (extreme-magnitude features) would push alpha to
    float32 overflow downstream; the clamp caps evidence at 1e4 (alpha at
    10001 — digamma and the entropy formula are stable there);
  * low end: evidence is floored at a tiny positive value so alpha > 1
    STRICTLY for every input (the smoke experiment's assertion) and
    log/entropy consumers never see alpha == 1 edge behavior;
  * clamp even sanitizes +inf raw inputs to the cap. NaN input still
    propagates (garbage-in is not silently repaired — that would be a
    false guarantee).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Two-sided evidence clamp (see module docstring). Both ends are contract
#: requirements and are exercised by tests/test_numerical_stability.py.
EVIDENCE_CLAMP_MIN = 1e-6
EVIDENCE_CLAMP_MAX = 1e4


@dataclass
class DirichletParams:
    """The CANONICAL U-carrier (Part 1.A / 1.F) — supersedes Agent B's
    placeholder (Agent J integration checklist: delete the placeholder,
    import this)."""

    evidence: torch.Tensor  # (B, C), non-negative (softplus, clamped)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.evidence) or self.evidence.dim() != 2:
            got = (tuple(self.evidence.shape)
                   if torch.is_tensor(self.evidence)
                   else type(self.evidence).__name__)
            raise ValueError(f"evidence must be (B, C); got {got}")
        if bool((self.evidence < 0).any()):
            raise ValueError(
                "evidence must be non-negative (softplus output, Part 1.A)")

    @property
    def alpha(self) -> torch.Tensor:
        """alpha_t = e_t + 1 (Part 1.A), shape (B, C). Strictly > 1 per
        class (evidence is clamped positive)."""
        return self.evidence + 1.0

    @property
    def uncertainty(self) -> torch.Tensor:
        """Uncertainty scalar: C / sum(alpha_t) (Part 1.A), shape (B,)."""
        return self.evidence.shape[-1] / self.alpha.sum(dim=-1)

    def uncertainty_scalar(self) -> torch.Tensor:
        """The Part 1.A uncertainty scalar, (B,) — named accessor."""
        return self.uncertainty

    def entropy(self) -> torch.Tensor:
        """Differential entropy of Dirichlet(alpha), shape (B,).

        H = psi(sum_a) - sum_k psi(a_k) + sum_k (a_k - 1)(psi(a_k) - psi(sum_a))
        (psi = digamma). This is THE uncertainty quantity AIS-v2's candidate
        scoring predicts reductions of (Part 1.E) — one uncertainty
        representation, no separate scoring head.
        """
        a = self.alpha
        total = a.sum(dim=-1)                        # (B,)
        psi_k = torch.special.digamma(a)             # (B, C)
        psi_s = torch.special.digamma(total)         # (B,)
        return (psi_s - psi_k.sum(dim=-1)
                + ((a - 1.0) * (psi_k - psi_s.unsqueeze(-1))).sum(dim=-1))


class EvidentialHead(nn.Module):
    """Features -> class-readout Dirichlet evidence (Part 1.F).

    Small learned head: one hidden layer, softplus + two-sided clamp.
    NON-DETACHED by construction: forward returns DirichletParams whose
    evidence carries the autograd graph back into this head's parameters —
    the gradient requirement is checked from day one
    (tests/test_evidential_gradient_flow.py), the Gen-0 failure class.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 256):
        super().__init__()
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError(
                f"input_dim/num_classes must be positive; got "
                f"{input_dim}/{num_classes}")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> DirichletParams:
        """(B, input_dim) -> DirichletParams with evidence (B, C).

        Accepts (B, N, input_dim) token features too (pools by mean) so
        AIS-v2's scoring can feed predicted token features directly
        (Part 1.E) — pooling is part of the head, not a second mechanism.
        """
        if features.dim() == 3:
            features = features.mean(dim=1)
        if features.dim() != 2 or features.shape[-1] != self.input_dim:
            got = tuple(features.shape)
            raise ValueError(
                f"features must be (B, {self.input_dim}) or (B, N, "
                f"{self.input_dim}); got {got}")
        # Entry clamp: +/-inf features would go NaN INSIDE the MLP
        # (gelu(-inf) * 0) before the evidence clamp could act — sanitize
        # at the front door instead. NaN passes through clamp untouched:
        # the honest-propagation contract is unchanged.
        features = torch.clamp(features, min=-1e4, max=1e4)
        raw = self.fc2(F.gelu(self.fc1(features)))
        evidence = torch.clamp(
            F.softplus(raw),
            min=EVIDENCE_CLAMP_MIN, max=EVIDENCE_CLAMP_MAX)
        return DirichletParams(evidence=evidence)
