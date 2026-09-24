"""
BeliefState interface stubs — Agent 0 output (stub-only).

Agent B implements against this ABC. Shapes are Part 1.A VERBATIM
(MASTER_PLAN); every stub signature carries shapes in its docstring,
not just types. Every member raises NotImplementedError naming the
future agent responsible (Agent B) — clean, expected, never an
unrelated error.

None-safety (Part 1.A): S_t is None for the core build. EVERY consumer
of a BeliefState must have an explicit branch for `S_t is None`, tested
directly — including `as_tensor`, `drift_to`, the classifier head, and
F itself. This is not a hypothetical: it is the actual default
configuration for Gen-1's first several phases (Part 1.D).

Gradient contract (Part 1.A):
  - z_t and U_t carry gradients ALWAYS.
  - S_t carries gradients only when not None.
  - E_t is computed FROM a detached "observed" target and a
    non-detached "predicted" value; E_t itself must NEVER be detached
    before use in the update (the single most repeated failure mode in
    Gen 0's history was exactly this bug class).
  - A_t carries NO gradient (a coordinate record; differentiability
    lives in the policy that produced it).

Serialization (Part 1.A): B_t is NOT checkpointed across images — it
exists only within one image's forward pass (T glimpses), then is
discarded. What gets checkpointed is the MODEL, never a B_t instance.
"""

from __future__ import annotations

from abc import ABC
from typing import Optional, Sequence

import torch


class BeliefState(ABC):
    """B_t = (z_t, S_t, U_t, E_t, A_t) — Part 1.A, verbatim.

    Stub-only: every member raises NotImplementedError("Agent B: ...").
    Instantiable (no abstract methods) so smoke tests can call each
    member once and observe the clean, expected error.
    """

    @property
    def z(self) -> torch.Tensor:
        """Global content `z_t`, shape (B, D_z).

        D_z = backbone embed dim (set by the substrate, Agent C). Dense,
        continuous, differentiable; pooled CLS-token-equivalent output
        of the within-glimpse recurrent transformer. Carries gradients
        always.
        """
        raise NotImplementedError(
            "Agent B: implement z_t content storage per Part 1.A"
        )

    @property
    def s(self) -> Optional[torch.Tensor]:
        """Structured state `S_t`, shape (B, K, D_s) or **None**.

        DEFAULT None for the core build (Part 1.D). When present, an
        instance of the existing BeliefState ABC's structured variant —
        not a new type. Carries gradients only when not None. Every
        consumer must branch explicitly on None.
        """
        raise NotImplementedError(
            "Agent B: implement None-safe S_t handling per Part 1.A/1.D"
        )

    @property
    def evidence(self) -> torch.Tensor:
        """Dirichlet evidence `e_t`, shape (B, C) [C = num classes].

        Non-negative (softplus output). Carries gradients always.
        """
        raise NotImplementedError(
            "Agent B: implement Dirichlet evidence carrier per Part 1.A/1.F"
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Dirichlet concentration `alpha_t = e_t + 1`, shape (B, C)."""
        raise NotImplementedError(
            "Agent B: implement alpha_t = e_t + 1 per Part 1.A"
        )

    @property
    def uncertainty(self) -> torch.Tensor:
        """Uncertainty scalar per sample: `C / sum(alpha_t)`, shape (B,).

        From the class-readout Dirichlet (Part 1.A's stated tension:
        this is uncertainty over the classification readout, kept for
        Gen-1; representation-level uncertainty is PENDING DECISION).
        """
        raise NotImplementedError(
            "Agent B: implement C / sum(alpha_t) per Part 1.A"
        )

    @property
    def prediction_error(self) -> torch.Tensor:
        """Latent prediction error `E_t` — Part 1.B's design.

        Token-feature space, same space as the shared predictor's
        outputs. At t = 0 this is the zero tensor (FIRST_GLIMPSE
        boundary condition — LOCKED, MASTER_PLAN Part 1.B:
        `E_0 := 0`, no gradient contribution; prediction begins at
        t = 1). Must NEVER be detached before use in the update.
        """
        raise NotImplementedError(
            "Agent B: implement E_t carrier with the LOCKED E_0 := 0 "
            "boundary condition (MASTER_PLAN Part 1.B)"
        )

    @property
    def gaze_history(self) -> Sequence[torch.Tensor]:
        """Gaze history: list of (B, 2) coordinate tensors, length <= T.

        Part of `A_t`. Carries NO gradient.
        """
        raise NotImplementedError(
            "Agent B: implement A_t gaze-history record per Part 1.A"
        )

    @property
    def current_glimpse_idx(self) -> int:
        """Current glimpse index `t` in [0, T). Part of `A_t`."""
        raise NotImplementedError(
            "Agent B: implement A_t glimpse index per Part 1.A"
        )

    def as_tensor(self) -> torch.Tensor:
        """Pooled content readout, shape (B, D_z).

        MUST have an explicit branch for `S_t is None` (the core-build
        default), tested directly.
        """
        raise NotImplementedError(
            "Agent B: implement as_tensor with an explicit S_t-is-None "
            "branch per Part 1.A"
        )

    def drift_to(self, previous: "BeliefState") -> torch.Tensor:
        """Belief drift metric vs a previous belief, shape (B,).

        Consumer: Agent H's L_stab diagnostic (Part 1.G) — usable any
        time after this interface exists; it needs no trained model.
        MUST have an explicit branch for `S_t is None`.
        """
        raise NotImplementedError(
            "Agent B: implement drift_to with an explicit S_t-is-None "
            "branch per Part 1.A (consumed by Agent H's L_stab diagnostic)"
        )
