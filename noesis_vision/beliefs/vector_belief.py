"""
VectorBeliefState — Agent B. The concrete S=None belief carrier.
================================================================================

This IS the core-build case: `S_t = None` for the entire Gen-1 core system
and its first ablation matrix (MASTER_PLAN Part 1.D). Every member has an
EXPLICIT branch for `S_t is None`, tested directly (Part 1.A's
None-propagation rule — "not a hypothetical: the actual default
configuration").

Shapes are Part 1.A VERBATIM:
    z_t   (B, D_z)                       global content; gradients ALWAYS
    S_t   (B, K, D_s) or None            gradients only when not None
    U_t   Dirichlet evidence e_t (B, C)  non-negative (softplus);
          alpha_t = e_t + 1; uncertainty = C / sum(alpha_t)
    E_t   (B, N, D_feat)                 token-feature space (predictor
          contract, Part 1.B); NEVER detached before the update; at t = 0
          the LOCKED convention applies: E_0 := 0 (zero tensor, no gradient
          contribution — enforced in __init__, not left to caller discipline)
    A_t   gaze_history: list of (B, 2), length <= T; current_glimpse_idx: int
          NO gradient (a coordinate record; differentiability lives in the
          policy that produced it)

PLACEHOLDER POLICY (Agent B contract, now COMPLETE): the concrete U and A
dataclasses belonged to Agent D (DirichletParams) and Agent F (GazeState).
BOTH placeholders were DELETED by Agent J2's integration (the flagged
Agent-J checklist task): the CANONICAL definitions —
noesis_vision.uncertainty.evidential_head.DirichletParams (Agent D) and
noesis_vision.gaze.gaze_state.GazeState (Agent F) — are imported here and
re-exported under this module's names, so the public import surface
(`from noesis_vision.beliefs.vector_belief import DirichletParams,
GazeState`) is unchanged and now resolves to the canonical identities.
The two definitions no longer coexist anywhere (non-improvisation rule).
"""
from __future__ import annotations

from typing import List, Optional

import torch

from noesis_vision.beliefs.drift import drift_to as _module_drift_to
from noesis_vision.beliefs.interfaces import BeliefState

# Supersession COMPLETE (Agent J2 integration): no import cycles — both
# canonical modules import nothing from beliefs. The names below ARE the
# canonical classes (re-exported for backward-compatible imports).
from noesis_vision.gaze.gaze_state import GazeState  # noqa: E402,F401
from noesis_vision.uncertainty.evidential_head import (  # noqa: E402,F401
    DirichletParams)


class VectorBeliefState(BeliefState):
    """B_t = (z_t, S_t, U_t, E_t, A_t) with S_t = None — the core build.

    Field order mirrors Part 1.A's B_t; S_t defaults to None (it is a
    trailing __init__ argument only because Python requires defaults last).

    Construct via `noesis_vision.beliefs.factory.populate_belief` (same
    validation, named boundary). Construction enforces the LOCKED t=0
    boundary condition (MASTER_PLAN Part 1.B): E_0 := 0.
    """

    def __init__(self, z: torch.Tensor, U: DirichletParams,
                 E: torch.Tensor, A: GazeState,
                 S: Optional[torch.Tensor] = None):
        # ── shape validation (Part 1.A verbatim) ────────────────────────────
        if not torch.is_tensor(z) or z.dim() != 2:
            got = tuple(z.shape) if torch.is_tensor(z) else type(z).__name__
            raise ValueError(f"z must be (B, D_z); got {got}")
        B = z.shape[0]

        ev = getattr(U, "evidence", None)
        if not torch.is_tensor(ev) or ev.dim() != 2 or ev.shape[0] != B:
            got = tuple(ev.shape) if torch.is_tensor(ev) else type(ev).__name__
            raise ValueError(
                f"U.evidence must be (B, C) with z's batch ({B}); got {got}")
        if bool((ev < 0).any()):
            raise ValueError(
                "U.evidence must be non-negative (softplus output, Part 1.A)")

        if not torch.is_tensor(E) or E.dim() != 3 or E.shape[0] != B:
            got = tuple(E.shape) if torch.is_tensor(E) else type(E).__name__
            raise ValueError(
                f"E must be (B, N, D_feat) with z's batch ({B}) — the shared "
                f"predictor's output space (Part 1.B); got {got}")

        if not isinstance(A, GazeState):
            raise ValueError(
                "A must be Agent F's canonical GazeState "
                "(noesis_vision.gaze.gaze_state.GazeState — the placeholder "
                "was superseded; Agent J2 integration)")
        for g in A.gaze_history:
            if (not torch.is_tensor(g) or g.dim() != 2
                    or g.shape[0] != B or g.shape[1] != 2):
                got = tuple(g.shape) if torch.is_tensor(g) else type(g).__name__
                raise ValueError(
                    f"each gaze_history entry must be (B, 2) with z's batch "
                    f"({B}); got {got}")
        if not isinstance(A.current_glimpse_idx, int) or A.current_glimpse_idx < 0:
            raise ValueError(
                f"current_glimpse_idx must be a non-negative int; got "
                f"{A.current_glimpse_idx!r}")

        if S is not None:
            if not torch.is_tensor(S) or S.dim() != 3 or S.shape[0] != B:
                got = tuple(S.shape) if torch.is_tensor(S) else type(S).__name__
                raise ValueError(
                    f"S must be (B, K, D_s) or None (Part 1.A); got {got}")

        # ── LOCKED first-glimpse boundary condition (MASTER_PLAN Part 1.B) ──
        # E_0 := 0 — zero tensor, no gradient contribution; prediction and
        # genuine E_t begin at t = 1. Enforced structurally so no consumer
        # can inherit a different first-glimpse convention.
        if A.current_glimpse_idx == 0 and not torch.all(E == 0):
            raise ValueError(
                "E_0 := 0 is LOCKED (MASTER_PLAN Part 1.B): at t = 0 there is "
                "no predecessor belief to predict from — pass a zero tensor "
                "of the predictor's output shape. Agents must not choose "
                "their own first-glimpse convention.")

        self._z = z
        self._U = U
        self._E = E
        self._A = A
        self._S = S

    # ── Part 1.A members (ABC) ──────────────────────────────────────────────
    @property
    def z(self) -> torch.Tensor:
        """Global content z_t, (B, D_z). Carries gradients ALWAYS."""
        return self._z

    @property
    def s(self) -> Optional[torch.Tensor]:
        """S_t: None in the core build (Part 1.D). Gradients only if not None."""
        return self._S

    @property
    def evidence(self) -> torch.Tensor:
        """Dirichlet evidence e_t, (B, C), non-negative."""
        return self._U.evidence

    @property
    def alpha(self) -> torch.Tensor:
        """alpha_t = e_t + 1, (B, C) (Part 1.A)."""
        return self._U.alpha

    @property
    def uncertainty(self) -> torch.Tensor:
        """C / sum(alpha_t), shape (B,) (Part 1.A)."""
        return self._U.uncertainty

    @property
    def prediction_error(self) -> torch.Tensor:
        """E_t in token-feature space. Zero tensor at t = 0 (LOCKED).

        Never detached here — the update consumes the non-detached path
        (Part 1.A gradient rules; detaching is the Gen-0 bug class).
        """
        return self._E

    @property
    def gaze_history(self) -> List[torch.Tensor]:
        """A_t's gaze history: list of (B, 2), length <= T. NO gradient."""
        return self._A.gaze_history

    @property
    def current_glimpse_idx(self) -> int:
        """A_t's current glimpse index t in [0, T)."""
        return self._A.current_glimpse_idx

    def as_tensor(self) -> torch.Tensor:
        """Pooled content readout, (B, D_z).

        Explicit S_t-is-None branch (Part 1.A): for the vector belief the
        readout IS z. The S-present branch is deliberately NOT invented
        here — the structured readout lands with S_t's re-entry (Part 2
        step 7, Agent G's territory); no silent fusion.
        """
        if self._S is None:
            return self._z
        raise NotImplementedError(
            "structured S_t-present readout lands with the S_t re-entry "
            "(Part 2 step 7 — Agent G's territory); VectorBeliefState covers "
            "the S=None core build only (Part 1.D) — flag for explicit "
            "reconciliation rather than improvising a fusion")

    def drift_to(self, previous: "BeliefState") -> torch.Tensor:
        """Per-sample belief drift, (B,), differentiable — see beliefs.drift."""
        return _module_drift_to(self, previous)

    def detached_copy(self) -> "VectorBeliefState":
        """A graph-free, STORAGE-INDEPENDENT copy for LOGGING and diagnostics.

        Exists so diagnostic code can never accidentally detach the
        TRAINING-path object (Agent B contract): the live belief keeps its
        graph; the copy's z / E / evidence / S are detached AND cloned.
        The clone matters: bare .detach() shares the underlying storage
        with the live tensor, so an in-place write through the copy would
        silently corrupt the live object's values (caught by
        test_mutating_copy_never_touches_live_values). Mutating or
        backpropagating through the copy can never touch the live
        object's values or autograd graph.
        """
        A_copy = GazeState(
            gaze_history=[g.detach().clone() for g in self._A.gaze_history],
            current_glimpse_idx=self._A.current_glimpse_idx,
        )
        U_copy = DirichletParams(evidence=self._U.evidence.detach().clone())
        return VectorBeliefState(
            z=self._z.detach().clone(),
            U=U_copy,
            E=self._E.detach().clone(),
            A=A_copy,
            S=None if self._S is None else self._S.detach().clone(),
        )
