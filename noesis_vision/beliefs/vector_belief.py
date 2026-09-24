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

PLACEHOLDER POLICY (Agent B contract): the concrete U and A dataclasses
belong to Agent D (DirichletParams, EvidentialHead port) and Agent F
(GazeState, AIS-v2 gaze bookkeeping). Until they land, minimal placeholder
shapes are defined HERE AND ONLY HERE, marked SUPERSEDE. Two definitions
must not coexist past integration: when Agent D's / Agent F's versions
arrive, delete the placeholders and import theirs — if their shapes are
incompatible with this file, STOP and flag (do not silently coerce).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from noesis_vision.beliefs.drift import drift_to as _module_drift_to
from noesis_vision.beliefs.interfaces import BeliefState

# Agent F supersession (IN FLIGHT): the canonical GazeState now lives in
# noesis_vision.gaze.gaze_state (same SUPERSEDE pattern as Agent D's
# DirichletParams). Field names and shapes are identical by design, so
# until Agent J1 performs the flagged deletion/re-import, BOTH classes are
# accepted at this boundary — no silent coercion either way (entries are
# re-validated below regardless of type). No import cycle: gaze_state
# imports nothing from beliefs.
from noesis_vision.gaze.gaze_state import GazeState as _CanonicalGazeState


@dataclass
class DirichletParams:  # PLACEHOLDER — SUPERSEDED by noesis_vision.uncertainty.evidential_head.DirichletParams (Agent D, canonical since its landing). INTEGRATION TASK (Agent J checklist, Agent D contract): DELETE this placeholder and import the canonical one here and in factory.py — the two definitions must not coexist past integration.
    """Placeholder U-carrier — Part 1.A's Dirichlet evidence, minimal shape.

    Agent D's CANONICAL DirichletParams now exists
    (noesis_vision/uncertainty/evidential_head.py) with identical field
    name (`evidence`) and identical alpha/uncertainty formulas; this
    placeholder remains only so this module imports nothing circular
    until Agent J performs the flagged deletion/re-import.
    """

    evidence: torch.Tensor  # (B, C), non-negative (softplus output) — Part 1.A

    @property
    def alpha(self) -> torch.Tensor:
        """alpha_t = e_t + 1 (Part 1.A), shape (B, C)."""
        return self.evidence + 1.0

    @property
    def uncertainty(self) -> torch.Tensor:
        """Uncertainty scalar: C / sum(alpha_t) (Part 1.A), shape (B,)."""
        return self.evidence.shape[-1] / self.alpha.sum(dim=-1)


@dataclass
class GazeState:  # PLACEHOLDER — SUPERSEDE with Agent F's definition (AIS-v2 gaze bookkeeping, Part 1.E) when it lands; same deletion/re-import integration task for Agent J.
    """Placeholder A-carrier — Part 1.A's gaze record, minimal shape.

    Agent F's real GazeState supersedes this at integration (same rule
    as the DirichletParams placeholder above).
    """

    gaze_history: List[torch.Tensor] = field(default_factory=list)
    # list of (B, 2) coordinate tensors, length <= T; NO gradient
    current_glimpse_idx: int = 0


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

        if not isinstance(A, (GazeState, _CanonicalGazeState)):
            raise ValueError(
                "A must be a GazeState (the beliefs placeholder until "
                "Agent J1's supersession, or Agent F's canonical "
                "noesis_vision.gaze.gaze_state.GazeState — the two are "
                "shape-identical by design)")
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
