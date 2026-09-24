"""
populate_belief — Agent B's named factory for the S=None core build.
================================================================================

Thin by design: all Part 1.A shape validation and the LOCKED t=0 boundary
condition live in VectorBeliefState.__init__ (one validation site, so
direct construction is guarded identically). This factory exists so every
consumer constructs beliefs through ONE named boundary — the same
one-source-of-truth discipline Agent 0's interfaces encode.

t = 0 usage (LOCKED, MASTER_PLAN Part 1.B): pass
`E=torch.zeros(B, N, D_feat)` — E_0 := 0, the first glimpse is pure
observation. Prediction and genuine E_t begin at t = 1.
"""
from __future__ import annotations

from typing import Optional

import torch

from noesis_vision.beliefs.vector_belief import (
    DirichletParams,
    GazeState,
    VectorBeliefState,
)

__all__ = ["populate_belief", "DirichletParams", "GazeState", "VectorBeliefState"]


def populate_belief(z: torch.Tensor, U: DirichletParams, E: torch.Tensor,
                    A: GazeState,
                    S: Optional[torch.Tensor] = None) -> VectorBeliefState:
    """Construct and validate a VectorBeliefState (the S=None core build).

    Args:
        z: (B, D_z) global content — gradients always (Part 1.A).
        U: DirichletParams placeholder (SUPERSEDE with Agent D's).
        E: (B, N, D_feat) latent prediction error; the zero tensor at t = 0
            (LOCKED E_0 := 0 — enforced, not convention-by-honor-system).
        A: GazeState placeholder (SUPERSEDE with Agent F's).
        S: None in the core build (Part 1.D).

    Returns:
        VectorBeliefState with every Part 1.A member live.
    """
    return VectorBeliefState(z=z, U=U, E=E, A=A, S=S)
