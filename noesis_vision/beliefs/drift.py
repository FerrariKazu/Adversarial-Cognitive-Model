"""
drift_to — Agent B. The belief-drift metric Agent H's L_stab consumes.
================================================================================

Contract notes (Agent B):
  * BOTH distance types are exposed now — "l2" and "cosine",
    config-selectable at the call site. The exact distance and the exact
    weighting are Agent H's staged decision (flagged discrepancy #4);
    nothing here is chosen permanently, and no default beyond a neutral
    z=1.0/s=0.0 is baked in.
  * The S-term is a ZERO-CONTRIBUTION placeholder: S_t is None in the core
    build (Part 1.D). An S-present branch exists and raises — structured
    drift lands with S_t's re-entry (Part 2 step 7), not improvised here.
  * DIFFERENTIABILITY: returns a (B,) TENSOR differentiable w.r.t. both
    beliefs' z (Agent H will eventually put drift in a loss).

DISCREPANCY RESOLVED, FLAGGED IN HANDOFF (not silently): the contract
sketch writes `drift_to(other, weights) -> float`, but the same contract's
gradient requirement ("must be differentiable w.r.t. z") and Agent 0's ABC
docstring ("shape (B,)") both require a tensor — a python float cannot
carry gradient. This module returns the (B,) tensor; callers wanting a
python float for LOGGING take `.detach().mean().item()` on a
`detached_copy()` (never on the training-path object).
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

#: Distance types exposed for Agent H's staged decision (discrepancy #4).
DISTANCES = ("l2", "cosine")

#: Neutral component weights. "s" is the zero-contribution placeholder's
#: weight — it stays 0.0 while S_t is None; the S branch raises regardless
#: (the placeholder is not a license to weight a term that cannot exist).
DEFAULT_WEIGHTS = {"z": 1.0, "s": 0.0}


def drift_to(belief, other, weights: Optional[Mapping[str, float]] = None,
             distance: str = "l2") -> torch.Tensor:
    """Weighted per-sample drift between two beliefs, shape (B,).

    Args:
        belief: the current BeliefState (duck-typed on the ABC's members).
        other: the previous belief (its z may be detached — e.g. a frozen
            history entry; the result stays differentiable w.r.t. `belief.z`).
        weights: component weights, {"z": float, "s": float}. Default:
            z=1.0, s=0.0. Unknown components raise.
        distance: "l2" (||z - z'||_2 per sample) or "cosine"
            (1 - cosine similarity per sample). Both exposed; Agent H
            decides the permanent choice (flagged discrepancy #4).

    Returns:
        (B,) tensor, differentiable w.r.t. belief.z (and other.z when it
        carries a graph).
    """
    if distance not in DISTANCES:
        raise ValueError(
            f"distance must be one of {DISTANCES} (both are exposed for "
            f"Agent H's staged decision, discrepancy #4); got {distance!r}")

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(w)
        if unknown:
            raise ValueError(
                f"unknown drift components {sorted(unknown)} — weighted "
                f"components are {sorted(w)} (S-term is the placeholder)")
        w.update(weights)

    if belief.z.shape != other.z.shape:
        raise ValueError(
            f"z shape mismatch: {tuple(belief.z.shape)} vs "
            f"{tuple(other.z.shape)} — drift is defined between beliefs of "
            f"the same substrate (Part 1.A)")

    if distance == "l2":
        d_z = torch.norm(belief.z - other.z, p=2, dim=-1)
    else:  # cosine
        d_z = 1.0 - F.cosine_similarity(belief.z, other.z, dim=-1)

    drift = w["z"] * d_z

    # ── S-term: explicit branch (Part 1.A None-propagation rule) ────────────
    if belief.s is None and other.s is None:
        pass  # zero-contribution placeholder — the S=None core build
    else:
        raise NotImplementedError(
            "structured S-term drift lands with the S_t re-entry (Part 2 "
            "step 7); the zero-contribution placeholder covers only the "
            "S=None core build (Part 1.D) — flag for explicit reconciliation "
            "rather than improvising a fusion")

    return drift
