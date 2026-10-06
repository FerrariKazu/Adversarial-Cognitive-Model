"""
GazeState — Agent F. The CANONICAL A_t carrier (Part 1.A).
================================================================================

Supersedes Agent B's placeholder GazeState in
noesis_vision/beliefs/vector_belief.py, exactly as Agent D's DirichletParams
superseded B's U placeholder (same pattern, no silent coexistence):

  * B's placeholder carries a SUPERSEDE marker pointing HERE;
  * the deletion/re-import into beliefs/ is an explicit Agent J1
    integration step (VectorBeliefState.__init__ isinstance-checks a
    GazeState, so beliefs continue to accept THIS class unchanged —
    field names and shapes are identical by design);
  * the two definitions must not coexist past integration (Agent J
    checklist; non-improvisation rule).

What "canonical" adds over the placeholder (Part 1.A + Part 1.E):
  * an explicit capacity rule: history length is capped at T
    (num_glimpses); appending at capacity raises LOUDLY instead of
    silently truncating (the Gen-0 silent-state-loss failure class);
  * guarded advance/reset of current_glimpse_idx;
  * the E_0:=0 convention is untouched here — it lives in the belief
    construction (VectorBeliefState), which validates the pair.

A_t is a coordinate RECORD: every stored tensor is detached+cloned at
the boundary (no autograd graph may leak into the record), while
differentiability lives in the policy's soft selection (ais_v2_policy).
Shapes: history entries are (B, 2); current_glimpse_idx is an int in
[0, T).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

#: LOCKED glimpse count for the Gen-1 core (Part 1.C; schema num_glimpses).
GAZE_HISTORY_MAX_T = 4


@dataclass
class GazeState:
    """The canonical A_t record: gaze history + current glimpse index.

    Args:
        gaze_history: list of (B, 2) coordinate tensors, length <= T.
            Stored detached (A_t carries no gradient, Part 1.A).
        current_glimpse_idx: int t in [0, T).
    """

    gaze_history: List[torch.Tensor] = field(default_factory=list)
    current_glimpse_idx: int = 0
    max_history: int = GAZE_HISTORY_MAX_T

    def __post_init__(self) -> None:
        if not isinstance(self.current_glimpse_idx, int) \
                or isinstance(self.current_glimpse_idx, bool) \
                or self.current_glimpse_idx < 0:
            raise ValueError(
                f"current_glimpse_idx must be a non-negative int; got "
                f"{self.current_glimpse_idx!r}")
        if self.current_glimpse_idx >= self.max_history:
            raise ValueError(
                f"current_glimpse_idx {self.current_glimpse_idx} out of "
                f"[0, T={self.max_history})")
        for g in self.gaze_history:
            _validate_history_entry(g, self.max_history)
        # A_t is a RECORD: normalize at the boundary so even a directly
        # constructed GazeState can never carry an autograd graph.
        self.gaze_history = [
            _validate_history_entry(g, self.max_history, store=True)
            for g in self.gaze_history
        ]

    @property
    def current(self) -> torch.Tensor:
        """The most recent (B, 2) fixation. Raises at t = 0 (nothing
        recorded yet) rather than inventing a default location — the
        first fixation is the caller's decision, not a silent constant."""
        if not self.gaze_history:
            raise IndexError(
                "no gaze recorded yet (t = 0): the first fixation is the "
                "caller's decision — no default location is invented here")
        return self.gaze_history[-1]

    def record(self, location: torch.Tensor) -> "GazeState":
        """Append a (B, 2) fixation; returns self for chaining.

        Raises LOUDLY at capacity (len == T): the caller has overrun the
        LOCKED glimpse count — silent truncation would falsify every
        downstream trajectory record.
        """
        entry = _validate_history_entry(location, self.max_history,
                                        store=True)
        if len(self.gaze_history) >= self.max_history:
            raise ValueError(
                f"gaze_history at capacity (T={self.max_history}, "
                f"LOCKED num_glimpses): refusing to append — extend the "
                f"history limit via a plan ruling, not silent truncation")
        self.gaze_history.append(entry)
        return self

    def advance(self) -> "GazeState":
        """t -> t + 1; raises if that would leave [0, T)."""
        nxt = self.current_glimpse_idx + 1
        if nxt >= self.max_history:
            raise ValueError(
                f"advance() would move current_glimpse_idx to {nxt}, "
                f"outside [0, T={self.max_history})")
        self.current_glimpse_idx = nxt
        return self

    def reset(self) -> "GazeState":
        """New image: clear history and return to t = 0."""
        self.gaze_history.clear()
        self.current_glimpse_idx = 0
        return self

    def __repr__(self) -> str:
        return (f"GazeState(t={self.current_glimpse_idx}, "
                f"len(history)={len(self.gaze_history)}, "
                f"T={self.max_history}) — canonical (Agent F; supersedes "
                f"the beliefs.vector_belief placeholder)")


def _validate_history_entry(location: torch.Tensor, max_history: int,
                            store: bool = False) -> torch.Tensor:
    """Shape/range guard shared by __post_init__ and record()."""
    if not torch.is_tensor(location) or location.dim() != 2 \
            or location.shape[1] != 2:
        got = (tuple(location.shape) if torch.is_tensor(location)
               else type(location).__name__)
        raise ValueError(f"gaze history entries must be (B, 2); got {got}")
    if bool((location.abs() > 1.0 + 1e-6).any()):
        raise ValueError(
            f"gaze coordinates must lie in [-1, +1] (the A_t convention, "
            f"Part 1.A); got min/max "
            f"{location.min().item():.3f}/{location.max().item():.3f}")
    return location.detach().clone() if store else location
