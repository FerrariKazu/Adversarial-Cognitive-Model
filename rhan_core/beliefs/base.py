"""Abstract belief representation (BeliefState)."""
from abc import ABC, abstractmethod

import torch


class BeliefState(ABC):
    """
    Abstract representation of the model's current belief about the input.

    Pillars 1 & 2 use `VectorBeliefState` (dense tensor, identical in spirit
    to v12's belief vector `s`). Pillar 3 will introduce
    `StructuredBeliefState` (object slots + relational graph) without
    requiring any caller of BeliefState to change.

    Every concrete belief must expose:
      - as_tensor(): a flattened (B, D) view for legacy compatibility
        with the existing classifier head
      - uncertainty(): a (B,) scalar per-sample uncertainty measure,
        consumed by GazePolicy and PrecisionModulator
    """

    @abstractmethod
    def as_tensor(self) -> torch.Tensor:
        """Flattened (B, D) view of the belief for legacy classifier compat."""

    @abstractmethod
    def uncertainty(self) -> torch.Tensor:
        """Per-sample uncertainty in (B,), higher = less certain."""
