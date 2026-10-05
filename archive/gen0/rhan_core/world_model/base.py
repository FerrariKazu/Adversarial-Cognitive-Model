"""World model ABC — Pillar 4, scaffold only."""
from abc import ABC, abstractmethod

import torch

from rhan_core.beliefs.base import BeliefState


class WorldModel(ABC):
    """
    PILLAR 4 — SCAFFOLD ONLY IN THIS REFACTOR. NOT TRAINED, NOT CALLED IN THE
    DEFAULT FORWARD PATH.

    Given a belief and a hypothetical action, predicts the resulting future
    observation WITHOUT executing it against the real input (Dreamer/MuZero-
    style internal rollout). `NullWorldModel` is the default: simulate()
    returns the input unchanged and logs a debug-level notice that no real
    world model is wired in. This keeps every downstream call site functional
    today.

    A future `SimulatedGazePolicy` (Pillar 2 extension) will internally roll
    out a WorldModel before committing to a fixation — this interface must
    not need to change when that's added.
    """

    @abstractmethod
    def simulate(self, belief: BeliefState, action: torch.Tensor) -> torch.Tensor:
        """Predict the future observation for `action` given `belief`.

        Args:
            belief: (B, D) current belief state.
            action: (B, A) hypothetical action.
        Returns:
            (B, ...) predicted future observation/representation.
        """
