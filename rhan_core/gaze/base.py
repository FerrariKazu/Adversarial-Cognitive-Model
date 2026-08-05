"""Gaze policy ABC."""
from abc import ABC, abstractmethod

import torch

from rhan_core.beliefs.base import BeliefState


class GazePolicy(ABC):
    """
    Decides where to look next (select_action) and whether enough evidence
    has been gathered to stop iterating (should_halt).

    Pillar 2's `InformationGainGazePolicy` selects actions that maximize an
    EXPECTED REDUCTION IN BELIEF UNCERTAINTY (a tractable proxy for expected
    information gain — exact mutual information I(z_future; a) is intractable;
    the exact proxy used is documented in that class's docstring).

    Pillar 4's future `SimulatedGazePolicy` will internally roll out a
    WorldModel before committing to a fixation — this interface must not need
    to change when that's added.
    """

    @abstractmethod
    def select_action(self, belief: BeliefState,
                      history: list) -> torch.Tensor:
        """Choose the next gaze action.

        Args:
            belief:  current belief state (uncertainty-aware).
            history: list of per-step context dicts pushed by the model.
        Returns:
            (B, 2) normalized gaze coordinates in [-1, +1].
        """

    @abstractmethod
    def should_halt(self, belief: BeliefState,
                    history: list) -> torch.Tensor:
        """Whether enough evidence has been gathered.

        Args:
            belief:  current belief state.
            history: list of per-step context dicts pushed by the model.
        Returns:
            (B,) boolean tensor — True = halt (stop gathering evidence).
        """
