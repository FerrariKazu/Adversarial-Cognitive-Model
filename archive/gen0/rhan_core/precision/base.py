"""Precision modulator ABC."""
from abc import ABC, abstractmethod

import torch


class PrecisionModulator(ABC):
    """
    Computes a per-sample precision signal (Pi_D, unsupervised — NOT trained
    against a saturating binary correctness target, per the diagnosed
    v10/v11 failure) and exposes it for use by OTHER components.

    This class does not itself modulate anything; it is queried by
    GazePolicy, the recurrence loop, and the loss function separately, so
    each consumer's use of precision can be isolated and tested
    independently.
    """

    @abstractmethod
    def compute_precision(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """Per-sample precision from a prediction-error signal.

        Args:
            prediction_error: (B,) per-sample error magnitude, or (B, ...)
                              error tensor (magnitude is derived internally).
        Returns:
            (B,) precision in a bounded range (e.g. [0.2, 0.8]).
        """
