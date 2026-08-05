"""Predictive coding ABCs: LevelPredictor and ErrorUnit."""
from abc import ABC, abstractmethod

import torch


class LevelPredictor(ABC):
    """
    One level of a hierarchical predictive coding stack (Rao & Ballard 1999).

    Predicts the representation expected at this level given the
    belief/prediction arriving from the level above (top-down), and compares
    it against the actual bottom-up representation to produce a prediction
    error that propagates upward.

    `feature_target` documents WHAT this level predicts — pixels for the
    lowest level are acceptable, but higher levels should predict features
    (edges, shape, texture embeddings), never raw pixels, per the project's
    feature-vs-pixel-target rationale.

    Concrete implementations must be nn.Module subclasses (predict() must be
    differentiable and reachable by a backward pass so the gradient-flow
    tests in tests/test_gradient_flow.py pass).
    """

    #: str — what this level predicts: "raw_pixel", "edge_map",
    #: "orientation_map", "shape_embedding", ...
    feature_target: str

    @abstractmethod
    def predict(self, top_down: torch.Tensor) -> torch.Tensor:
        """Top-down prediction of this level's target.

        Args:
            top_down: (B, D) — belief / prediction from the level above.
        Returns:
            (B, ...) predicted target representation.
        """

    @abstractmethod
    def compute_error(self, prediction: torch.Tensor,
                      bottom_up_actual: torch.Tensor) -> torch.Tensor:
        """Prediction error magnitude for one level.

        Args:
            prediction:      (B, ...) — from `predict()`.
            bottom_up_actual: (B, ...) — ground-truth target extracted from
                the bottom-up representation.
        Returns:
            (B,) per-sample error magnitude (higher = more surprising).
        """


class ErrorUnit(ABC):
    """
    The comparator that turns a prediction vs bottom-up actual pair into a
    prediction error. `LevelPredictor.compute_error` is implemented in terms
    of an ErrorUnit; the spatial `error_map()` is what future precision /
    attention mechanisms will consume (Pillar 2 precision currently consumes
    the image-space error from ImageSpacePrecision, not this map — see
    docs/ARCHITECTURE.md for the isolation boundary).
    """

    @abstractmethod
    def compute_error(self, prediction: torch.Tensor,
                      bottom_up_actual: torch.Tensor) -> torch.Tensor:
        """(B,) per-sample error magnitude."""

    @abstractmethod
    def error_map(self, prediction: torch.Tensor,
                  bottom_up_actual: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) spatial error map for visualization / future gating."""
