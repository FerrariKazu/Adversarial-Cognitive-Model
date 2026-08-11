"""
HPCLevel1 — the Stage 2 wiring module for Pillar 1 (Hierarchical Predictive Coding).
====================================================================================

Wires ONE additional predictive-coding level into RHANNext, alongside the
existing top-level predictor inherited from v12 (precision_ctrl.prior_predictor
+ generative_prior, which predict foveal features and raw pixels from the
belief). This level predicts an **edge_map feature target** (Sobel,
non-learnable, already built in feature_targets.py) from top-down belief and
computes the prediction error against the actual bottom-up edge map.

Level / tap-point contract (Stage 2 cycle 1 — exactly one level):

    level 0 (feature_target = "edge_map")  [the only implemented level]
        top-down      : s (B, proj_dim) — the current belief from _forage
        bottom-up tap : the FOVeal crop x_foveal (B, 3, 48, 48) sampled at the
                        CURRENT GAZE POSITION — i.e. the input to the foveal
                        stream, i.e. the output of `foveal_sample` in
                        model.RHANNext._forage. This is the ONLY bottom-up
                        input in this first pass; mid-transformer-layer
                        hooking was evaluated and DEFERRED as fragile under
                        gradient checkpointing (see docs/ARCHITECTURE.md §5.2).
                        The tap is a single layer (the foveal sampling
                        operation), never an average across multiple layers.
        predicted     : (B, 1, 48, 48) edge map decoded from the belief
        target        : EdgeMapExtractor(x_foveal) — Sobel magnitude, [0, 1]
        error         : (B,) per-sample MSE  (this is L_hpc, enters the loss)

Gradient-flow contract (project lesson #1 — the v11/v12 detached-recon bug):
`forward` returns the error tensor fully connected to the computation graph —
`error.requires_grad is True` and its backward reaches the stack's fc/decoder
parameters. tests/test_hpc_gradient_flow.py asserts this as a HARD check, not
a manual review step.

feature_target rule: "edge_map" — a FEATURE, never raw pixels above the lowest
level (the raw-pixel level remains the existing generative_prior). The
extractor consumes the raw crop only to PRODUCE the edge feature target; what
is predicted is always the feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rhan_core.predictive_coding.base import LevelPredictor
from rhan_core.predictive_coding.hierarchical_stack import (
    HierarchicalPredictiveStack,
)


class HPCLevel1(nn.Module):
    """
    Wires the HierarchicalPredictiveStack (level 0 = EdgeFeatureLevelPredictor)
    into the main forward pass at hpc_num_levels=1.

    Args:
        embed_dim: model belief/embedding dimension (768 in RHANNext) —
            recorded for interface parity; the level's predictor consumes the
            proj_dim belief actually produced by _forage.
        tap_layer: documented name of the layer whose activations feed this
            level's bottom-up input. Stage 2 pass: "foveal_crop" (the foveal
            sampling operation in _forage). See the module docstring for the
            full tap-point contract.
        proj_dim: belief dimension consumed by the predictor (512).
        fovea_size: spatial size of the foveal crop / edge-map target (48).
    """

    #: what this level predicts — a feature, never raw pixels above the lowest level
    feature_target: str = "edge_map"

    def __init__(self, embed_dim: int, tap_layer: str,
                 proj_dim: int = 512, fovea_size: int = 48):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.tap_layer = str(tap_layer)
        # Exactly ONE level in this pass (config.validate() enforces
        # hpc_num_levels <= 1; the stack guards the registry itself).
        self.stack = HierarchicalPredictiveStack(
            proj_dim=proj_dim, num_levels=1, fovea_size=fovea_size)

    # ── LevelPredictor passthroughs (single level) ────────────────────────────
    def feature_targets(self):
        """list[str] — ["edge_map"] in this pass."""
        return self.stack.feature_targets()

    @property
    def levels(self) -> nn.ModuleList:
        """Expose the stack's levels (test/API compat with the old hpc_stack)."""
        return self.stack.levels

    # ── Main wiring call ──────────────────────────────────────────────────────
    def forward(self, top_down_belief: torch.Tensor,
                bottom_up_activations: torch.Tensor):
        """
        One predictive-coding step at level 0.

        Args:
            top_down_belief:      (B, proj_dim) — current belief `s` from
                                  model._forage.
            bottom_up_activations: (B, 3, 48, 48) — the tap-point activations
                                  (Stage 2: the foveal crop x_foveal at the
                                  current gaze position).

        Returns:
            (prediction, error, error_map):
                prediction: (B, 1, 48, 48) predicted edge map (Tanh, [-1, 1])
                error:      (B,) per-sample MSE — NOT detached; enters L_hpc
                error_map:  (B, 1, 48, 48) |prediction - target| spatial error
                            map for diagnostics (collapse/explosion flags).
        """
        prediction = self.stack.predict(top_down_belief, 0)
        target = self.stack.extract_targets(bottom_up_activations, 0)
        error = self.stack.compute_error(prediction, target, 0)   # (B,)
        error_map = (prediction - target).abs()                   # (B, 1, H, W)
        return prediction, error, error_map
