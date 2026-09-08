"""
Belief-level HPC — the RHAN-NX belief-space swap target (D3).
================================================================================

Replaces EdgeFeatureLevelPredictor's edge-map PIXEL target with a
BELIEF-SPACE target: predict belief_{t+1} from belief_t. This is the
LevelPredictor ABC's `feature_target` generality used for its intended
purpose for the first time — the field becomes "belief_state" instead of
"edge_map".

Motivation (E1's Lens finding): pixel reconstruction (the edge-map target)
competes for gradient budget without transferring to robustness. A
belief-space target changes WHAT gets predicted: it predicts the same
representational space the precision/gaze mechanisms already operate in
(the 512-dim belief), rather than a disjoint pixel space — potentially
avoiding that specific competition.

Gradient contract (project lesson #1): `compute_error` returns the error
tensor fully connected to the prediction path — backward reaches the
predictor's parameters AND the top-down belief input (the model wires the
target side DETACHED, exactly like EdgeFeatureLevelPredictor.extract_target
detaches the bottom-up actual). tests/test_hpc_belief_gradient_flow.py
asserts this as a HARD check.

Generation 0: the predictor's parameters are registered in the optimizer's
"hpc" group (the belief predictor is an auxiliary head like the pixel
predictor), with a pre-flight |dW| measurement (scripts/measure_group_dw.py
--group-name hpc) BEFORE the smoke — HPC starvation is the single most
likely failure mode given Stage 2 precedent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rhan_core.predictive_coding.base import LevelPredictor


class HPCBeliefLevel(LevelPredictor, nn.Module):
    """
    LevelPredictor whose target is the NEXT belief state (not pixels).

    feature_target = "belief_state".

    predict(belief_t) -> (B, proj_dim) predicted belief_{t+1}.
    compute_error(prediction, actual) -> (B,) per-sample MSE over the
    belief dimension.

    The wiring module mirrors HPCLevel1's contract but has no bottom-up
    extractor: the "bottom-up actual" IS the belief state itself, so the
    module is both the predictor and the comparator. The model delays the
    error by one foraging step (predict s_{t-1} -> compare with s_t), so the
    predictor genuinely sees belief_t and targets belief_{t+1}.
    """

    feature_target: str = "belief_state"

    def __init__(self, proj_dim: int = 512):
        super().__init__()
        self.proj_dim = int(proj_dim)
        # Small MLP predictor: belief -> predicted next belief. Fresh-init
        # (never loaded from the pixel HPC's weights — the targets differ).
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, 2 * proj_dim),
            nn.GELU(),
            nn.Linear(2 * proj_dim, proj_dim),
        )

    # ── LevelPredictor ───────────────────────────────────────────────────────
    def predict(self, top_down: torch.Tensor) -> torch.Tensor:
        """(B, D) belief_t -> (B, D) predicted belief_{t+1}."""
        return self.predictor(top_down)

    def compute_error(self, prediction: torch.Tensor,
                      bottom_up_actual: torch.Tensor) -> torch.Tensor:
        """(B,) per-sample MSE between predicted and actual next belief."""
        return (prediction - bottom_up_actual).pow(2).mean(dim=1)

    # ── Wiring helper used by the model's foraging loop ─────────────────────
    def step(self, prev_belief: torch.Tensor,
             current_belief: torch.Tensor):
        """One delayed belief-prediction step.

        Args:
            prev_belief:    belief_t (the predictor's input, ATTACHED — the
                            prediction path must backprop into it).
            current_belief: belief_{t+1} (the target, DETACHED by the caller —
                            the bottom-up actual never contributes gradients).

        Returns:
            (prediction (B, D), error (B,), error_map (B, D)):
                prediction — predicted belief_{t+1};
                error — per-sample MSE (ATTACHED, enters L_hpc);
                error_map — |prediction - target| per-dimension for
                diagnostics (collapse/explosion flags).
        """
        prediction = self.predict(prev_belief)
        error = self.compute_error(prediction, current_belief)
        error_map = (prediction - current_belief).abs()
        return prediction, error, error_map

    def __repr__(self) -> str:
        return (f"HPCBeliefLevel(feature_target='belief_state', "
                f"proj_dim={self.proj_dim})")