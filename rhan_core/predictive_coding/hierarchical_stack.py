"""
Concrete predictive-coding stack — Pillar 1.

This pass implements exactly ONE hierarchy level (Stage 2 cycle 1):

    Level 0 — EdgeFeatureLevelPredictor (feature_target = "edge_map").
        Predicts the Sobel edge map of the foveal crop from the current
        belief (top-down), and computes the error against the non-learnable
        EdgeMapExtractor output (bottom-up actual). This sits *alongside* the
        existing top-level predictor (RHANv12's precision_ctrl.prior_predictor
        and generative_prior), which already predict foveal features and raw
        pixels from the belief.

The hierarchy is deliberately belief-anchored (see docs/ARCHITECTURE.md for
the rationale): mid-transformer-layer hooking into the gradient-checkpointed
dual-stream encoder was evaluated and deferred as fragile; the belief-level
level is gradient-safe, isolated, and satisfies the feature-vs-pixel target
rule (edges, never raw pixels, above the lowest level).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rhan_core.predictive_coding.base import LevelPredictor
from rhan_core.predictive_coding.feature_targets import EdgeMapExtractor


class _Reshape(nn.Module):
    """Reshape helper for use inside nn.Sequential decoders."""

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class EdgeFeatureLevelPredictor(LevelPredictor, nn.Module):
    """
    HPC Level 0 — predicts the edge map of the foveal crop from the belief.

    feature_target = "edge_map" (a feature, never raw pixels above the
    lowest level). The target is produced by the non-learnable
    EdgeMapExtractor (Sobel), so the predictor is the only trainable piece
    of this level — which is exactly what the gradient-flow test asserts.

    Architecture (small MLP -> decoder, mirroring the GenerativePrior shape):
        Linear(512 -> 64*6*6) -> Reshape(64, 6, 6)
        ConvTranspose2d(64->32, k=4, s=2) -> 12x12
        ConvTranspose2d(32->16, k=4, s=2) -> 24x24
        ConvTranspose2d(16->1,  k=4, s=2) -> 48x48  (Tanh, [-1, 1])
    """

    feature_target: str = "edge_map"

    def __init__(self, proj_dim: int = 512, spatial: int = 48):
        super().__init__()
        self.spatial = spatial
        init_spatial = spatial // 8                      # 6 for spatial=48
        self.extractor = EdgeMapExtractor()              # non-learnable target
        self.fc = nn.Linear(proj_dim, 64 * init_spatial * init_spatial)
        self.decoder = nn.Sequential(
            _Reshape(64, init_spatial, init_spatial),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),   # -> 12x12
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),   # -> 24x24
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),                # -> 48x48
            nn.Tanh(),
        )

    # ── LevelPredictor ───────────────────────────────────────────────────────
    def predict(self, top_down: torch.Tensor) -> torch.Tensor:
        """(B, D) belief -> (B, 1, spatial, spatial) predicted edge map."""
        return self.decoder(self.fc(top_down))

    def compute_error(self, prediction: torch.Tensor,
                      bottom_up_actual: torch.Tensor) -> torch.Tensor:
        """(B,) per-sample MSE between predicted and extracted edge maps."""
        return (prediction - bottom_up_actual).pow(2).mean(dim=[1, 2, 3])

    # ── Level helpers ────────────────────────────────────────────────────────
    def extract_target(self, x_foveal: torch.Tensor) -> torch.Tensor:
        """(B, 3, spatial, spatial) crop -> (B, 1, spatial, spatial) target."""
        return self.extractor(x_foveal)


class HierarchicalPredictiveStack(nn.Module):
    """
    Composes the implemented HPC levels. Stage 2 implements exactly level 0.

    The registry maps level index -> concrete LevelPredictor class. Adding a
    level requires implementing its class and extending LEVEL_REGISTRY — then
    bumping hpc_num_levels by ONE for the next validation cycle.
    """

    LEVEL_REGISTRY = {0: EdgeFeatureLevelPredictor}

    def __init__(self, proj_dim: int = 512, num_levels: int = 1,
                 fovea_size: int = 48):
        super().__init__()
        if num_levels < 0:
            raise ValueError(f"num_levels must be >= 0, got {num_levels}")
        missing = [i for i in range(num_levels) if i not in self.LEVEL_REGISTRY]
        if missing:
            raise NotImplementedError(
                f"HPC levels {missing} are not implemented. The roadmap "
                f"requires one level per validation cycle; levels beyond "
                f"{max(self.LEVEL_REGISTRY)} must be added and validated "
                f"one at a time.")
        self.levels = nn.ModuleList([
            self.LEVEL_REGISTRY[i](proj_dim=proj_dim, spatial=fovea_size)
            for i in range(num_levels)
        ])

    # ── Level accessors ──────────────────────────────────────────────────────
    def extract_targets(self, x_foveal: torch.Tensor, level_idx: int) -> torch.Tensor:
        """Bottom-up actual for one level."""
        return self.levels[level_idx].extract_target(x_foveal)

    def predict(self, top_down: torch.Tensor, level_idx: int) -> torch.Tensor:
        """Top-down prediction for one level."""
        return self.levels[level_idx].predict(top_down)

    def compute_error(self, prediction: torch.Tensor,
                      bottom_up_actual: torch.Tensor, level_idx: int) -> torch.Tensor:
        """(B,) prediction error for one level."""
        return self.levels[level_idx].compute_error(prediction, bottom_up_actual)

    def feature_targets(self):
        """list[str] of feature_target per level, in order."""
        return [lvl.feature_target for lvl in self.levels]

    def num_levels(self) -> int:
        return len(self.levels)
