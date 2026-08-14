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

    Target contract (2026-08-12 dead-head fix): the raw EdgeMapExtractor
    output is Sobel magnitude in [0, 1]; extract_target() rescales it to
    [-1, 1] (2*t - 1) so MSE is computed against a target spanning the SAME
    range as the Tanh-bounded head, and the final ConvTranspose2d is
    small-inited (zero bias, uniform +-0.01 weights) so the head starts just
    inside the linear regime (~0 output, Tanh slope > 0.5 everywhere). The
    Stage 2 smoke showed the un-fixed head pinned at Tanh saturation with
    the prediction error frozen at its init value across all logged epochs.
    """

    feature_target: str = "edge_map"

    #: Affine applied to the [0, 1] Sobel magnitude to match the Tanh head's
    #: [-1, 1] output range: target' = TARGET_SCALE * target + TARGET_SHIFT.
    TARGET_SHIFT, TARGET_SCALE = -1.0, 2.0

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
        # Dead-head fix (2026-08-12 Stage 2 smoke): the default kaiming init
        # of the output conv produced large pre-Tanh activations, pinning the
        # head at the Tanh saturation extremes where gradients vanish — the
        # smoke's hpc_error_mean stayed frozen at its init value through all
        # 10 logged epochs (including 5 main-phase, all-components epochs).
        # Fix: small-init the last layer (zero bias + tiny uniform weights)
        # so the head starts just inside the linear regime and learns the
        # mean target first, then the structure. NOTE: a hard zero-init of
        # the weight was rejected — the gradient wrt the INPUT of a
        # zero-weight layer is 0, so it would wash out ALL upstream
        # gradients on the first backward (breaking the gradient-flow tests
        # and wasting the first optimizer step). Small-but-nonzero keeps
        # every parameter reachable from step 1.
        _head = self.decoder[-2]
        assert isinstance(_head, nn.ConvTranspose2d), \
            "decoder[-2] must be the output ConvTranspose2d (before Tanh)"
        if _head.bias is not None:
            nn.init.zeros_(_head.bias)
        nn.init.uniform_(_head.weight, -0.01, 0.01)

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
        """(B, 3, spatial, spatial) crop -> (B, 1, spatial, spatial) target.

        The raw EdgeMapExtractor output is Sobel magnitude in [0, 1]; it is
        rescaled here to [-1, 1] (2*t - 1) so MSE is computed against a
        target spanning the same range as the Tanh-bounded head — part of the
        2026-08-12 dead-head fix (a [0, 1] target against a [-1, 1] head
        forced the network to learn an offset through a saturating
        activation; the smoke's prediction error froze at its init value).

        DETACHED (2026-08-15): the target is a bottom-up *actual* produced by
        the non-learnable EdgeMapExtractor, so it must never contribute
        gradients (feature_targets.py module contract: "pure target
        generators and never contribute gradients"). In the full-model wiring
        the crop x_foveal is differentiable back through grid_sample into the
        gaze action -> backbone, and the Sobel magnitude's sqrt backward is
        inf at flat patches (gx^2+gy^2 = 0 -> 1/(2*sqrt(x)) -> inf), so
        inf*0 = NaN poisoned the backbone's gradients at every flat region of
        real images. The NaN backbone grads collapsed the trainer's
        GradScaler to scale 0, silently disabling optimizer.step() for ALL
        params across the entire 15-epoch smoke (zero weight change — the
        observed ratio-1.00 freeze was NOT optimizer starvation; nothing
        trained at all). Detaching here keeps gradient flowing through the
        prediction path (head -> belief -> backbone) only.
        """
        t = self.extractor(x_foveal)          # (B, 1, H, W) Sobel in [0, 1]
        return (t * self.TARGET_SCALE + self.TARGET_SHIFT).detach()


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
