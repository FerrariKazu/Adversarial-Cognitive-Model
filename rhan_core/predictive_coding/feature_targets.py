"""
Feature-target extractors for Hierarchical Predictive Coding (Pillar 1).

These extract the *bottom-up actual* representations that each LevelPredictor
must predict. Non-learnable (fixed optical/visual properties), so they are
pure target generators and never contribute gradients.

Level convention (documented in docs/ARCHITECTURE.md):
    Level 0 (edge_map)        — implemented in this pass (Stage 2).
    Level 1 (orientation_map) — extractor implemented; predictor wiring
                                 deferred until Level 0 is validated.
    Level 2+ (shape embedding) — scaffold only; raises on use.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SobelKernels:
    """Fixed 3x3 Sobel kernels for horizontal/vertical gradients."""

    @staticmethod
    def build() -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).clone()
        return sobel_x, sobel_y


class EdgeMapExtractor(nn.Module):
    """
    Non-learnable Sobel edge-magnitude extractor.

    Input:  (B, C, H, W) normalized image tensor (any C; gradients are
            computed per channel and collapsed by max across channels).
    Output: (B, 1, H, W) edge magnitude in [0, 1] (per-batch max-normalized).
    """

    def __init__(self):
        super().__init__()
        sobel_x, sobel_y = _SobelKernels.build()
        # Buffers (not params) so .to(device) / .cuda() just works.
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, 1, H, W) edge magnitude, [0, 1].

        DETACHED output (2026-08-15): these extractors are "pure target
        generators and never contribute gradients" (module contract) — but
        nothing enforced it, and in the full-model wiring the crop x_foveal
        is differentiable back through grid_sample into the gaze action ->
        backbone. The Sobel magnitude's sqrt backward is inf at flat patches
        (gx^2+gy^2 = 0 -> 1/(2*sqrt(x)) -> inf), so inf*0 = NaN poisoned the
        backbone's gradients on every flat region of real images. The NaN
        grads collapsed the trainer's GradScaler to scale 0, silently
        disabling optimizer.step() for ALL params across a whole 15-epoch
        smoke (ratio-1.00 freeze — the 2026-08-13 "starvation" smoke never
        trained anything). Detaching the output enforces the documented
        contract at the boundary where it is declared.
        """
        C = x.shape[1]
        # Depthwise Sobel (kernel expanded to (C, 1, 3, 3)) so any channel
        # count works; magnitude collapses channels by max.
        gx = F.conv2d(x, self.sobel_x.expand(C, 1, 3, 3), padding=1, groups=C)
        gy = F.conv2d(x, self.sobel_y.expand(C, 1, 3, 3), padding=1, groups=C)
        mag = (gx ** 2 + gy ** 2).sqrt()                    # (B, C, H, W)
        mag = mag.max(dim=1, keepdim=True).values           # (B, 1, H, W)
        peak = mag.amax(dim=(2, 3), keepdim=True) + 1e-8
        return (mag / peak).clamp(0.0, 1.0).detach()


class OrientationMapExtractor(nn.Module):
    """
    Non-learnable orientation extractor (Pillar 1, Level 1 target).

    Input:  (B, C, H, W)
    Output: (B, 2, H, W) — [sin(theta), cos(theta)] of the dominant gradient
            orientation per pixel, in [-1, 1]. Sin/cos encoding avoids the
            angle-wrap discontinuity (0 == 2*pi) that a raw atan2 map has.
    """

    def __init__(self):
        super().__init__()
        sobel_x, sobel_y = _SobelKernels.build()
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, 2, H, W) sin/cos orientation in [-1, 1].

        DETACHED output (2026-08-15): same contract as EdgeMapExtractor — the
        extractor is a pure target generator and never contributes gradients
        (see EdgeMapExtractor.forward for the NaN-poisoning mechanism this
        prevents).
        """
        C = x.shape[1]
        gx = F.conv2d(x, self.sobel_x.expand(C, 1, 3, 3), padding=1, groups=C)
        gy = F.conv2d(x, self.sobel_y.expand(C, 1, 3, 3), padding=1, groups=C)
        theta = torch.atan2(gy, gx)                         # (B, C, H, W)
        # Collapse channels by the orientation of the strongest gradient.
        mag = (gx ** 2 + gy ** 2).sqrt().max(dim=1, keepdim=True).values
        # Use the mean angle per spatial location across channels.
        theta = theta.mean(dim=1, keepdim=True)
        return (torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)
                * (mag > 1e-6)).detach()


class ShapeEmbeddingExtractor(nn.Module):
    """
    PILLAR 1, Level 2+ — SCAFFOLD ONLY.

    A learned shape/texture embedding extractor is planned for higher HPC
    levels. Importing and instantiating this class is safe; CALLING it raises
    a clear NotImplementedError. Do not wire it into HierarchicalPredictiveStack
    until Level 1 (orientation) has been validated.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "ShapeEmbeddingExtractor is scaffold-only (HPC Level 2+). It must "
            "not be called until Levels 0-1 are validated. See "
            "docs/rhan_next_roadmap.json stage 2.")
