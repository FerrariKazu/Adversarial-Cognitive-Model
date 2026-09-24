"""
Differentiable foveal crop — Agent C. ADAPTED from the STL-10 project.
================================================================================

Source: `foveal_sample` in phase1_training/model_rhan_v10.py (Part 5 port
table disposition: ADAPT — "same concept, new input resolution"). The
validated MECHANISM is carried unchanged:

  * theta is built DIFFERENTIABLY from the gaze coordinates (torch.cat /
    stack on tensors that carry gaze's autograd graph), so d(output)/d(gaze)
    flows through F.grid_sample — this IS the motor Jacobian
    df_stem(a)/da the update and gaze-selection consume;
  * bilinear sampling, border padding, align_corners=False (the exact
    sampling conventions Gen-0 validated);
  * gaze coordinates are normalized to [-1, +1], origin center.

Adaptations (flagged in the Agent C handoff):
  * input size is a parameter (Gen-0 hardcoded the 96 px STL-10 frame);
  * fovea_size default is 56, not the STL-10 convention's 48: 48 is not
    divisible by the DINOv2-small patch size 14, and a clean patch-embed
    mapping is an explicit contract requirement. 56 = 4 x 14 -> a 4x4
    patch grid per crop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: 56 = 4 x 14 (DINOv2-small patch): a clean patch grid per foveal crop.
#: The STL-10 project used 48 — not divisible by 14; deviation flagged.
DEFAULT_FOVEA_SIZE = 56


def foveal_sample(x_image: torch.Tensor, gaze_coords: torch.Tensor,
                  fovea_size: int = DEFAULT_FOVEA_SIZE) -> torch.Tensor:
    """Differentiable foveal crop at the gaze position.

    Args:
        x_image: (B, 3, H, W) full input image (square).
        gaze_coords: (B, 2) normalized gaze in [-1, +1] (x, y), origin
            center — the convention the gaze policy writes (Part 1.A's
            A_t record; gradient flows THROUGH this argument).
        fovea_size: crop edge length in pixels; must equal
            k * patch_size for the backbone's patch embedding (default 56
            = 4 x 14).

    Returns:
        (B, 3, fovea_size, fovea_size) crop. Differentiable w.r.t. BOTH
        x_image and gaze_coords (grid_sample's sampling Jacobian).
    """
    if x_image.dim() != 4 or x_image.shape[1] != 3:
        got = tuple(x_image.shape)
        raise ValueError(f"x_image must be (B, 3, H, W); got {got}")
    H = x_image.shape[-1]
    if x_image.shape[-2] != H:
        raise ValueError(
            f"x_image must be square (foveation assumes it); got "
            f"{tuple(x_image.shape)}")
    if gaze_coords.dim() != 2 or gaze_coords.shape[1] != 2 \
            or gaze_coords.shape[0] != x_image.shape[0]:
        got = tuple(gaze_coords.shape)
        raise ValueError(
            f"gaze_coords must be (B, 2) matching x_image's batch; got {got}")
    if bool((gaze_coords.abs() > 1.0 + 1e-6).any()):
        raise ValueError(
            f"gaze_coords must lie in [-1, +1] (normalized, origin center "
            f"— the A_t convention, Part 1.A); got min/max "
            f"{gaze_coords.min().item():.3f}/{gaze_coords.max().item():.3f} "
            f"— a convention mismatch here corrupts every downstream shape")
    if fovea_size > H:
        raise ValueError(
            f"fovea_size {fovea_size} exceeds input {H}")

    B = x_image.shape[0]
    scale = fovea_size / float(H)
    device, dtype = x_image.device, x_image.dtype

    # Differentiable theta construction (carried from the source verbatim):
    # autograd tracks the dependency on gaze_coords because the translation
    # columns ARE gaze tensors, not python floats.
    scale_col = torch.full((B, 1), scale, device=device, dtype=dtype)
    zero_col = torch.zeros((B, 1), device=device, dtype=dtype)

    row0 = torch.cat([scale_col, zero_col, gaze_coords[:, 0:1]], dim=1)
    row1 = torch.cat([zero_col, scale_col, gaze_coords[:, 1:2]], dim=1)
    theta = torch.stack([row0, row1], dim=1)  # (B, 2, 3)

    grid = F.affine_grid(theta, (B, 3, fovea_size, fovea_size),
                         align_corners=False)
    return F.grid_sample(x_image, grid,
                         mode="bilinear",
                         padding_mode="border",
                         align_corners=False)
