"""
Agent C contract test — differentiable foveation.

Gradient must reach gaze_coords THROUGH the crop (the motor Jacobian
Gen-0's foveal_sample provides), end-to-end through encode_glimpse.
Includes the contract's smoke experiment (one dummy image + one gaze
coordinate -> both output shapes) and the convention validations.
"""
import pytest
import torch

from noesis_vision.models.backbone import D_Z, CompactViT
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE, foveal_sample

B = 1  # the contract's smoke experiment: ONE image, ONE gaze coordinate


def test_gradient_reaches_gaze_coords_through_crop():
    x = torch.randn(B, 3, 96, 96)
    gaze = torch.tensor([[0.3, -0.2]], requires_grad=True)
    crop = foveal_sample(x, gaze, fovea_size=48)
    crop.sum().backward()
    assert gaze.grad is not None
    assert torch.any(gaze.grad != 0), (
        "d(crop)/d(gaze) must be nonzero — the motor Jacobian is the "
        "whole point of the differentiable crop")


def test_gradient_reaches_gaze_coords_through_encode_glimpse():
    m = CompactViT(img_size=56)
    x = torch.randn(B, 3, 96, 96)
    gaze = torch.zeros(B, 2, requires_grad=True)
    pooled, tokens = m.encode_glimpse(x, gaze)
    (pooled.sum() + tokens.sum()).backward()
    assert gaze.grad is not None and torch.any(gaze.grad != 0)


def test_image_path_also_differentiable():
    x = torch.randn(B, 3, 96, 96, requires_grad=True)
    gaze = torch.zeros(B, 2)
    foveal_sample(x, gaze, fovea_size=48).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)


def test_smoke_experiment_encode_glimpse_shapes():
    """Contract smoke: one dummy image + one gaze coordinate through
    encode_glimpse; confirm BOTH output shapes: (B, D_z) and (B, N, D_z)."""
    m = CompactViT(img_size=56)
    image = torch.randn(B, 3, 96, 96)
    gaze = torch.zeros(B, 2)  # center fixation
    pooled_z, tokens = m.encode_glimpse(image, gaze)
    assert pooled_z.shape == (B, D_Z)
    n = (DEFAULT_FOVEA_SIZE // 14) ** 2
    assert tokens.shape == (B, n, D_Z)


def test_gaze_convention_enforced():
    x = torch.randn(B, 3, 96, 96)
    with pytest.raises(ValueError, match=r"\[-1, \+1\]"):
        foveal_sample(x, torch.tensor([[1.5, 0.0]]), fovea_size=48)
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        foveal_sample(x, torch.zeros(B, 3), fovea_size=48)


def test_non_square_input_refused():
    x = torch.randn(B, 3, 96, 64)
    with pytest.raises(ValueError, match="square"):
        foveal_sample(x, torch.zeros(B, 2), fovea_size=48)


def test_border_gaze_is_valid():
    """Gaze at the frame edge is legal (border padding handles it) —
    the policy may legitimately fixate near frame borders."""
    x = torch.randn(B, 3, 96, 96)
    crop = foveal_sample(x, torch.tensor([[1.0, -1.0]]), fovea_size=48)
    assert crop.shape == (B, 3, 48, 48)
    assert torch.isfinite(crop).all()
