"""
Pytest suite for Evaluation Harness Parity.
==========================================

Tests:
1. Val transform matches training curriculum constants and structure.
2. Dataset and 1,000-image evaluation slice hash stability.
3. Checkpoint gate passes on real checkpoints within 1.0 pp.
4. Checkpoint gate raises HarnessGateError if monitor differs by >1.0 pp.
5. Perturbation budget L_inf projection strictly <= eps.
"""

import os
import sys
import pytest
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from evaluation.harness import (
    get_canonical_val_transform,
    get_canonical_slice,
    verify_checkpoint_gate,
    load_model_from_checkpoint,
    build_val_loaders,
    run_pgd_pixel_space,
    HarnessGateError,
    CANONICAL_VAL_PATH_SHA256,
    CANONICAL_TORCH_SLICE_SHA256,
    DEFAULT_EPS_PIXEL,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_ROOT = "data/imagenet100/val"


def test_transform_matches_training_code():
    tf = get_canonical_val_transform(56, normalize=True)
    assert len(tf.transforms) == 3
    assert isinstance(tf.transforms[0], T.Resize)
    assert tf.transforms[0].size == (56, 56)
    assert isinstance(tf.transforms[1], T.ToTensor)
    assert isinstance(tf.transforms[2], T.Normalize)
    assert tuple(tf.transforms[2].mean) == IMAGENET_MEAN
    assert tuple(tf.transforms[2].std) == IMAGENET_STD


def test_slice_hash_stable():
    if not os.path.exists(VAL_ROOT):
        pytest.skip(f"{VAL_ROOT} not present")

    indices, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    assert len(indices) == 1000
    assert slice_hash == CANONICAL_TORCH_SLICE_SHA256


def test_checkpoint_gate_passes():
    ckpt_path = "checkpoints/foundation_ff_56_final.pth"
    if not os.path.exists(ckpt_path) or not os.path.exists(VAL_ROOT):
        pytest.skip("Checkpoint or dataset not present")

    model, meta = load_model_from_checkpoint("ff_56", 56, ckpt_path, DEVICE, wrap_normalized=False)
    loader_5k = build_val_loaders(56, VAL_ROOT, batch_size=256, num_workers=2, raw_pixel=False)

    acc_5k, monitor_clean = verify_checkpoint_gate(model, meta, loader_5k, DEVICE, tol_pp=1.0)
    gap = abs(acc_5k - monitor_clean) * 100.0
    assert gap <= 1.0, f"Gap {gap:.2f} pp exceeded tolerance"


def test_checkpoint_gate_fails_on_corrupt_monitor():
    ckpt_path = "checkpoints/foundation_ff_56_final.pth"
    if not os.path.exists(ckpt_path) or not os.path.exists(VAL_ROOT):
        pytest.skip("Checkpoint or dataset not present")

    model, meta = load_model_from_checkpoint("ff_56", 56, ckpt_path, DEVICE, wrap_normalized=False)
    corrupted_meta = dict(meta)
    corrupted_meta["val_clean_acc"] = 0.99  # fake 99% monitor

    loader_5k = build_val_loaders(56, VAL_ROOT, batch_size=256, num_workers=2, raw_pixel=False)
    with pytest.raises(HarnessGateError):
        verify_checkpoint_gate(model, corrupted_meta, loader_5k, DEVICE, tol_pp=1.0)


def test_pgd_linfty_within_eps():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return (x * 1.0).flatten(1)[:, :100]

    dummy_model = DummyModel()
    dummy_x = torch.rand(4, 3, 56, 56, device=DEVICE)
    dummy_y = torch.randint(0, 100, (4,), device=DEVICE)

    x_adv = run_pgd_pixel_space(dummy_model, dummy_x, dummy_y, eps=DEFAULT_EPS_PIXEL, steps=10)
    realized_linf = (x_adv - dummy_x).abs().max().item()
    assert realized_linf <= DEFAULT_EPS_PIXEL + 1e-5
