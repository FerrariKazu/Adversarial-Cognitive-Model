"""Unit tests for Stage S3 dense window tuning script."""
import pathlib
import sys
import tempfile
import torch
import torch.nn as nn
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2
from scripts.rhan_im.train_s3_dense_window import (
    load_trunk_for_s3,
    partition_train_split,
    EXPECTED_TRUNK_TRAINABLE_PARAMS,
)
from scripts.rhan_im.gate1_measure_dprime import load_trunk


def test_s3_trainable_parameter_slice():
    """Confirms blocks[9:12] and norm are trainable, blocks 0..8 are frozen, parameter count matches."""
    trunk = CompactViTGen2(img_size=112)
    # Freeze trunk
    trunk.eval().requires_grad_(False)
    # Unfreeze blocks[9:12] and norm
    for b in trunk.blocks[9:12]:
        b.train().requires_grad_(True)
    trunk.norm.train().requires_grad_(True)

    trainable_params = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    assert trainable_params == EXPECTED_TRUNK_TRAINABLE_PARAMS
    assert trainable_params == 5326464

    # Verify blocks 0..8 are frozen
    for i in range(9):
        for p in trunk.blocks[i].parameters():
            assert not p.requires_grad

    # Verify blocks 9..11 are trainable
    for i in range(9, 12):
        for p in trunk.blocks[i].parameters():
            assert p.requires_grad

    # Verify norm is trainable
    for p in trunk.norm.parameters():
        assert p.requires_grad

    # Verify patch_embed is frozen
    for p in trunk.patch_embed.parameters():
        assert not p.requires_grad


def test_s3_gradient_flow_and_isolation():
    """Tests that backpropagation flows to blocks[9:12] and norm, but NOT to blocks 0..8."""
    trunk = CompactViTGen2(img_size=112)
    trunk.eval().requires_grad_(False)
    for b in trunk.blocks[9:12]:
        b.train().requires_grad_(True)
    trunk.norm.train().requires_grad_(True)

    head = nn.Linear(384, 51)
    x = torch.randn(2, 3, 112, 112)
    out = trunk(x)[1]  # (2, 64, 384)
    logits = head(out)
    loss = logits.sum()
    loss.backward()

    # Trainable parts have gradients
    for i in range(9, 12):
        for p in trunk.blocks[i].parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    for p in trunk.norm.parameters():
        assert p.grad is not None

    # Frozen parts have NO gradients
    for i in range(9):
        for p in trunk.blocks[i].parameters():
            assert p.grad is None

    for p in trunk.patch_embed.parameters():
        assert p.grad is None


def test_s3_partition_is_strictly_disjoint():
    """Confirms partition_train_split partitions classes disjointly with correct counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        train_dir = root / "train"
        train_dir.mkdir(parents=True)

        tcls = [f"cls_{i}" for i in range(5)]
        bcls = [f"bg_{i}" for i in range(5)]
        all_cls = tcls + bcls

        for c in all_cls:
            cdir = train_dir / c
            cdir.mkdir()
            for img_i in range(60):
                (cdir / f"img_{img_i:03d}.JPEG").write_text("dummy")

        out_m = root / "s3_split.json"
        train_files, heldout_files = partition_train_split(
            root, tcls, bcls, heldout_per_class=10, out_manifest=out_m
        )

        assert len(heldout_files) == 10 * 10  # 10 classes * 10 heldout = 100
        assert len(train_files) == 10 * 50    # 10 classes * 50 train = 500
        assert set(heldout_files).isdisjoint(set(train_files))
        assert out_m.exists()


def test_s3_saved_checkpoint_can_be_reloaded_by_gate1():
    """Confirms the output checkpoint dictionary format is strictly compatible with load_trunk."""
    trunk = CompactViTGen2(img_size=112)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_ckpt = pathlib.Path(tmpdir) / "s3_test_ckpt.pth"
        payload = {
            "phase": "ff_112_s3",
            "arm": "s3_tuned",
            "epoch": 1,
            "base_checkpoint": "foundation_ff_112_seed17_final_d00fb750.pth",
            "base_checkpoint_sha256": "dummy",
            "best_heldout_ratio": 1.15,
            "model": {f"backbone.{k}": v for k, v in trunk.state_dict().items()},
        }
        torch.save(payload, out_ckpt)

        # Load back via Gate 1's load_trunk
        reloaded_trunk, info = load_trunk(str(out_ckpt), window=112, device="cpu")
        assert info["trunk_keys_loaded"] == info["trunk_keys_needed"]
        assert info["phase"] == "ff_112_s3"
        assert info["arm"] == "s3_tuned"

