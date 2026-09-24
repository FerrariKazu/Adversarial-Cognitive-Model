"""
Agent C contract test — DINOv2 warm-start key matching.

Builds a synthetic "DINOv2-small checkpoint" from the trunk's own keys
(the trunk naming mirrors DINOv2 1:1), then verifies: exact-match report,
threshold check, refusal below the band (nothing applied), and that
new-by-design keys are reported without poisoning the metric.
"""
import pytest
import torch

from noesis_vision.models.backbone import (
    WARM_START_MATCH_FRACTION_MIN,
    CompactViT,
)


def _dino_like_checkpoint(path, m, drop_keys=(), corrupt_shapes=()):
    """Simulate a DINOv2-small checkpoint: the trunk's mappable keys with
    fresh values (DINOv2 trunk naming == our trunk naming)."""
    sd = {k: v.clone() for k, v in m.state_dict().items()
          if k.startswith(("blocks.", "patch_embed.", "norm."))}
    for k in drop_keys:
        sd.pop(k, None)
    for k in corrupt_shapes:
        sd[k] = torch.zeros(1)  # wrong shape on purpose
    torch.save({"model": sd}, path)
    return path


def test_perfect_mapping_matches_and_loads(tmp_path):
    m = CompactViT(img_size=56)
    ref_sd = {k: v.clone() for k, v in m.state_dict().items()}
    ckpt = _dino_like_checkpoint(str(tmp_path / "dino.pth"), m)
    fresh = CompactViT(img_size=56)
    report = fresh.load_dino_warm_start(ckpt)
    assert report["accepted"] is True
    assert report["matched_fraction"] == pytest.approx(1.0)
    # Trunk weights ACTUALLY loaded (values, not just key counting).
    for k in ("blocks.0.attn.qkv.weight", "norm.weight",
              "patch_embed.proj.weight"):
        assert torch.allclose(fresh.state_dict()[k], ref_sd[k]), k
    # New-by-design keys keep their fresh init and are reported.
    assert any(k.startswith("refinement.") for k in
               report["missing_new_by_design"])


def test_missing_trunk_keys_lower_fraction(tmp_path):
    m = CompactViT(img_size=56)
    # Drop one block's 8 keys -> fraction (100-8)/100 = 0.92 < 0.95.
    ckpt = _dino_like_checkpoint(
        str(tmp_path / "dino.pth"), m,
        drop_keys=[k for k in m.state_dict()
                   if k.startswith("blocks.3.")])
    fresh = CompactViT(img_size=56)
    before = fresh.state_dict()["blocks.3.attn.qkv.weight"].clone()
    report = fresh.load_dino_warm_start(ckpt)
    assert report["matched_fraction"] < WARM_START_MATCH_FRACTION_MIN
    assert report["accepted"] is False
    # NOTHING applied on refusal — never a partial, silently-degraded load.
    after = fresh.state_dict()["blocks.3.attn.qkv.weight"]
    assert torch.equal(before, after)


def test_shape_mismatch_is_skipped_not_forced(tmp_path):
    m = CompactViT(img_size=56)
    ckpt = _dino_like_checkpoint(
        str(tmp_path / "dino.pth"), m,
        corrupt_shapes=["blocks.5.norm1.weight"])  # the STOP-condition shape
    fresh = CompactViT(img_size=56)
    report = fresh.load_dino_warm_start(ckpt)
    assert any("shape mismatch" in (s.get("reason") or "")
               for s in report["skipped"].values())
    # The corrupted key was NOT loaded.
    fresh_sd = fresh.state_dict()
    ref_sd = CompactViT(img_size=56).state_dict()
    assert torch.equal(fresh_sd["blocks.5.norm1.weight"],
                       ref_sd["blocks.5.norm1.weight"])


def test_trunk_keys_accounted(tmp_path):
    """The mappable trunk set is the 12 blocks + patch embed + final norm:
    12 keys per block (2x norm1 + 2x qkv + 2x proj + 2x norm2 + 2x fc1
    + 2x fc2) = 144, + patch_embed 2 + final norm 2 = 148. The metric
    denominates over exactly that."""
    m = CompactViT(img_size=56)
    trunk_keys = [k for k in m.state_dict()
                  if k.startswith(("blocks.", "patch_embed.", "norm."))]
    assert len(trunk_keys) == 148
    ckpt = _dino_like_checkpoint(str(tmp_path / "dino.pth"), m)
    report = CompactViT(img_size=56).load_dino_warm_start(ckpt)
    assert report["trunk_keys"] == 148
    assert report["matched_trunk_keys"] == 148


def test_threshold_constant_documented():
    assert WARM_START_MATCH_FRACTION_MIN == 0.95
