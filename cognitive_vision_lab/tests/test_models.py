"""Unit tests for cognitive_vision_lab.backend.models (resolution/registry only)."""
import pytest

from cognitive_vision_lab.backend.models import (
    RHAN_ARCHES,
    RHAN_MODELS,
    TORCHVISION_MODELS,
    _arch_key_for,
    _stem,
    discover_checkpoints,
    list_models,
)


class TestStem:
    def test_strips_suffixes(self):
        assert _stem("rhan_stl10_v11_best.pth") == "rhan_stl10_v11"
        assert _stem("rhan_stl10_large_pseudolabel_best.pth") == "rhan_stl10_large_pseudolabel"
        assert _stem("rhan_v10_final.pth") == "rhan_v10"

    def test_strips_zone_identifier(self):
        assert _stem("foo_best.pth:Zone.Identifier") == "foo"


class TestArchResolution:
    def test_v11_prefix_wins_over_stl10(self):
        # rhan_stl10_v11 must resolve to RHANv11, not base RHANSTL10
        assert _arch_key_for("rhan_stl10_v11_best.pth") == "rhan_v11"

    def test_large_pseudolabel(self):
        assert _arch_key_for("rhan_stl10_large_pseudolabel_best.pth") == "rhan_stl10_large"

    def test_unknown_falls_back(self):
        assert _arch_key_for("mystery_network_best.pth") == "rhan"


class TestRegistry:
    def test_torchvision_baselines_present(self):
        for mid in ["ResNet-18", "ViT-B-16", "EfficientNet-B0", "Swin-T"]:
            assert mid in TORCHVISION_MODELS

    def test_curated_rhan_models_present(self):
        assert "RHAN-v11 (Best)" in RHAN_MODELS
        assert "RHAN-Large (Pseudolabel best)" in RHAN_MODELS

    def test_registry_arch_keys_valid(self):
        for label, info in RHAN_MODELS.items():
            assert info["arch"] in RHAN_ARCHES, label

    def test_run_b_freeze_gaze_flag(self):
        assert RHAN_MODELS["RHAN-v11 (Isolation Run B)"].get("freeze_gaze") is True

    def test_checkpoint_names_well_formed(self):
        for label, info in RHAN_MODELS.items():
            assert info["checkpoint"].endswith(".pth"), label
            assert ":Zone.Identifier" not in info["checkpoint"], label

    def test_curated_checkpoints_exist_on_disk(self):
        """Every curated checkpoint must exist locally — the regression that
        previously left Model Zoo rows permanently unavailable."""
        from cognitive_vision_lab.config import CHECKPOINTS_DIR, CHECKPOINTS_TIER2_DIR

        on_disk = set()
        for d in (CHECKPOINTS_DIR, CHECKPOINTS_TIER2_DIR):
            if d.exists():
                on_disk |= {p.name.replace(":Zone.Identifier", "")
                            for p in d.iterdir() if p.name.endswith(".pth")}
        if not on_disk:
            pytest.skip("no checkpoints present on this host (fresh clone)")
        for label, info in RHAN_MODELS.items():
            assert info["checkpoint"] in on_disk, \
                f"{label} -> {info['checkpoint']} missing on disk"

    def test_list_models_has_flags(self):
        rows = list_models()
        ids = [r["id"] for r in rows]
        assert "ResNet-18" in ids
        assert any(r["stl10"] for r in rows)
        for r in rows:
            assert "available" in r and "family" in r

    def test_discover_checkpoints_no_crash(self):
        found = discover_checkpoints()
        assert isinstance(found, dict)
        for fname, info in found.items():
            assert fname.endswith(".pth")
            assert "arch" in info and "path" in info
