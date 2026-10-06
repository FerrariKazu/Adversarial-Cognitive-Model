"""
Agent A contract test — provenance manifests.

Fields present; hash changes on config change; overwrite of an existing
experiment_id is refused (never silent).
"""
import pytest

from noesis_vision.core.provenance import (
    config_sha256,
    file_sha256,
    load_manifest,
    manifest_exists,
    write_manifest,
)
from noesis_vision.core.schema import RHANNXAConfig


def test_manifest_fields_present(tmp_path):
    cfg = RHANNXAConfig()
    m = write_manifest("exp_fields", cfg, root_dir=str(tmp_path),
                       seed=41, dataset_version="stl10:test")
    for field in ("experiment_id", "git_commit", "config_sha256",
                  "dataset_version", "seed", "timestamp_utc"):
        assert field in m, f"manifest missing provenance field {field}"
    assert m["experiment_id"] == "exp_fields"
    assert m["seed"] == 41
    assert m["config_sha256"] == config_sha256(cfg)
    # Round-trips through load_manifest.
    loaded = load_manifest("exp_fields", root_dir=str(tmp_path))
    assert loaded == m


def test_hash_changes_on_config_change():
    base = config_sha256(RHANNXAConfig())
    # A non-status field change.
    changed_k = config_sha256(RHANNXAConfig(num_candidates=8))
    assert changed_k != base
    # A STATUS-FLAG change also changes the hash (status is config).
    changed_status = config_sha256(RHANNXAConfig(enable_v1_frontend=True))
    assert changed_status != base
    # Same config -> same hash (canonical serialization, deterministic).
    assert config_sha256(RHANNXAConfig()) == base


def test_overwrite_refused(tmp_path):
    cfg = RHANNXAConfig()
    write_manifest("exp_once", cfg, root_dir=str(tmp_path), seed=41)
    assert manifest_exists("exp_once", root_dir=str(tmp_path))
    # Same id, even with the SAME config: refuse — provenance is immutable.
    with pytest.raises(FileExistsError, match="refusing silent overwrite"):
        write_manifest("exp_once", cfg, root_dir=str(tmp_path), seed=41)
    # ...and with a DIFFERENT config: refuse just as loudly.
    with pytest.raises(FileExistsError, match="refusing silent overwrite"):
        write_manifest("exp_once", RHANNXAConfig(num_candidates=5),
                       root_dir=str(tmp_path), seed=41)
    # The original manifest is untouched by the refused attempts.
    m = load_manifest("exp_once", root_dir=str(tmp_path))
    assert m["config_sha256"] == config_sha256(cfg)


def test_checkpoint_hash_recorded(tmp_path):
    ckpt = tmp_path / "model_best.pth"
    ckpt.write_bytes(b"\x00" * 64)
    m = write_manifest("exp_ckpt", RHANNXAConfig(), root_dir=str(tmp_path),
                       checkpoint_path=str(ckpt))
    assert m["checkpoint_sha256"] == file_sha256(str(ckpt))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_manifest("exp_missing_ckpt", RHANNXAConfig(),
                       root_dir=str(tmp_path),
                       checkpoint_path=str(tmp_path / "nope.pth"))


def test_extra_cannot_override_core_fields(tmp_path):
    with pytest.raises(ValueError, match="collide"):
        write_manifest("exp_extra", RHANNXAConfig(), root_dir=str(tmp_path),
                       extra={"git_commit": "forged"})
