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
from noesis_vision.core.checkpoint import (
    _checkpoint_embedded_config_hash,
    canonical_experiment_config_hash,
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


def test_experiment_config_hash_is_single_source_of_truth():
    """The reconciled RHAN-NXA foundation experiment config hash is computed by
    ONE canonicalization: the embedded config MINUS runtime/deployment fields
    (hf_token, ckpt_dir, report_dir, runs_dir, data_root).

    That canonicalization lives in noesis_vision/core/checkpoint.py and is
    exposed publicly as canonical_experiment_config_hash(). Both the checkpoint
    module and the recovery pre-flight cell use it, so the same embedded config
    always produces the same hash regardless of which module computes it.
    """
    import torch
    ckpt = torch.load(
        "recovery_artifacts/checkpoints/"
        "foundation_backbone_only_best_fef50f3_metric0.059.pth",
        map_location="cpu", weights_only=False)
    embedded = ckpt["config"]
    h1 = canonical_experiment_config_hash(embedded)
    h2 = _checkpoint_embedded_config_hash(embedded)
    assert h1 is not None
    assert h1 == h2, (
        "canonical_experiment_config_hash and "
        "_checkpoint_embedded_config_hash must agree on the SAME embedded config")
    # The full-field provenance.config_sha256 is intentionally DIFFERENT because
    # it still includes runtime/deployment fields.
    assert config_sha256(embedded) != h1
    # Round-trip: the same embedded config must produce the same canonical hash.
    assert canonical_experiment_config_hash(embedded) == h1
    # A genuine experiment-field change must change the canonical hash.
    modified = dict(embedded)
    modified["pgd_steps"] = 4
    assert canonical_experiment_config_hash(modified) != h1
    # A runtime/deployment-field change must NOT change the canonical experiment hash.
    modified2 = dict(embedded)
    modified2["data_root"] = "/kaggle/working/weird"
    assert canonical_experiment_config_hash(modified2) == h1


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


def test_experiment_config_hash_does_not_include_runtime_fields():
    """Confirm the explicit decision behind the reconciled config hash: the
    experiment hash intentionally drops hf_token/ckpt_dir/report_dir/runs_dir/
    data_root, so a runtime-path-only change does not change the experiment hash.
    """
    embedded = {
        "clean_only": False,
        "recipe_version": "gen1-adv-curriculum-v1",
        "seed": 41,
        "w_trades": 0.55,
        "pgd_steps": 10,
        "eps_list": [0.0, 0.031, 0.062, 0.094],
        "roll_every": 1,
        "data_root": "/kaggle/tmp/imagenet100",
        "ckpt_dir": "/kaggle/working/x",
        "report_dir": "/kaggle/working/y",
        "runs_dir": "/kaggle/working/z",
        "hf_token": None,
    }
    h_base = canonical_experiment_config_hash(embedded)
    assert h_base is not None
    # Runtime-field changes do not change the canonical experiment hash.
    for field, new_val in (
        ("data_root", "/kaggle/working/imagenet100"),
        ("ckpt_dir", "/home/ferrarikazu/checkpoints"),
        ("report_dir", "/home/ferrarikazu/report"),
        ("runs_dir", "/home/ferrarikazu/runs"),
        ("hf_token", "secret"),
    ):
        m = dict(embedded)
        m[field] = new_val
        assert canonical_experiment_config_hash(m) == h_base, (
            f"runtime field {field!r} must not affect the experiment config hash")
    # Experiment-field changes DO change the canonical experiment hash.
    for field, new_val in (
        ("pgd_steps", 4),
        ("clean_only", True),
        ("w_trades", 0.4),
        ("seed", 7),
    ):
        m = dict(embedded)
        m[field] = new_val
        assert canonical_experiment_config_hash(m) != h_base, (
            f"experiment field {field!r} must change the experiment config hash")

def test_extra_cannot_override_core_fields(tmp_path):
    with pytest.raises(ValueError, match="collide"):
        write_manifest("exp_extra", RHANNXAConfig(), root_dir=str(tmp_path),
                       extra={"git_commit": "forged"})
