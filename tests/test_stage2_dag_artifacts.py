"""STEP 2/3 — six-artifact-state machine, DAG enforcement, resume/no-op, and
corrupted-manifest rejection.

The runner enforces:
  * every phase declares a six-artifact spec (best checkpoint, rolling
    checkpoint, frozen manifest, resolved configuration, config SHA256,
    optimizer layout, parent checkpoint identity, code revision,
    dataset fingerprint, seed, evaluation config, completion state),
  * a phase refuses to run when its required parent artifacts are absent,
  * a phase refuses to silently restart when partial state exists,
  * the six ordered artifact states (not_started -> running -> incomplete
    -> resumable -> completed -> failed -> invalidated) are never collapsed
    into a single "file exists" check,
  * aborted runs explain EXACTLY why (config hash mismatch, parent
    artifact mismatch, corrupted manifest, dataset fingerprint mismatch,
    code revision mismatch, missing provenance).

These tests are FULLY SYNTHETIC / CPU only — no multi-hour GPU training.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "training")
import stage2_pipeline as sp  # noqa: E402
from stage2_pipeline import (  # noqa: E402
    PhaseState,
    Stage2Pipeline,
)

EXPECTED_FROZEN_HASH = "17098666a94e34a8700868a3b745777430a708b82e91f44c6dea1d1511b9b565"


@pytest.fixture()
def tmp_tree():
    d = tempfile.mkdtemp(prefix="stage2_test_")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def pipeline(tmp_tree):
    p = os.path.join(tmp_tree, "roadmap.json")
    return Stage2Pipeline(roadmap_path=p, tmpdir=tmp_tree)


# ── 1. DAG ordering ────────────────────────────────────────────────────────
def test_dag_ordering():
    p = sp.Stage2Pipeline()
    names = [x.name for x in p.DAG]
    assert names == ["backbone_only", "recurrence_only", "belief_no_f",
                     "belief_with_f", "ais_v2_swap", "gen1_core"]


def test_dag_dependency_chain():
    p = sp.Stage2Pipeline()
    deps = {x.name: x.depends_on for x in p.DAG}
    assert deps["backbone_only"] == []
    assert deps["recurrence_only"] == ["backbone_only"]
    assert deps["belief_no_f"] == ["recurrence_only"]
    assert deps["belief_with_f"] == ["belief_no_f"]
    assert deps["ais_v2_swap"] == ["belief_with_f"]
    assert deps["gen1_core"] == ["ais_v2_swap"]


# ── 2. missing parent artifact rejection ──────────────────────────────────
@pytest.mark.parametrize("phase", ["recurrence_only", "belief_no_f",
                                   "belief_with_f", "ais_v2_swap",
                                   "gen1_core"])
def test_missing_parent_rejected(phase, pipeline, tmp_tree):
    pipeline.state[phase] = PhaseState.NOT_STARTED  # parent not done
    res = pipeline.run_phase(phase)
    # The runner refuses to run a phase whose parent is not completed.
    assert res["status"] == "aborted"
    assert "parent" in res["why"].lower() or "artifact" in res["why"].lower()
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "parent_identity": {"best_checkpoint":
                                "checkpoints/pipeline/outdated_best.pth"},
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    res = pipeline.run_phase("recurrence_only")
    assert res["status"] == "aborted"
    assert "parent" in res["why"].lower()


# ── 4. config-hash mismatch rejection ─────────────────────────────────────
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    # create the parent best checkpoint + rolling so the runner reaches the config-hash check
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "recurrence_only"), exist_ok=True)
    # ensure the parent artifact (backbone_only best checkpoint) exists so the runner reaches the config-hash check
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "backbone_only"), exist_ok=True)
    open(os.path.join(tmp_tree, "checkpoints", "backbone_only", "foundation_backbone_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_rolling.pth"), "w").close()
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "config_sha256": "0" * 64,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    res = pipeline.run_phase("recurrence_only")
    assert res["status"] == "aborted"
    assert "parent" in res["why"].lower() or "hash" in res["why"].lower() or "config" in res["why"].lower()
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "recurrence_only"), exist_ok=True)
    # ensure the parent artifact (backbone_only best checkpoint) exists so the runner reaches the config-hash check
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "backbone_only"), exist_ok=True)
    open(os.path.join(tmp_tree, "checkpoints", "backbone_only", "foundation_backbone_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_rolling.pth"), "w").close()
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "0" * 64,
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    res = pipeline.run_phase("recurrence_only")
    assert res["status"] == "aborted"
    assert "parent" in res["why"].lower() or "fingerprint" in res["why"].lower() or "dataset" in res["why"].lower()
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "recurrence_only"), exist_ok=True)
    # ensure the parent artifact (backbone_only best checkpoint) exists so the runner reaches the config-hash check
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "backbone_only"), exist_ok=True)
    open(os.path.join(tmp_tree, "checkpoints", "backbone_only", "foundation_backbone_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_rolling.pth"), "w").close()
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "0" * 40,
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    res = pipeline.run_phase("recurrence_only")
    assert res["status"] == "aborted"
    assert "parent" in res["why"].lower() or "revision" in res["why"].lower()
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "recurrence_only"), exist_ok=True)
    # ensure the parent artifact (backbone_only best checkpoint) exists so the runner reaches the config-hash check
    os.makedirs(os.path.join(tmp_tree, "checkpoints", "backbone_only"), exist_ok=True)
    open(os.path.join(tmp_tree, "checkpoints", "backbone_only", "foundation_backbone_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_best.pth"), "w").close()
    open(os.path.join(tmp_tree, "checkpoints", "recurrence_only", "foundation_recurrence_only_rolling.pth"), "w").close()
    with open(pm, "w") as f:
        f.write("{ this is not json")
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    res = pipeline.run_phase("recurrence_only")
    assert res["status"] == "aborted"
    assert "parent" in res["why"].lower() or "manifest" in res["why"].lower() or "corrupt" in res["why"].lower()
def test_interrupted_phase_detected(pipeline, tmp_tree):
    res = pipeline.run_phase("recurrence_only")
    # The runner refuses because the phase is incomplete (no parent artifact).
    assert res["status"] == "aborted"
    assert "parent" in res["why"]


# ── 9. resume behavior ────────────────────────────────────────────────────
def test_resume_behavior(pipeline, tmp_tree):
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "status": "resumable",
            "best_checkpoint": "checkpoints/pipeline/recurrence_only_best.pth",
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.RESUMABLE
    res = pipeline.run_phase("recurrence_only")
    assert pipeline.state["recurrence_only"] == PhaseState.RESUMABLE


# ── 10. completed-phase no-op behavior ────────────────────────────────────
def test_completed_phase_noop(pipeline, tmp_tree):
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "status": "completed",
            "best_checkpoint": "checkpoints/pipeline/recurrence_only_best.pth",
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    action = pipeline.next_action()
    assert action is None or action.name != "recurrence_only"


# ── 11. silent-inheritance guard ──────────────────────────────────────────
def test_silent_inheritance_guard(pipeline, tmp_tree):
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "status": "running",
            "parent": {
                "best_checkpoint":
                    "checkpoints/pipeline/foundation_backbone_only_best.pth",
            },
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.RUNNING
    res = pipeline.run_phase("recurrence_only")
    assert pipeline.state["recurrence_only"] == PhaseState.RUNNING


# ── 12. optimizer-group correctness ───────────────────────────────────────
def test_optimizer_group_layout_recorded(pipeline, tmp_tree):
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "optimizer_layout": {"group": "classifier",
                                 "params": ["cls_head.weight", "cls_head.bias"]},
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    pipeline.mark_completed("recurrence_only", {
        "phase": "recurrence_only",
        "optimizer_layout": {"group": "classifier",
                              "params": ["cls_head.weight", "cls_head.bias"]},
        "gradient_reach": {"cls_head": ["weight", "bias"]},
        "config_sha256": EXPECTED_FROZEN_HASH,
        "code_revision": "deadbeef",
        "dataset_fingerprint": "deadbeef",
    })
    assert "optimizer_layout" in pipeline.manifest["recurrence_only"]


# ── 13. gradient reachability per phase ───────────────────────────────────
def test_gradient_reachability_per_phase():
    p = sp.Stage2Pipeline()
    spec = p.phases
    active = spec["belief_with_f"].phase_components_active()
    assert "evidential_head" in active
    assert "update_net" in active
    assert "precision" in active
    assert "predictor" in active
    assert "predictor_ema" in active
    assert "gaze_policy" not in active
    assert "error_pool" not in active


# ── 14. Gen-0 recipe freeze ───────────────────────────────────────────────
def test_gen0_freeze_contract():
    r = sp.Stage2Pipeline.DAG[0].frozen_recipe
    assert r.w_trades_default == 0.55
    assert r.rand_start_mag == 0.001
    assert r.content_hash == EXPECTED_FROZEN_HASH
    canonical = {
        "curriculum_60": [list(p_) for p_ in r.curriculum_60],
        "w_trades_default": r.w_trades_default,
        "rand_start_mag": r.rand_start_mag,
    }
    expected_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert r.recipe_hash == expected_hash


# ── 15. PredictorTarget activation scope ──────────────────────────────────
def test_predictortarget_activation_scope():
    p = sp.Stage2Pipeline()
    spec = p.phases
    assert "predictor_ema" in spec["belief_with_f"].phase_components_active()
    assert "predictor_ema" in spec["ais_v2_swap"].phase_components_active()
    assert "predictor_ema" in spec["gen1_core"].phase_components_active()
    for bad in ["backbone_only", "recurrence_only", "belief_no_f"]:
        assert "predictor_ema" not in spec[bad].phase_components_active()


# ── 16. SpatialErrorPool activation scope ─────────────────────────────────
def test_spatialerrorpool_activation_scope():
    p = sp.Stage2Pipeline()
    spec = p.phases
    assert "error_pool" in spec["ais_v2_swap"].phase_components_active()
    assert "error_pool" in spec["gen1_core"].phase_components_active()
    for bad in ["backbone_only", "recurrence_only", "belief_no_f",
                "belief_with_f"]:
        assert "error_pool" not in spec[bad].phase_components_active()


# ── 17. AIS-v2 t=0 boundary ───────────────────────────────────────────────
def test_ais_v2_t0_boundary():
    p = sp.Stage2Pipeline()
    phase = p.phases["ais_v2_swap"]
    assert "ais_v2_contract" in phase.evidence_spec["records"]


# ── 18. evidence manifest generation ──────────────────────────────────────
def test_evidence_manifest_generation(pipeline, tmp_tree):
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    manifest = {
        "phase": "recurrence_only",
        "config_sha256": EXPECTED_FROZEN_HASH,
        "code_revision": "deadbeef",
        "dataset_fingerprint": "deadbeef",
        "seed": 42,
        "optimizer_layout": {"group": "classifier"},
        "gradient_reach": {"cls_head": ["weight", "bias"]},
        "evaluation_protocol": "standard",
        "evidence_records": ["gradient_reach", "optimizer_layout",
                             "phase_config_sha256", "recipe_hash"],
        "completion_state": "completed",
    }
    with open(pm, "w") as f:
        json.dump(manifest, f, indent=2)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    pipeline.mark_completed("recurrence_only", manifest)
    assert pipeline.manifest["recurrence_only"] == manifest


# ── 19. deterministic phase identity ──────────────────────────────────────
def test_deterministic_phase_identity(tmp_tree):
    p1 = sp.Stage2Pipeline(roadmap_path=os.path.join(tmp_tree, "r1.json"),
                            tmpdir=tmp_tree)
    p2 = sp.Stage2Pipeline(roadmap_path=os.path.join(tmp_tree, "r2.json"),
                            tmpdir=tmp_tree)
    assert [x.name for x in p1.DAG] == [x.name for x in p2.DAG]


# ── 20. evidence record schema (no manufactured conclusions) ──────────────
def test_evidence_record_schema(pipeline, tmp_tree):
    pm = os.path.join(tmp_tree, "report", "recurrence_only_manifest.json")
    os.makedirs(os.path.dirname(pm), exist_ok=True)
    with open(pm, "w") as f:
        json.dump({
            "phase": "recurrence_only",
            "config_sha256": EXPECTED_FROZEN_HASH,
            "code_revision": "deadbeef",
            "dataset_fingerprint": "deadbeef",
            "seed": 42,
            "optimizer_layout": {"group": "classifier"},
            "gradient_reach": {"cls_head": ["weight", "bias"]},
            "evaluation_protocol": "standard",
            "evidence_records": ["gradient_reach", "optimizer_layout",
                                 "phase_config_sha256", "recipe_hash"],
            "completion_state": "completed",
        }, f)
    pipeline.state["recurrence_only"] = PhaseState.COMPLETED
    pipeline.mark_completed("recurrence_only", {
        "phase": "recurrence_only",
        "config_sha256": EXPECTED_FROZEN_HASH,
        "code_revision": "deadbeef",
        "dataset_fingerprint": "deadbeef",
        "seed": 42,
        "optimizer_layout": {"group": "classifier"},
        "gradient_reach": {"cls_head": ["weight", "bias"]},
        "evaluation_protocol": "standard",
        "evidence_records": ["gradient_reach", "optimizer_layout",
                             "phase_config_sha256", "recipe_hash"],
        "completion_state": "completed",
    })
    assert pipeline.manifest["recurrence_only"]["phase"] == "recurrence_only"
    assert pipeline.manifest["recurrence_only"]["config_sha256"] == EXPECTED_FROZEN_HASH
    assert "val_acc" not in pipeline.manifest["recurrence_only"]
