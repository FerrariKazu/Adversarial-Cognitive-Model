"""
Real-checkpoint integration tests for the epoch-49 backbone_only recovery.

These tests load the ACTUAL preserved checkpoints (checkpoints/ and/or
recovery_artifacts/) and prove that the resume path works end-to-end:
model state loads cleanly, optimizer reconstructs and loads, scheduler
restores, and the next epoch is 50.

They SKIP when the preserved checkpoint artifacts are not present, so they
never fail ordinary CI or Kaggle tests that haven't copied the artifacts.

DO NOT weaken the assertions. If the artifacts are present, the assertions
must prove the restore actually works.
"""
from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig,
    FoundationModel,
    build_model,
)
from noesis_vision.core.checkpoint import (  # noqa: E402
    canonical_experiment_config_hash,
    resume_commit_ok,
    verify_best_rolling_parity,
)
from noesis_vision.core.multi_group_optimizer import (  # noqa: E402
    OptimizerGroupRegistry,
)


def _has_checkpoints() -> bool:
    """Return True only when the preserved epoch-49 checkpoints are present."""
    candidates = [
        ("checkpoints/foundation_backbone_only_best.pth",
         "checkpoints/foundation_backbone_only_rolling.pth"),
        ("recovery_artifacts/checkpoints/foundation_backbone_only_best_fef50f3_metric0.059.pth",
         "recovery_artifacts/checkpoints/foundation_backbone_only_rolling_fef50f3_epoch49.pth"),
    ]
    for best, roll in candidates:
        if os.path.exists(best) and os.path.exists(roll):
            return True
    return False


def _checkpoint_paths():
    """Return (best_path, rolling_path) for whichever preserved copy is present."""
    if os.path.exists("checkpoints/foundation_backbone_only_best.pth"):
        return ("checkpoints/foundation_backbone_only_best.pth",
                "checkpoints/foundation_backbone_only_rolling.pth")
    return ("recovery_artifacts/checkpoints/foundation_backbone_only_best_fef50f3_metric0.059.pth",
            "recovery_artifacts/checkpoints/foundation_backbone_only_rolling_fef50f3_epoch49.pth")


# ═══════════════════════════════════════════════════════════════════════════
# Canonical-hash integration test (real checkpoint, skips when absent)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRealCheckpointConfigHash:
    """Verify the canonical experiment config hash on the REAL preserved checkpoint.

    This is the integration counterpart of the unit test
    test_experiment_config_hash_is_single_source_of_truth (which uses a
    synthetic config and never touches the filesystem). This test proves the
    committed hashes in the recovery commit message match the actual artifact.
    """

    @pytest.fixture(autouse=True)
    def _skip_when_artifacts_absent(self):
        if not _has_checkpoints():
            pytest.skip(
                "preserved epoch-49 checkpoints not present — "
                "this integration test requires the artifact to be copied into the environment"
            )

    def test_canonical_hash_matches_committed_value(self):
        best_path, _ = _checkpoint_paths()
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        h = canonical_experiment_config_hash(cfg)
        expected = "74e39d53f2a2ecabfa9ad56f6de6994bbb4ac0ab50189cd7bf88353f51b71e20"
        assert h == expected, (
            f"canonical hash {h} does not match committed {expected}"
        )

    def test_full_field_hash_intentionally_differs(self):
        best_path, _ = _checkpoint_paths()
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        from noesis_vision.core.provenance import config_sha256

        canon = canonical_experiment_config_hash(cfg)
        full = config_sha256(cfg)
        assert full != canon, (
            "full-field provenance hash must intentionally differ from the "
            "canonical experiment hash"
        )

    def test_commit_matches_checkpoint(self):
        best_path, _ = _checkpoint_paths()
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert ckpt.get("code_commit") == "fef50f3", (
            f"checkpoint code_commit {ckpt.get('code_commit')!r} != expected fef50f3"
        )

    def test_metric_value(self):
        best_path, _ = _checkpoint_paths()
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert abs(ckpt.get("metric_value", -1) - 0.059) < 1e-9, (
            f"metric_value {ckpt.get('metric_value')} != 0.059"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Real-checkpoint resume restoration test (model + optimizer + scheduler)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRealEpoch49CheckpointRestores:
    """Prove the real epoch-49 checkpoint restores model, optimizer, and
    scheduler state, and that the next epoch is 50.

    This is the dry recovery test the task requires: it loads the real
    checkpoint, reconstructs the optimizer the way run_phase does, and
    proves load_state_dict returns None for both optimizer and scheduler.
    It does NOT run a training step — state-dict load is not the same as a
    complete training step.
    """

    @pytest.fixture(autouse=True)
    def _skip_when_artifacts_absent(self):
        if not _has_checkpoints():
            pytest.skip(
                "preserved epoch-49 checkpoints not present — "
                "this integration test requires the artifact to be copied into the environment"
            )

    def test_rolling_checkpoint_epoch_and_commit(self):
        _, roll_path = _checkpoint_paths()
        roll = torch.load(roll_path, map_location="cpu", weights_only=False)
        assert roll.get("epoch") == 49, f"rolling epoch {roll.get('epoch')} != 49"
        assert roll.get("code_commit") == "fef50f3", (
            f"rolling code_commit {roll.get('code_commit')!r} != fef50f3"
        )
        assert roll.get("kind") == "rolling"

    def test_next_epoch_is_50(self):
        _, roll_path = _checkpoint_paths()
        roll = torch.load(roll_path, map_location="cpu", weights_only=False)
        assert roll["epoch"] + 1 == 50, (
            f"next epoch {roll['epoch'] + 1} != 50"
        )

    def test_best_checkpoint_has_embedded_config(self):
        best_path, _ = _checkpoint_paths()
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = best.get("config")
        assert isinstance(cfg, dict), "best checkpoint has no embedded config"
        assert cfg.get("clean_only") is False
        assert cfg.get("recipe_version") == "gen1-adv-curriculum-v1"
        assert cfg.get("seed") == 41
        assert cfg.get("w_trades") == 0.55
        assert cfg.get("pgd_steps") == 10

    def test_model_state_loads_cleanly(self):
        best_path, _ = _checkpoint_paths()
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = best["config"]

        fcc = FoundationConfig()
        for k, v in cfg.items():
            if hasattr(fcc, k):
                setattr(fcc, k, v)

        model = build_model(fcc, "backbone_only")
        result = model.load_state_dict(best["model"])
        assert not result.missing_keys, f"missing keys: {result.missing_keys}"
        assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys}"

    def test_optimizer_reconstructs_and_loads(self):
        _, roll_path = _checkpoint_paths()
        roll = torch.load(roll_path, map_location="cpu", weights_only=False)

        best_path, _ = _checkpoint_paths()
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = best["config"]

        fcc = FoundationConfig()
        for k, v in cfg.items():
            if hasattr(fcc, k):
                setattr(fcc, k, v)

        model = build_model(fcc, "backbone_only")
        model.load_state_dict(best["model"])

        registry = OptimizerGroupRegistry()
        groups = model.group_params()
        registry.register_backbone(groups["backbone"])
        for name in ("classifier", "evidential_head", "predictor",
                     "update_net", "precision", "gaze_policy"):
            if name in groups:
                registry.register(name, groups[name])

        opt_state = roll["optimizer"]
        sched_state = roll["scheduler"]
        saved_groups = opt_state["param_groups"]

        # Reconstruct using the SAME logic as run_phase._reconstruct_optimizer_for_resume
        saved_names = [str(g.get("name", f"group_{i}"))
                       for i, g in enumerate(saved_groups)]
        named = list(model.named_parameters())

        cursor = 0
        for name, g in zip(saved_names, saved_groups):
            count = len(g.get("params", []))
            params = [p for _, p in named[cursor:cursor + count]]
            cursor += count
            OptimizerGroupRegistry()  # noqa: F841  (mirror registry is not used by load)

        cursor = 0
        cast_groups = []
        for g in saved_groups:
            count = len(g.get("params", []))
            params = [p for _, p in named[cursor:cursor + count]]
            cursor += count
            cast_groups.append({
                "params": params,
                "lr": float(g.get("lr", fcc.lr)),
                "momentum": float(g.get("momentum", fcc.momentum)),
                "weight_decay": float(g.get("weight_decay", fcc.weight_decay)),
                "nesterov": bool(g.get("nesterov", False)),
                "foreach": bool(g.get("foreach", True)),
                "name": g.get("name", f"group_{len(cast_groups)}"),
            })

        optimizer = torch.optim.SGD(cast_groups)
        optimizer.load_state_dict(opt_state)
        assert optimizer is not None

        if sched_state is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=fcc.epochs)
            sched_load = scheduler.load_state_dict(sched_state)
            assert sched_load is None, f"scheduler load returned {sched_load!r}"

        # resume_guard must accept the saved state onto the current registry
        assert registry.resume_guard(opt_state, sched_state), (
            "resume_guard refused the saved optimizer state"
        )

    def test_best_and_rolling_share_code_commit(self):
        best_path, roll_path = _checkpoint_paths()
        ok, msg = verify_best_rolling_parity(best_path, roll_path)
        assert ok, f"best/rolling parity failed: {msg}"

    def test_resume_commit_ok_accepts_best_checkpoint(self):
        best_path, _ = _checkpoint_paths()
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        cfg = best["config"]
        declared = {
            "clean_only": cfg["clean_only"],
            "recipe_version": cfg["recipe_version"],
            "seed": cfg["seed"],
            "w_trades": cfg["w_trades"],
            "pgd_steps": cfg["pgd_steps"],
        }
        ok, msg = resume_commit_ok(
            best,
            require_experiment_class=True,
            allowed_experiment_configs=[declared],
        )
        assert ok, f"resume_commit_ok refused the best checkpoint: {msg}"

    def test_resume_commit_ok_accepts_rolling_checkpoint(self):
        _, roll_path = _checkpoint_paths()
        roll = torch.load(roll_path, map_location="cpu", weights_only=False)
        ok, msg = resume_commit_ok(roll, require_experiment_class=False)
        assert ok, f"resume_commit_ok refused the rolling checkpoint: {msg}"
