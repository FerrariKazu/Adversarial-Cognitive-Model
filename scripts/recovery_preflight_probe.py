#!/usr/bin/env python3
"""
Standalone recovery preflight probe for the Gen-1 epoch-49
backbone_only resume path. Mirrors the Kaggle Step 7.5 logic closely
but is runnable from a normal shell / CI environment (it is NOT the
notebook itself). It exercises the SAME checkpoint module and the
same canonicalization decisions, and it verifies hash + commit + resume
compatibility + parity + next epoch, without starting training.

It does not import the notebook. It imports the project modules.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from noesis_vision.core.checkpoint import (
    canonical_experiment_config_hash,
    resume_commit_ok,
    verify_best_rolling_parity,
    current_code_commit,
)
from noesis_vision.core.provenance import config_sha256

ROLLING_NAME = "foundation_backbone_only_rolling_fef50f3_epoch49.pth"
BEST_NAME = "foundation_backbone_only_best_fef50f3_metric0.059.pth"
RECOVERY_DIR = os.path.join("recovery_artifacts", "checkpoints")

COMMITTED_CANONICAL_HASH = (
    "74e39d53f2a2ecabfa9ad56f6de6994bbb4ac0ab50189cd7bf88353f51b71e20"
)
COMMITTED_BEST_SHA256 = (
    "8fadb049ceafd4243a6cf7b7685b4f90433d8823ebc57250352efc03dba6bf8a"
)
COMMITTED_ROLLING_SHA256 = (
    "8f6a88d60e0f6c44c37434f94924a03ca774bb6d42003a251d678017c46343be"
)
COMMITTED_COMMIT = "fef50f3"
COMMITTED_METRIC = 0.059


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    roll_path = os.path.join(RECOVERY_DIR, ROLLING_NAME)
    best_path = os.path.join(RECOVERY_DIR, BEST_NAME)

    print("== Recovery preflight probe (shell, not notebook) ==")
    print(f"  rolling: {roll_path}")
    print(f"  best   : {best_path}")
    print(f"  current_code_commit (record only): {current_code_commit()}")

    for label, path, expected_sha in [
        ("rolling", roll_path, COMMITTED_ROLLING_SHA256),
        ("best", best_path, COMMITTED_BEST_SHA256),
    ]:
        if not os.path.exists(path):
            print(f"  MISSING: {label} checkpoint at {path}")
            return 2
        sha = file_sha256(path)
        ok = sha == expected_sha
        print(f"  {label} sha256 {'OK' if ok else 'MISMATCH'}: {sha}")
        if not ok:
            print(f"    expected {expected_sha}")
            return 3

    roll = torch.load(str(roll_path), map_location="cpu", weights_only=False)
    best = torch.load(str(best_path), map_location="cpu", weights_only=False)

    print("  rolling epoch        :", roll.get("epoch"))
    print("  rolling code_commit  :", roll.get("code_commit"))
    print("  rolling kind         :", roll.get("kind"))
    print("  best code_commit     :", best.get("code_commit"))
    print("  best metric_value    :", best.get("metric_value"))
    print("  best kind            :", best.get("kind"))
    print("  best embedded config :", bool(best.get("config")))
    print("  rolling optimizer    :", "optimizer" in roll)
    print("  rolling scheduler    :", "scheduler" in roll)

    best_cfg = best.get("config")
    if not isinstance(best_cfg, dict) or not best_cfg:
        print("  STOP: best checkpoint has no usable embedded config")
        return 4

    canon = canonical_experiment_config_hash(best_cfg)
    full = config_sha256(best_cfg)
    print("  canonical_experiment_config_hash:", canon)
    print("  full-field provenance.config_sha256:", full)

    if canon != COMMITTED_CANONICAL_HASH:
        print(f"  STOP: canonical hash mismatch: got {canon}, expected {COMMITTED_CANONICAL_HASH}")
        return 5
    if full == canon:
        print("  STOP: canonical hash unexpectedly equals full-field hash")
        return 6

    declared_experiment_class = {
        "clean_only": best_cfg.get("clean_only"),
        "recipe_version": best_cfg.get("recipe_version"),
        "seed": best_cfg.get("seed"),
        "w_trades": best_cfg.get("w_trades"),
        "pgd_steps": best_cfg.get("pgd_steps"),
    }
    allowed = [declared_experiment_class]
    print("  declared experiment class (verbatim from best config):")
    for k, v in sorted(declared_experiment_class.items()):
        print(f"    {k:<14}: {v!r}")

    ok_best, why_best = resume_commit_ok(
        best,
        require_experiment_class=True,
        allowed_experiment_configs=allowed,
    )
    print("  GUARD A best-experiment-class compatibility ok:", ok_best, "why:", why_best)
    if not ok_best:
        print(f"  STOP: {why_best}")
        return 7

    ok_roll, why_roll = resume_commit_ok(roll, require_experiment_class=False)
    print("  GUARD B rolling training-fingerprint identity ok:", ok_roll, "why:", why_roll)
    if not ok_roll:
        print(f"  STOP: {why_roll}")
        return 8

    par_ok, par_why = verify_best_rolling_parity(str(best_path), str(roll_path))
    print("  GUARD C best/rolling parity ok:", par_ok, "why:", par_why)
    if not par_ok:
        print(f"  STOP: {par_why}")
        return 9

    next_epoch = int(roll["epoch"]) + 1
    print("  next resume epoch (rolling epoch + 1):", next_epoch)
    if next_epoch != 50:
        print("  STOP: expected next epoch 50, got", next_epoch)
        return 10

    print()
    print("== PROBE PASSED: provenance, hashes, resume compatibility, parity, next epoch ==",)
    print("  NOTE: this is a PROVENANCE/STATE-DICT probe. It does NOT run a training step,")
    print("  so it is not proof that a full training iteration works. It does prove that the")
    print("  checkpoint is loadable and that the expected next epoch is 50.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
