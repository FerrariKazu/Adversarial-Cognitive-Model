#!/usr/bin/env python3
"""
freeze_run_manifest.py — PHASE 7: the FINAL RUN MANIFEST (Gen-1 production).
================================================================================
Called ONCE, immediately before the production launch, by the operator.
Captures everything needed to reproduce exactly what is being launched and
FREEZES it:

  * code identity: git SHA, branch, dirty/untracked state, config hash
    (config_sha256 of the trainer's FoundationConfig.to_dict() — the SAME
    hash the provenance manifests use);
  * dataset identity: the converter's fingerprint.json (pinned HF revision,
    file counts, aggregate listing hash) — copied into the manifest;
  * environment: python/torch/cuda versions, GPU name/VRAM/driver,
    hostname, timestamp;
  * protocol: stage sequence, seed policy, optimizer/scheduler settings,
    gaze schemes per phase, eval protocol, checkpoint cadence;
  * destinations: local checkpoint/report/runs dirs, HF repos.

The manifest is written to runs/production_launch_manifest.json and synced
to the rolling HF repo. FROM THIS POINT the scientific configuration is
FROZEN: no architecture, hyperparameter, preprocessing, optimizer, loss,
seed, or mechanism changes mid-run (Phase 9 discipline).

Usage:  python3 scripts/freeze_run_manifest.py [--data-root data/imagenet100]

NOTE on the config hash: it MUST be computed over the configuration the
run will actually use — the canonical local launcher
(cloud_setup/run_j1_local.sh) applies DEFAULTS --batch-size 48
--num-workers 4, so the same overrides are applied HERE before hashing.
If the launcher's defaults ever change, this script changes with them
(audibly, before a new freeze — never during a run).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from noesis_vision.core.provenance import config_sha256  # noqa: E402
from training.stage_state_machine import FOUNDATION_PHASES  # noqa: E402
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, gaze_scheme_for_phase)


def _git(*args: str) -> str:
    return subprocess.run(f"git {' '.join(args)}", shell=True, cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    data_root = os.path.join(REPO_ROOT, "data", "imagenet100")
    for i, a in enumerate(sys.argv):
        if a == "--data-root" and i + 1 < len(sys.argv):
            data_root = os.path.abspath(sys.argv[i + 1])

    fp_path = os.path.join(data_root, "fingerprint.json")
    if not os.path.exists(fp_path):
        print(f"STOP — dataset fingerprint missing: {fp_path}\n"
              f"  run scripts/prepare_imagenet100.py first (Phase 3).")
        return 1
    fingerprint = json.load(open(fp_path))

    cfg = FoundationConfig()
    cfg.data_root = data_root
    # Mirror cloud_setup/run_j1_local.sh DEFAULTS exactly (see docstring).
    cfg.batch_size = 48
    cfg.num_workers = 4
    config_hash = config_sha256(cfg.to_dict())

    try:
        import torch
        env = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None),
            "gpu_vram_bytes": (torch.cuda.get_device_properties(0).total_memory
                               if torch.cuda.is_available() else None),
            "gpu_count": torch.cuda.device_count(),
        }
    except Exception as e:                                   # pragma: no cover
        env = {"error": str(e)}

    try:
        driver = subprocess.run(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
            shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        driver = None

    manifest = {
        "schema": "gen1_production_launch_manifest_v1",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": {
            "git_sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": _git("status", "--porcelain").splitlines(),
            "config_sha256": config_hash,
        },
        "dataset": {
            "root": data_root,
            "fingerprint": fingerprint,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            **{k: v for k, v in env.items()},
            "nvidia_driver": driver,
        },
        "protocol": {
            "stage_sequence": list(FOUNDATION_PHASES),
            "seed_policy": {
                "training_seed": cfg.seed,
                "eval_seeds": list(cfg.eval_seeds),
                "note": "eval subsets drawn per seed via torch.randperm(seed)",
            },
            "optimizer": {
                "type": "SGD(momentum)",
                "base_lr": cfg.lr,
                "momentum": cfg.momentum,
                "weight_decay": cfg.weight_decay,
                "scheduler": "CosineAnnealingLR(T_max=epochs)",
                "groups": "backbone, classifier, evidential_head, predictor, "
                          "update_net, precision, gaze_policy(steps 5-6)",
            },
            "recurrence": {
                "num_glimpses_T": cfg.num_glimpses,
                "within_glimpse_iters": cfg.within_glimpse_iters,
                "fovea_size": cfg.fovea_size,
                "img_size": cfg.img_size,
            },
            "gaze": {
                "steps_1_4": gaze_scheme_for_phase("belief_with_f"),
                "steps_5_6": gaze_scheme_for_phase("gen1_core"),
                "num_candidates_K": cfg.num_candidates,
            },
            "uncertainty": {
                "carrier": "Agent D EvidentialHead Dirichlet (the ONE U_t)",
                "S_t": None,
                "L_stab": "diagnostic-only (not a training objective)",
            },
            "batch_size": cfg.batch_size,
            "epochs_per_phase": cfg.epochs,
            "amp": cfg.amp,
            "eval": {
                "seeds": list(cfg.eval_seeds),
                "n_samples_per_seed": cfg.n_eval_samples,
                "eps_list": list(cfg.eps_list),
                "pgd_steps": cfg.pgd_steps,
                "harness": "Agent I run_clean_and_robust (consistency-asserted)",
            },
            "checkpoint_cadence_epochs": cfg.roll_every,
        },
        "launcher": {
            "canonical": "cloud_setup/run_j1_local.sh",
            "command": (f"J1_DATA_ROOT={data_root} "
                        f"./cloud_setup/run_j1_local.sh"),
            "overrides_applied_to_hash": {"batch_size": 48,
                                          "num_workers": 4},
        },
        "destinations": {
            "checkpoints_local": cfg.ckpt_dir,
            "reports_local": cfg.report_dir,
            "runs_local": cfg.runs_dir,
            "hf_best_repo": "FerrariKazu/rhan-nxa-checkpoints",
            "hf_rolling_repo": "FerrariKazu/rhan-nxa-checkpoints-rolling",
        },
    }

    out = os.path.join(cfg.runs_dir, "production_launch_manifest.json")
    os.makedirs(cfg.runs_dir, exist_ok=True)
    # Deliberately NOT silently overwritable: a second freeze must be a
    # conscious act (delete the old manifest audibly first).
    if os.path.exists(out):
        print(f"STOP — a frozen launch manifest already exists at {out}.\n"
              f"  Freezing twice would blur which configuration the running\n"
              f"  experiment actually uses. Delete it EXPLICITLY to re-freeze\n"
              f"  (and never freeze a second config without stopping the run).")
        return 1
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"FROZEN -> {out}")
    print(f"  git {manifest['code']['git_sha'][:12]} "
          f"config_sha256 {config_hash[:16]}…")
    print(f"  dataset {fingerprint['source']}@{fingerprint['revision'][:12]} "
          f"train={fingerprint['splits']['train']['images']} "
          f"val={fingerprint['splits']['val']['images']}")
    print("  The scientific configuration is now FROZEN (Phase 9 discipline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
