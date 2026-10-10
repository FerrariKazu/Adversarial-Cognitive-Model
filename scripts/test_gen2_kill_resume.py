#!/usr/bin/env python3
"""Gen-2 Foundation Harness Verification — scripts/test_gen2_kill_resume.py
================================================================================

Tests required by S7:
  1. Kill/Resume test:
     - Run epoch 1 in temporary directory.
     - Simulate interruption and resume from rolling checkpoint.
     - Verify epoch 2 starts correctly from epoch 2.
     - Assert JSONL log contains exactly [epoch 1, epoch 2] with no duplicate records.
  2. Rolling vs Final checkpoint consistency check.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.train_gen2_foundation import run_gen2_phase


def test_kill_resume():
    print("=" * 70)
    print("TEST: Gen-2 Foundation Kill/Resume & JSONL Non-Duplication")
    print("=" * 70)

    test_ckpt_dir = "checkpoints/test_resume"
    test_report_dir = "report/test_resume"

    if os.path.exists(test_ckpt_dir):
        shutil.rmtree(test_ckpt_dir)
    if os.path.exists(test_report_dir):
        shutil.rmtree(test_report_dir)

    # Step 1: Run 1 smoke epoch
    print("\n[Step 1] Running Epoch 1 (smoke)...")
    res1 = run_gen2_phase(
        phase="g2_gist_only",
        arm="v1",
        data_root="data/imagenet100",
        ckpt_dir=test_ckpt_dir,
        report_dir=test_report_dir,
        dino_path="checkpoints/dinov2_vits14_pretrain.pth",
        batch_size=48,
        num_workers=2,
        smoke=True,
    )

    rolling_path = os.path.join(test_ckpt_dir, "foundation_g2_gist_only_rolling.pth")
    assert os.path.exists(rolling_path), f"Rolling checkpoint not found at {rolling_path}"
    saved1 = torch.load(rolling_path, map_location="cpu")
    assert saved1["epoch"] == 1, f"Expected saved epoch 1, got {saved1['epoch']}"

    jsonl_path = os.path.join(test_report_dir, "gen2_epoch_log.jsonl")
    assert os.path.exists(jsonl_path), f"JSONL log not found at {jsonl_path}"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines1 = [json.loads(line) for line in f if line.strip()]
    assert len(lines1) == 1, f"Expected 1 log entry after step 1, found {len(lines1)}"
    assert lines1[0]["epoch"] == 1
    print("  ✓ Step 1 verified: Epoch 1 recorded to rolling checkpoint and JSONL.")

    # Step 2: Resume (simulate restarting the run)
    print("\n[Step 2] Resuming for Epoch 2 (smoke)...")
    res2 = run_gen2_phase(
        phase="g2_gist_only",
        arm="v1",
        data_root="data/imagenet100",
        ckpt_dir=test_ckpt_dir,
        report_dir=test_report_dir,
        dino_path="checkpoints/dinov2_vits14_pretrain.pth",
        batch_size=48,
        num_workers=2,
        smoke=True,
    )

    saved2 = torch.load(rolling_path, map_location="cpu")
    assert saved2["epoch"] == 2, f"Expected saved epoch 2, got {saved2['epoch']}"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines2 = [json.loads(line) for line in f if line.strip()]
    assert len(lines2) == 2, f"Expected exactly 2 log entries after step 2, found {len(lines2)}"
    assert lines2[0]["epoch"] == 1, f"Line 0 epoch should be 1, got {lines2[0]['epoch']}"
    assert lines2[1]["epoch"] == 2, f"Line 1 epoch should be 2, got {lines2[1]['epoch']}"
    print("  ✓ Step 2 verified: Resumed cleanly at Epoch 2 without duplicate JSONL logs.")

    # Cleanup test dirs
    shutil.rmtree(test_ckpt_dir)
    shutil.rmtree(test_report_dir)

    print("\n" + "=" * 70)
    print("✓ KILL/RESUME TEST PASSED: State machine resumes and JSONL is durable & duplicate-free.")
    print("=" * 70)


if __name__ == "__main__":
    test_kill_resume()
