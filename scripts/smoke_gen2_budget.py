#!/usr/bin/env python3
"""Gen-2 Budget Projection Probe — scripts/smoke_gen2_budget.py
================================================================================

Runs 1 smoke epoch on a 2k subset for each phase, measures s/epoch,
and computes projected total GPU-hours for the whole 6-phase foundation ladder.
If projected total > 30 GPU-hours, reports loudly.
"""

from __future__ import annotations

import os
import sys
import time
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.train_gen2_foundation import (
    PHASE_LADDER,
    PHASE_TOTAL_EPOCHS,
    run_gen2_phase,
)

BUDGET_CEILING_HOURS = 30.0
SUBSET_SIZE = 2000
FULL_SIZE = 126689
SCALE_FACTOR = FULL_SIZE / SUBSET_SIZE  # ~63.34x


def main():
    print("=" * 70)
    print("GEN-2 FOUNDATION BUDGET PROJECTION (Single GPU)")
    print("=" * 70)

    import shutil
    if os.path.exists("checkpoints/smoke"):
        shutil.rmtree("checkpoints/smoke")
    if os.path.exists("report/smoke"):
        shutil.rmtree("report/smoke")

    phase_timings = {}
    total_projected_hours = 0.0

    for phase in PHASE_LADDER:
        print(f"\n[Benchmarking] 1 smoke epoch on 2k subset for phase: {phase}")
        t0 = time.time()
        # Run 1 epoch smoke
        res = run_gen2_phase(
            phase=phase,
            arm="v1",
            data_root="data/imagenet100",
            ckpt_dir="checkpoints/smoke",
            report_dir="report/smoke",
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            batch_size=48,
            num_workers=4,
            smoke=True,
            center_crop_mode=False,
        )
        dt = time.time() - t0
        epochs_in_phase = PHASE_TOTAL_EPOCHS.get(phase, 15)

        # Extrapolate full dataset epoch time
        full_epoch_s = dt * SCALE_FACTOR
        phase_hours = (full_epoch_s * epochs_in_phase) / 3600.0

        phase_timings[phase] = {
            "smoke_s": dt,
            "projected_epoch_s": full_epoch_s,
            "epochs": epochs_in_phase,
            "projected_hours": phase_hours,
        }
        total_projected_hours += phase_hours
        print(f"  -> Smoke epoch: {dt:.1f}s | Full epoch est: {full_epoch_s:.1f}s | Phase total ({epochs_in_phase} eps): {phase_hours:.2f}h")

    print("\n" + "=" * 70)
    print("BUDGET PROJECTION SUMMARY:")
    for p, d in phase_timings.items():
        print(f"  {p:<16}: {d['epochs']:>2} epochs @ {d['projected_epoch_s']:>5.1f} s/ep -> {d['projected_hours']:>5.2f} GPU-hours")
    print("-" * 70)
    print(f"TOTAL PROJECTED GPU-HOURS: {total_projected_hours:.2f} hours (Budget ceiling: {BUDGET_CEILING_HOURS:.1f} hours)")
    print("=" * 70)

    if total_projected_hours > BUDGET_CEILING_HOURS:
        print(f"ALERT: Total projected hours ({total_projected_hours:.2f}h) EXCEEDS budget ceiling ({BUDGET_CEILING_HOURS:.1f}h)!")
        sys.exit(1)
    else:
        print("✓ BUDGET OK: Within 30 GPU-hour ceiling.")
        sys.exit(0)


if __name__ == "__main__":
    main()
