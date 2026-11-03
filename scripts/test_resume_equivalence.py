#!/usr/bin/env python3
"""Resume-Equivalence Test — scripts/test_resume_equivalence.py
================================================================================

PURPOSE
-------
Verify that a training run interrupted at epoch N and resumed produces
model weights, losses, and CUDA RNG state equivalent to an uninterrupted run
with EXACT equality (tolerance 0.0) under the ff_112 curriculum (TRADES + PGD-4).

Supports:
1. In-process kill/resume test
2. SIGKILL subprocess variant (hard kill via SIGKILL, fresh-process resume)
3. Infra key verification in checkpoints (scaler, rng, skipped_steps, commit_hash,
   dirty_tree, source_hashes, seed)
4. JSONL log LR logging verification (lr_backbone before scheduler.step:
   Epoch 1 = 1.666667e-05, Epoch 2 = 3.333333e-05).

USAGE
-----
  python3 scripts/test_resume_equivalence.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.train_gen2_foundation import run_gen2_phase


def run_worker_subproc(phase: str, ckpt_dir: str, report_dir: str, n_epochs: int, seed: int = 42) -> subprocess.Popen:
    """Launch training in a separate operating system subprocess."""
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--worker",
        "--phase", phase,
        "--ckpt-dir", ckpt_dir,
        "--report-dir", report_dir,
        "--n-epochs", str(n_epochs),
        "--seed", str(seed),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_resume_equivalence():
    print("=" * 70)
    print("TEST: Resume-Equivalence under ff_112 Curriculum (TRADES + PGD-4)")
    print("=" * 70)

    data_root = "data/imagenet100"
    dino_path = "checkpoints/dinov2_vits14_pretrain.pth"
    bs = 48
    nw = 2
    phase = "ff_112"
    seed = 42

    dir_uninterrupted_ckpt = "checkpoints/test_equiv_uninterrupted"
    dir_uninterrupted_report = "report/test_equiv_uninterrupted"
    dir_resumed_ckpt = "checkpoints/test_equiv_resumed"
    dir_resumed_report = "report/test_equiv_resumed"
    dir_sigkill_ckpt = "checkpoints/test_equiv_sigkill"
    dir_sigkill_report = "report/test_equiv_sigkill"

    for d in [
        dir_uninterrupted_ckpt, dir_uninterrupted_report,
        dir_resumed_ckpt, dir_resumed_report,
        dir_sigkill_ckpt, dir_sigkill_report,
    ]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # ── Step 1: Uninterrupted 2-epoch run ────────────────────────────────
    print("\n[Step 1] Uninterrupted: 2 smoke epochs continuous with seed=42 (ff_112, TRADES+PGD4)...")
    run_gen2_phase(
        phase=phase, arm="v1",
        data_root=data_root, ckpt_dir=dir_uninterrupted_ckpt,
        report_dir=dir_uninterrupted_report, dino_path=dino_path,
        batch_size=bs, num_workers=nw, smoke=True, n_epochs=2, training_seed=seed,
    )

    rolling_uninterrupted = os.path.join(
        dir_uninterrupted_ckpt, f"foundation_{phase}_rolling.pth"
    )
    state_uninterrupted = torch.load(rolling_uninterrupted, map_location="cpu", weights_only=False)
    assert state_uninterrupted["epoch"] == 2, f"Expected epoch 2, got {state_uninterrupted['epoch']}"
    print("  ✓ Uninterrupted run completed 2 epochs continuously.")

    # ── Step 2a: Interrupted + Resumed 2-epoch run (In-Process) ──────────
    print("\n[Step 2a] Interrupted+Resumed (In-Process): epoch 1 → resume → epoch 2...")
    # Epoch 1
    run_gen2_phase(
        phase=phase, arm="v1",
        data_root=data_root, ckpt_dir=dir_resumed_ckpt,
        report_dir=dir_resumed_report, dino_path=dino_path,
        batch_size=bs, num_workers=nw, smoke=True, n_epochs=1, training_seed=seed,
    )
    # Epoch 2 (resuming from epoch 1 rolling checkpoint)
    run_gen2_phase(
        phase=phase, arm="v1",
        data_root=data_root, ckpt_dir=dir_resumed_ckpt,
        report_dir=dir_resumed_report, dino_path=dino_path,
        batch_size=bs, num_workers=nw, smoke=True, n_epochs=2, training_seed=seed,
    )

    rolling_resumed = os.path.join(
        dir_resumed_ckpt, f"foundation_{phase}_rolling.pth"
    )
    state_resumed = torch.load(rolling_resumed, map_location="cpu", weights_only=False)
    assert state_resumed["epoch"] == 2, f"Expected epoch 2, got {state_resumed['epoch']}"
    print("  ✓ In-process resumed run completed 2 epochs.")

    # ── Step 2b: SIGKILL Subprocess Variant (Hard Kill + Fresh Process) ─
    print("\n[Step 2b] SIGKILL Subprocess Variant: process 1 trained ep 1 -> hard SIGKILL -> fresh process resumes...")
    # Launch subproc for epoch 1
    p1 = run_worker_subproc(phase, dir_sigkill_ckpt, dir_sigkill_report, n_epochs=1, seed=seed)
    out1, _ = p1.communicate()
    assert p1.returncode == 0, f"Subproc 1 failed:\n{out1}"
    assert os.path.exists(os.path.join(dir_sigkill_ckpt, f"foundation_{phase}_rolling.pth")), "Subproc 1 did not write rolling ckpt!"

    # Test actual SIGKILL handling: launch a subproc with 2 epochs, kill it with SIGKILL
    p_to_kill = run_worker_subproc(phase, dir_sigkill_ckpt, dir_sigkill_report, n_epochs=2, seed=seed)
    time.sleep(1.0)
    p_to_kill.send_signal(signal.SIGKILL)
    p_to_kill.wait()
    print("  ✓ Sent SIGKILL to active worker subprocess. Worker terminated.")

    # Launch fresh subproc to resume cleanly to epoch 2
    p2 = run_worker_subproc(phase, dir_sigkill_ckpt, dir_sigkill_report, n_epochs=2, seed=seed)
    out2, _ = p2.communicate()
    assert p2.returncode == 0, f"Subproc 2 failed:\n{out2}"

    rolling_sigkill = os.path.join(
        dir_sigkill_ckpt, f"foundation_{phase}_rolling.pth"
    )
    state_sigkill = torch.load(rolling_sigkill, map_location="cpu", weights_only=False)
    assert state_sigkill["epoch"] == 2, f"Expected epoch 2, got {state_sigkill['epoch']}"
    print("  ✓ Fresh-process resumed run completed 2 epochs after SIGKILL.")

    # ── Step 3: Exact Weight Comparison (Tolerance 0.0) ──────────────────
    print("\n[Step 3] Asserting EXACT equality for model weights (tolerance 0.0)...")
    sd_unint = state_uninterrupted["model"]
    sd_resum = state_resumed["model"]
    sd_sigkill = state_sigkill["model"]

    assert set(sd_unint.keys()) == set(sd_resum.keys()) == set(sd_sigkill.keys()), "State dict keys mismatch!"

    max_diff_inproc = 0.0
    max_diff_sigkill = 0.0
    for k in sd_unint:
        d_in = (sd_unint[k] - sd_resum[k]).abs().max().item()
        d_sk = (sd_unint[k] - sd_sigkill[k]).abs().max().item()
        max_diff_inproc = max(max_diff_inproc, d_in)
        max_diff_sigkill = max(max_diff_sigkill, d_sk)

    print(f"  Max weight diff (uninterrupted vs in-process resumed): {max_diff_inproc:.2e}")
    print(f"  Max weight diff (uninterrupted vs SIGKILL fresh resumed): {max_diff_sigkill:.2e}")
    assert max_diff_inproc == 0.0, f"In-process resumed weights mismatch: max diff = {max_diff_inproc}"
    assert max_diff_sigkill == 0.0, f"SIGKILL resumed weights mismatch: max diff = {max_diff_sigkill}"
    print("  ✓ EXACT model weight match (tolerance 0.00e+00) achieved across all parameters!")

    # ── Step 4: Compare CUDA RNG states ──────────────────────────────────
    print("\n[Step 4] Asserting CUDA RNG state equality...")
    if torch.cuda.is_available():
        cuda_unint = state_uninterrupted["rng"]["cuda"]
        cuda_resum = state_resumed["rng"]["cuda"]
        cuda_sigkill = state_sigkill["rng"]["cuda"]
        assert len(cuda_unint) == len(cuda_resum) == len(cuda_sigkill), "CUDA RNG list length mismatch!"
        for idx in range(len(cuda_unint)):
            diff_cuda_in = (cuda_unint[idx] != cuda_resum[idx]).sum().item()
            diff_cuda_sk = (cuda_unint[idx] != cuda_sigkill[idx]).sum().item()
            assert diff_cuda_in == 0, f"CUDA device {idx} RNG state mismatch (inproc)"
            assert diff_cuda_sk == 0, f"CUDA device {idx} RNG state mismatch (sigkill)"
        print("  ✓ CUDA RNG states match byte-for-byte across all runs!")

    # ── Step 5: Check Infra Keys in Checkpoint ───────────────────────────
    print("\n[Step 5] Verifying infra keys in checkpoint...")
    required_keys = [
        "scaler", "rng", "cumulative_skipped_steps",
        "commit_hash", "dirty_tree", "source_hashes", "seed",
    ]
    for k in required_keys:
        assert k in state_resumed, f"Missing infra key '{k}' in state_resumed"
        assert k in state_sigkill, f"Missing infra key '{k}' in state_sigkill"
        print(f"  ✓ Key '{k}' verified present: {repr(state_resumed[k])[:60]}")

    # Verify source hashes content
    src_hashes = state_resumed["source_hashes"]
    assert isinstance(src_hashes, dict), "source_hashes must be dict"
    for expected_src in ["training/train_gen2_foundation.py", "gen2_foundation/model.py", "gen2_foundation/backbone.py"]:
        assert expected_src in src_hashes, f"Missing hash for {expected_src}"
        print(f"  ✓ Source hash verified for {expected_src}: {src_hashes[expected_src][:16]}...")

    # ── Step 6: Verify JSONL log LR logging & Loss Exactness ─────────────
    print("\n[Step 6] Verifying JSONL log fields, LR logging, and loss exactness...")
    jsonl_unint = os.path.join(dir_uninterrupted_report, "gen2_epoch_log.jsonl")
    jsonl_resumed = os.path.join(dir_resumed_report, "gen2_epoch_log.jsonl")
    jsonl_sigkill = os.path.join(dir_sigkill_report, "gen2_epoch_log.jsonl")

    with open(jsonl_unint, "r") as f:
        recs_unint = [json.loads(line) for line in f if line.strip()]
    with open(jsonl_resumed, "r") as f:
        recs_resumed = [json.loads(line) for line in f if line.strip()]
    with open(jsonl_sigkill, "r") as f:
        recs_sigkill = [json.loads(line) for line in f if line.strip()]

    assert len(recs_unint) == len(recs_resumed) == len(recs_sigkill) == 2, "Expected 2 log records per run"

    expected_lrs = [5.0e-5 / 3.0, 10.0e-5 / 3.0]  # Ep 1: 1.666667e-05, Ep 2: 3.333333e-05

    for ep_i in range(2):
        ep_num = ep_i + 1
        u = recs_unint[ep_i]
        r = recs_resumed[ep_i]
        s = recs_sigkill[ep_i]

        print(f"  Epoch {ep_num}:")
        print(f"    lr_backbone: {u['lr_backbone']:.6e} (expected: {expected_lrs[ep_i]:.6e})")
        print(f"    loss_total:  unint={u['loss_total']:.6f}, resumed={r['loss_total']:.6f}, sigkill={s['loss_total']:.6f}")

        # Assert LR logged before scheduler.step() matches expected formula
        assert abs(u["lr_backbone"] - expected_lrs[ep_i]) < 1e-10, f"Epoch {ep_num} LR mismatch with formula!"
        assert abs(r["lr_backbone"] - expected_lrs[ep_i]) < 1e-10, f"Epoch {ep_num} resumed LR mismatch!"
        assert abs(s["lr_backbone"] - expected_lrs[ep_i]) < 1e-10, f"Epoch {ep_num} sigkill LR mismatch!"

        # Assert EXACT total loss equality
        assert u["loss_total"] == r["loss_total"] == s["loss_total"], f"Epoch {ep_num} loss not bit-identical!"
        assert u["val_acc_clean"] == r["val_acc_clean"] == s["val_acc_clean"], f"Epoch {ep_num} clean val mismatch!"

    print("  ✓ Exact loss, accuracy, and LR matches confirmed!")

    # ── Cleanup ──────────────────────────────────────────────────────────
    for d in [
        dir_uninterrupted_ckpt, dir_uninterrupted_report,
        dir_resumed_ckpt, dir_resumed_report,
        dir_sigkill_ckpt, dir_sigkill_report,
    ]:
        if os.path.exists(d):
            shutil.rmtree(d)

    print("\n" + "=" * 70)
    print("✓ ALL RESUME EQUIVALENCE AND INFRA GATE CHECKS PASSED (TOLERANCE 0.0)")
    print("=" * 70)


def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", type=str, default="ff_112")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--n-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_gen2_phase(
        phase=args.phase,
        arm="v1",
        data_root="data/imagenet100",
        ckpt_dir=args.ckpt_dir,
        report_dir=args.report_dir,
        dino_path="checkpoints/dinov2_vits14_pretrain.pth",
        batch_size=48,
        num_workers=2,
        smoke=True,
        n_epochs=args.n_epochs,
        training_seed=args.seed,
    )


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
    else:
        test_resume_equivalence()
