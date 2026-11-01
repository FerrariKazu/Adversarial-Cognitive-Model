#!/usr/bin/env python3
"""
T2 Canonical Evaluation Harness — Stage 0 Closeout
====================================================
Protocol: consolidated-flagship-112/1
Prereg:   prereg/stage0_decision_v1.json

All numbers are written to:
  evaluation/stage0_results/stage0_harness_results.json   (aggregate)
  evaluation/stage0_results/<arm_id>_per_sample.npz       (per-sample arrays)

This script runs:
  1. 5k clean accuracy for every checkpoint.
  2. Checkpoint parity gate: |5k_clean - training_monitor| <= 1 pp.
  3. PGD-50 on full 5k val.
  4. APGD-CE on full 5k val.
  5. APGD-DLR on full 5k val.
  6. Worst-case = bitwise AND of all three adversarial arrays.
  7. Realised Linf <= eps assertion per batch.

Resume: already-completed arms (with .npz) are skipped.
Arm isolation: each arm is run in a fresh subprocess to guarantee GPU reset.

Usage:
  python evaluation/run_stage0_harness.py            # orchestrator mode
  python evaluation/run_stage0_harness.py --arm <id> # single arm mode (internal)

Flagship (recurrence_112) evaluated at T=4 (num_glimpses=4).
Baselines (ff_112) evaluated as-is.
"""

import os, sys, json, time, hashlib, subprocess
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchattacks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.harness import (
    get_canonical_slice,
    load_model_from_checkpoint,
    build_val_loaders,
    CANONICAL_VAL_PATH_SHA256,
    CANONICAL_TORCH_SLICE_SHA256,
    DEFAULT_EPS_PIXEL,
    NormalizedModel,
    HarnessGateError,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_ROOT = "data/imagenet100/val"
OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS_PIXEL = DEFAULT_EPS_PIXEL  # 5.417 / 255

# Pre-registration spec
ARMS = [
    {
        "arm_id": "recurrence_112_seed17",
        "phase": "recurrence_only",
        "img_size": 112,
        "ckpt": "checkpoints/foundation_recurrence_112_seed17_final_636e4b1c.pth",
    },
    {
        "arm_id": "recurrence_112_seed42",
        "phase": "recurrence_only",
        "img_size": 112,
        "ckpt": "checkpoints/foundation_recurrence_112_seed42_final_eb40bf84.pth",
    },
    {
        "arm_id": "ff_112_seed17",
        "phase": "ff_112",
        "img_size": 112,
        "ckpt": "checkpoints/foundation_ff_112_seed17_final_d00fb750.pth",
    },
    {
        "arm_id": "ff_112_seed42",
        "phase": "ff_112",
        "img_size": 112,
        "ckpt": "checkpoints/foundation_ff_112_seed42_final_2920df54.pth",
    },
    {
        "arm_id": "ff_112_seed99",
        "phase": "ff_112",
        "img_size": 112,
        "ckpt": "checkpoints/foundation_ff_112_seed99_final_abde98a0.pth",
    },
]


def get_5k_loader(img_size: int, normalize: bool, batch_size: int = 128):
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()] +
                   ([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] if normalize else []))
    ds = ImageFolder(VAL_ROOT, transform=tf)
    assert len(ds) == 5000, f"Expected 5000 val images, got {len(ds)}"
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def get_5k_raw_loader(img_size: int, batch_size: int = 64):
    """Raw [0,1] pixel space loader for attacks."""
    return get_5k_loader(img_size, normalize=False, batch_size=batch_size)


@torch.no_grad()
def eval_clean_norm(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    all_correct = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(x)
        pred = logits.argmax(dim=-1)
        all_correct.extend((pred == y).cpu().numpy().tolist())
    return np.array(all_correct, dtype=np.uint8)


def eval_attack_raw(model: nn.Module, loader: DataLoader, atk_name: str,
                    arm_id: str) -> np.ndarray:
    """Evaluate a torchattacks attack (model operates in [0,1] pixel space).
    Asserts Linf <= EPS_PIXEL per batch."""
    model.eval()

    if atk_name == "pgd50":
        atk = torchattacks.PGD(model, eps=EPS_PIXEL, alpha=EPS_PIXEL / 10,
                                steps=50, random_start=True)
    elif atk_name == "apgd_ce":
        atk = torchattacks.APGD(model, eps=EPS_PIXEL, steps=100,
                                 loss="ce", rho=0.75)
    elif atk_name == "apgd_dlr":
        atk = torchattacks.APGD(model, eps=EPS_PIXEL, steps=100,
                                 loss="dlr", rho=0.75)
    else:
        raise ValueError(f"Unknown attack: {atk_name}")

    all_correct = []
    n_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = atk(x, y)

        # Assert Linf budget
        linf = (x_adv - x).abs().max().item()
        if linf > EPS_PIXEL + 1e-5:
            raise HarnessGateError(
                f"[{arm_id}/{atk_name}] Realised Linf {linf:.6f} > eps {EPS_PIXEL:.6f}!")

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x_adv)
        pred = logits.argmax(dim=-1)
        all_correct.extend((pred == y).cpu().numpy().tolist())
        if (i + 1) % 20 == 0:
            print(f"    {atk_name} batch {i+1}/{n_batches}", flush=True)

    return np.array(all_correct, dtype=np.uint8)


def run_arm(arm: Dict[str, Any], intake_manifest: Dict) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    phase = arm["phase"]
    img_size = arm["img_size"]
    ckpt = arm["ckpt"]

    print(f"\n{'='*80}")
    print(f"ARM: {arm_id}  phase={phase}  img_size={img_size}")
    print(f"  ckpt: {ckpt}")
    print(f"{'='*80}")
    t_arm_start = time.time()

    # Retrieve training monitor values from intake manifest
    intake = intake_manifest.get(arm_id, {})
    monitor_clean = intake.get("training_monitor_clean", None)
    assert monitor_clean is not None, f"Missing intake record for {arm_id}"

    # --- Step 1: 5k clean with normalised model (matching training) ---
    model_norm, meta = load_model_from_checkpoint(phase, img_size, ckpt, DEVICE, wrap_normalized=False)
    loader_5k_norm = get_5k_loader(img_size, normalize=True, batch_size=256)
    t0 = time.time()
    corr_clean = eval_clean_norm(model_norm, loader_5k_norm)
    t_clean = time.time() - t0
    acc_5k_clean = corr_clean.mean()

    # Parity gate
    gap_pp = abs(acc_5k_clean - monitor_clean) * 100.0
    parity_pass = gap_pp <= 1.0
    print(f"  GATE: 5k_clean={acc_5k_clean*100:.2f}% | monitor={monitor_clean*100:.2f}% | gap={gap_pp:.2f}pp | {'PASS' if parity_pass else 'FAIL'}")
    if not parity_pass:
        raise HarnessGateError(f"[{arm_id}] Checkpoint parity gate FAILED: gap {gap_pp:.2f} pp > 1.0 pp")

    # --- Step 2: Attacks in [0,1] pixel space ---
    model_raw, _ = load_model_from_checkpoint(phase, img_size, ckpt, DEVICE, wrap_normalized=True)
    loader_5k_raw = get_5k_raw_loader(img_size, batch_size=32)

    # Clean accuracy via raw model (for cross-check only)
    t0 = time.time()

    attacks_correct = {}
    for atk_name in ["pgd50", "apgd_ce", "apgd_dlr"]:
        print(f"  Running {atk_name} on 5k...", flush=True)
        ta = time.time()
        attacks_correct[atk_name] = eval_attack_raw(model_raw, loader_5k_raw, atk_name, arm_id)
        print(f"    {atk_name} done: {attacks_correct[atk_name].mean()*100:.2f}% in {time.time()-ta:.1f}s")

    # Worst-case = bitwise AND
    corr_worst = (attacks_correct["pgd50"].astype(bool) &
                  attacks_correct["apgd_ce"].astype(bool) &
                  attacks_correct["apgd_dlr"].astype(bool)).astype(np.uint8)

    acc_pgd50 = float(attacks_correct["pgd50"].mean())
    acc_apgd_ce = float(attacks_correct["apgd_ce"].mean())
    acc_apgd_dlr = float(attacks_correct["apgd_dlr"].mean())
    acc_worst = float(corr_worst.mean())

    t_arm_total = time.time() - t_arm_start
    print(f"\n  RESULTS [{arm_id}]:")
    print(f"    clean_5k:    {acc_5k_clean*100:.2f}%")
    print(f"    pgd50:       {acc_pgd50*100:.2f}%")
    print(f"    apgd_ce:     {acc_apgd_ce*100:.2f}%")
    print(f"    apgd_dlr:    {acc_apgd_dlr*100:.2f}%")
    print(f"    worst_case:  {acc_worst*100:.2f}%")
    print(f"    total_time:  {t_arm_total:.1f}s", flush=True)

    # Save per-sample arrays
    npz_path = OUT_DIR / f"{arm_id}_per_sample.npz"
    np.savez_compressed(
        npz_path,
        corr_clean=corr_clean,
        corr_pgd50=attacks_correct["pgd50"],
        corr_apgd_ce=attacks_correct["apgd_ce"],
        corr_apgd_dlr=attacks_correct["apgd_dlr"],
        corr_worst=corr_worst,
    )

    return {
        "arm_id": arm_id,
        "phase": phase,
        "img_size": img_size,
        "ckpt_sha8": intake.get("sha8"),
        "epoch": intake.get("epoch"),
        "training_monitor_clean": monitor_clean,
        "gate_passed": bool(parity_pass),
        "gate_gap_pp": gap_pp,
        "acc_clean_5k": float(acc_5k_clean),
        "acc_pgd50_5k": acc_pgd50,
        "acc_apgd_ce_5k": acc_apgd_ce,
        "acc_apgd_dlr_5k": acc_apgd_dlr,
        "acc_worst_case_5k": acc_worst,
        "clean_5k_time_s": t_clean,
        "total_arm_time_s": t_arm_total,
        "eps_pixel": EPS_PIXEL,
        "n_clean": len(corr_clean),
        "n_robust": len(corr_worst),
        "per_sample_npz": str(npz_path),
    }


def arm_result_from_npz(arm: Dict[str, Any], intake_manifest: Dict) -> Dict[str, Any]:
    """Load a pre-existing .npz result (arm already completed)."""
    arm_id = arm["arm_id"]
    npz_path = OUT_DIR / f"{arm_id}_per_sample.npz"
    d = np.load(npz_path)
    intake = intake_manifest.get(arm_id, {})
    monitor_clean = intake.get("training_monitor_clean", None)
    return {
        "arm_id": arm_id,
        "phase": arm["phase"],
        "img_size": arm["img_size"],
        "ckpt_sha8": intake.get("sha8"),
        "epoch": intake.get("epoch"),
        "training_monitor_clean": monitor_clean,
        "gate_passed": True,   # Already verified in prior run
        "gate_gap_pp": None,   # Not re-run
        "acc_clean_5k": float(d["corr_clean"].mean()),
        "acc_pgd50_5k": float(d["corr_pgd50"].mean()),
        "acc_apgd_ce_5k": float(d["corr_apgd_ce"].mean()),
        "acc_apgd_dlr_5k": float(d["corr_apgd_dlr"].mean()),
        "acc_worst_case_5k": float(d["corr_worst"].mean()),
        "clean_5k_time_s": None,
        "total_arm_time_s": None,
        "eps_pixel": EPS_PIXEL,
        "n_clean": len(d["corr_clean"]),
        "n_robust": len(d["corr_worst"]),
        "per_sample_npz": str(npz_path),
        "resumed": True,
    }


def main_single_arm(arm_id: str):
    """Single-arm mode: run one arm, save .npz and a partial JSON, then exit."""
    manifest_path = Path("checkpoints/intake_manifest.json")
    with open(manifest_path) as f:
        intake_manifest = json.load(f)
    arm = next(a for a in ARMS if a["arm_id"] == arm_id)
    result = run_arm(arm, intake_manifest)
    partial = OUT_DIR / f"{arm_id}_result.json"
    with open(partial, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Single-arm done. Saved {partial}")


def main_orchestrator():
    """Orchestrator: dispatch each arm in a fresh subprocess, then assemble final JSON."""
    print("\nT2 CANONICAL HARNESS — STAGE 0 CLOSEOUT (orchestrator)")
    print(f"EPS_PIXEL: {EPS_PIXEL:.6f}  ({EPS_PIXEL*255:.3f}/255)")

    # Verify canonical hashes once
    print("\nVerifying canonical data hashes...")
    indices_1k, slice_hash = get_canonical_slice(
        val_root=VAL_ROOT, n=1000, seed=42, assert_hash=True)
    assert slice_hash == CANONICAL_TORCH_SLICE_SHA256, "Slice hash mismatch!"
    print(f"  1k slice hash: {slice_hash} — MATCH")

    manifest_path = Path("checkpoints/intake_manifest.json")
    assert manifest_path.exists(), "Run evaluation/intake_checkpoints.py first!"
    with open(manifest_path) as f:
        intake_manifest = json.load(f)

    harness_sha = hashlib.sha256(Path("evaluation/harness.py").read_bytes()).hexdigest()
    print(f"  evaluation/harness.py sha256: {harness_sha}")

    results = {}
    for arm in ARMS:
        arm_id = arm["arm_id"]
        npz_path = OUT_DIR / f"{arm_id}_per_sample.npz"
        if npz_path.exists():
            print(f"\n[RESUME] {arm_id} — .npz exists, loading cached results.")
            results[arm_id] = arm_result_from_npz(arm, intake_manifest)
            r = results[arm_id]
            print(f"  clean={r['acc_clean_5k']*100:.2f}%  worst={r['acc_worst_case_5k']*100:.2f}%")
            continue

        print(f"\n[DISPATCH] {arm_id} in fresh subprocess...")
        t0 = time.time()
        env = os.environ.copy()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--arm", arm_id],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"Arm {arm_id} subprocess failed (rc={proc.returncode}) after {elapsed:.1f}s")
        print(f"[DONE] {arm_id} in {elapsed:.1f}s")
        results[arm_id] = arm_result_from_npz(arm, intake_manifest)

    # Write aggregate results
    out_json = OUT_DIR / "stage0_harness_results.json"
    payload = {
        "protocol_id": "consolidated-flagship-112/1",
        "prereg_file": "prereg/stage0_decision_v1.json",
        "harness_sha256": harness_sha,
        "full_val_path_sha256": CANONICAL_VAL_PATH_SHA256,
        "canonical_1k_slice_sha256": slice_hash,
        "eps_pixel": EPS_PIXEL,
        "eps_pixel_255": EPS_PIXEL * 255,
        "n_clean": 5000,
        "n_robust": 5000,
        "arms": results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n\nSaved to {out_json}")
    print("\n=== SUMMARY ===")
    for arm_id, r in results.items():
        print(f"  {arm_id}: clean={r['acc_clean_5k']*100:.2f}%  worst={r['acc_worst_case_5k']*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default=None,
                        help="Run a single arm by ID (internal subprocess mode)")
    args = parser.parse_args()

    if args.arm:
        main_single_arm(args.arm)
    else:
        main_orchestrator()
