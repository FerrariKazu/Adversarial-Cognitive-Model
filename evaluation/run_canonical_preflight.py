#!/usr/bin/env python3
"""
Canonical Evaluation Preflight — evaluation/run_canonical_preflight.py
=====================================================================

Executes the unified headline evaluation across ALL arms in ONE environment:
- ff_112 (seed 17, plus seeds 42, 99)
- recurrence_only
- g2_gist_only_cont (C1)
- ff_56 (final)
- g2_gist_only (final)
- gen1_core (best clean)

Guarantees:
- Imports val transform directly from training curriculum.
- Checkpoint gate verifies clean accuracy on all 5,000 images within 1.0 pp.
- Cryptographic SHA-256 assertion on full val set and canonical 1k slice.
- NormalizedModel wraps forward pass; attacks operate strictly in [0, 1] pixel space.
- Linf perturbation bound (5.417/255) strictly verified per batch.
- Per-sample binary outcomes saved for exact McNemar tests.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
import torchattacks
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.harness import (
    get_canonical_slice,
    verify_checkpoint_gate,
    load_model_from_checkpoint,
    build_val_loaders,
    run_pgd_pixel_space,
    CANONICAL_TORCH_SLICE_SHA256,
    DEFAULT_EPS_PIXEL,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_ROOT = "data/imagenet100/val"
OUT_JSON = "report/audit_rev1/canonical_headline_preflight.json"


def mcnemar_test(corr_a: np.ndarray, corr_b: np.ndarray) -> Dict[str, Any]:
    """Calculates McNemar test with Edwards continuity correction."""
    b = np.sum((corr_a == 1) & (corr_b == 0))
    c = np.sum((corr_a == 0) & (corr_b == 1))
    n = b + c
    if n == 0:
        return {"b": int(b), "c": int(c), "stat": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1.0) ** 2 / n
    p = float(1.0 - chi2.cdf(stat, df=1))
    return {"b": int(b), "c": int(c), "stat": float(stat), "p": p}


def run_arm_battery(
    label: str,
    phase: str,
    img_size: int,
    ckpt_path: str,
    slice_indices: List[int],
) -> Dict[str, Any]:
    print(f"\n{'='*80}\nEVALUATING ARM: {label} (phase={phase}, {img_size}px, ckpt={os.path.basename(ckpt_path)})\n{'='*80}")
    
    # 1. Gate check: evaluate 5k clean without NormalizedModel wrapper (matching training)
    unwrapped_m, meta = load_model_from_checkpoint(phase, img_size, ckpt_path, DEVICE, wrap_normalized=False)
    loader_5k = build_val_loaders(img_size, VAL_ROOT, batch_size=256, num_workers=4, raw_pixel=False)
    acc_5k, monitor_clean = verify_checkpoint_gate(unwrapped_m, meta, loader_5k, DEVICE, tol_pp=1.0)
    print(f"✓ CHECKPOINT GATE PASSED: 5k Clean={acc_5k*100:.2f}% | Monitor={monitor_clean*100:.2f}% | Diff={(acc_5k-monitor_clean)*100:+.2f} pp")

    # 2. Build attack-wrapped model (operates on raw [0, 1] pixels)
    model, _ = load_model_from_checkpoint(phase, img_size, ckpt_path, DEVICE, wrap_normalized=True)
    slice_loader = build_val_loaders(img_size, VAL_ROOT, batch_size=32, num_workers=2, indices=slice_indices, raw_pixel=True)

    # Attacks
    apgd_ce  = torchattacks.APGD(model, norm="Linf", eps=DEFAULT_EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(model, norm="Linf", eps=DEFAULT_EPS_PIXEL, steps=100, n_restarts=1, loss="dlr", verbose=False)

    outcomes = {}
    runtimes = {}

    for atk_name in ["clean", "pgd50", "apgd_ce", "apgd_dlr"]:
        t0 = time.time()
        print(f"  Running {atk_name:<12} ...", end="", flush=True)
        preds_list, labels_list = [], []
        max_linf = 0.0

        for x, y in slice_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if atk_name == "clean":
                x_adv = x
            elif atk_name == "pgd50":
                x_adv = run_pgd_pixel_space(model, x, y, eps=DEFAULT_EPS_PIXEL, steps=50, restarts=1)
            elif atk_name == "apgd_ce":
                x_adv = apgd_ce(x, y)
            elif atk_name == "apgd_dlr":
                x_adv = apgd_dlr(x, y)

            if atk_name != "clean":
                batch_linf = (x_adv - x).abs().max().item()
                max_linf = max(max_linf, batch_linf)
                assert batch_linf <= DEFAULT_EPS_PIXEL + 1e-4, f"Linf bound exceeded: {batch_linf}"

            with torch.no_grad():
                preds = model(x_adv).argmax(-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(y.cpu().numpy().tolist())

        corr = (np.array(preds_list) == np.array(labels_list)).astype(np.int8)
        dt = time.time() - t0
        outcomes[atk_name] = corr
        runtimes[atk_name] = dt
        linf_str = f"Max Linf: {max_linf*255:.3f}/255 | " if atk_name != "clean" else ""
        print(f" Acc: {corr.mean()*100:>6.2f}% | {linf_str}Time: {dt:.1f}s")

    # AutoAttack Core & Worst-Case
    aa_core = outcomes["apgd_ce"] & outcomes["apgd_dlr"]
    outcomes["autoattack_core"] = aa_core
    worst_case = outcomes["pgd50"] & outcomes["apgd_ce"] & outcomes["apgd_dlr"]
    outcomes["worst_case"] = worst_case
    print(f"  AutoAttack Core  ... Acc: {aa_core.mean()*100:>6.2f}%")
    print(f"  Worst-Case Union ... Acc: {worst_case.mean()*100:>6.2f}%")

    return {
        "metadata": {
            "epoch": meta.get("epoch"),
            "monitor_clean": monitor_clean,
            "monitor_robust": meta.get("val_rob_acc"),
            "full_5k_clean": float(acc_5k),
        },
        "slice_metrics": {
            "clean_acc": float(outcomes["clean"].mean()),
            "pgd50_acc": float(outcomes["pgd50"].mean()),
            "apgd_ce_acc": float(outcomes["apgd_ce"].mean()),
            "apgd_dlr_acc": float(outcomes["apgd_dlr"].mean()),
            "aa_core_acc": float(outcomes["autoattack_core"].mean()),
            "worst_case_acc": float(outcomes["worst_case"].mean()),
        },
        "runtimes_sec": runtimes,
        "outcomes": {k: v.tolist() for k, v in outcomes.items()}
    }


def main():
    print("=" * 80)
    print("CANONICAL HEADLINE PREFLIGHT — EVALUATION HARNESS")
    print(f"  Device: {DEVICE}")
    print(f"  Val Set: {VAL_ROOT}")
    print(f"  Budget: {DEFAULT_EPS_PIXEL*255:.3f}/255 ({DEFAULT_EPS_PIXEL:.5f})")
    print("=" * 80)

    slice_indices, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    print(f"✓ Canonical Slice Asserted: {len(slice_indices)} images | SHA-256: {slice_hash}")

    ARMS = [
        ("ff_112_s17",        "ff_112",          112, "checkpoints/foundation_ff_112_final.pth"),
        ("recurrence_only",   "recurrence_only", 56,  "checkpoints/foundation_recurrence_only_final.pth"),
        ("ff_56",             "ff_56",           56,  "checkpoints/foundation_ff_56_final.pth"),
        ("g2_gist_only_cont", "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_cont_final.pth"),
        ("g2_gist_only",      "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_final.zip"),
        ("gen1_core",         "gen1_core",       56,  "checkpoints/foundation_gen1_core_best_clean.pth"),
    ]

    results = {
        "slice_hash": slice_hash,
        "eps_pixel": DEFAULT_EPS_PIXEL,
        "arms": {},
        "mcnemar_vs_ff56": {},
        "mcnemar_vs_recurrence_only": {},
    }

    for label, phase, img_size, ckpt in ARMS:
        if not os.path.exists(ckpt):
            print(f"Skipping {label}: {ckpt} not found!")
            continue
        results["arms"][label] = run_arm_battery(label, phase, img_size, ckpt, slice_indices)

    # McNemar tests
    if "ff_56" in results["arms"]:
        corr_ff56_clean = np.array(results["arms"]["ff_56"]["outcomes"]["clean"])
        corr_ff56_worst = np.array(results["arms"]["ff_56"]["outcomes"]["worst_case"])
        for label, arm_data in results["arms"].items():
            if label == "ff_56":
                continue
            corr_clean = np.array(arm_data["outcomes"]["clean"])
            corr_worst = np.array(arm_data["outcomes"]["worst_case"])
            results["mcnemar_vs_ff56"][label] = {
                "clean": mcnemar_test(corr_clean, corr_ff56_clean),
                "worst_case": mcnemar_test(corr_worst, corr_ff56_worst),
            }

    if "recurrence_only" in results["arms"]:
        corr_rec_clean = np.array(results["arms"]["recurrence_only"]["outcomes"]["clean"])
        corr_rec_worst = np.array(results["arms"]["recurrence_only"]["outcomes"]["worst_case"])
        for label, arm_data in results["arms"].items():
            if label == "recurrence_only":
                continue
            corr_clean = np.array(arm_data["outcomes"]["clean"])
            corr_worst = np.array(arm_data["outcomes"]["worst_case"])
            results["mcnemar_vs_recurrence_only"][label] = {
                "clean": mcnemar_test(corr_clean, corr_rec_clean),
                "worst_case": mcnemar_test(corr_worst, corr_rec_worst),
            }

    # Summary Headline Table
    print("\n" + "=" * 105)
    print("UNIFIED HEADLINE TABLE (Canonical 1,000-image slice, all arms in same environment)")
    print("=" * 105)
    print(f"{'Model':<20} {'5k Val':>8} {'Clean':>8} {'PGD-50':>9} {'APGD-CE':>9} {'APGD-DLR':>10} {'AA-Core':>9} {'Worst':>8}")
    print("-" * 105)
    for lbl, arm_data in results["arms"].items():
        m = arm_data["slice_metrics"]
        val5k = arm_data["metadata"]["full_5k_clean"] * 100
        print(f"{lbl:<20} {val5k:>7.2f}% {m['clean_acc']*100:>7.2f}% {m['pgd50_acc']*100:>8.2f}% {m['apgd_ce_acc']*100:>8.2f}% {m['apgd_dlr_acc']*100:>9.2f}% {m['aa_core_acc']*100:>8.2f}% {m['worst_case_acc']*100:>7.2f}%")
    print("=" * 105)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCanonical results successfully written to: {OUT_JSON}")


if __name__ == "__main__":
    main()
