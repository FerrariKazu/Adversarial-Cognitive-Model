#!/usr/bin/env python3
"""
scratch/eval_lens_e1.py — Lens introspection for Stage 4-E1 (recon-mod).
=========================================================================

Runs both D (rhan_next_ais_hpc) and E1 (rhan_next_ais_hpc_recon) on
a set of STL-10 test images (clean + PGD-ε=0.094 adversarial), and
reports per-step perception metrics:

  1. Π_D trajectory (clean + adversarial)
  2. Belief drift (cosine distance between clean/adv belief states)
  3. Gaze trajectory (positions + displacement under attack)
  4. Reconstruction error (generative prior MSE, clean + adversarial)
  5. HPC error (prediction error per step)
  6. Halting (continuation probability per step)
  7. Per-step behavior summary (gate alpha, uncertainty, error magnitude)

Usage:
    python3 scratch/eval_lens_e1.py [--n-images 20] [--seed 42] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Repo-root importability
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "phase1_training"),
           os.path.join(_ROOT, "phase2_attacks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rhan_core.lens import (
    LensSession, StepCapture, ForwardResult,
    compute_belief_drift, belief_drift_summary,
    run_captures,
)
import eval_full_epsilon_sweep as _sweep


# ── Checkpoint paths ──────────────────────────────────────────────────────────
D_CKPT = os.path.join(_ROOT, "checkpoints", "rhan_next_ais_hpc_best.pth")
E1_CKPT = os.path.join(_ROOT, "checkpoints", "rhan_next_ais_hpc_recon_best.pth")


def load_test_images(n: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load n normalized STL-10 test images."""
    xs, ys = _sweep.load_test_samples(n_samples=n, seed=seed)
    return xs, ys


def run_session_batch(
    session: LensSession,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 0.094,
    pgd_steps: int = 100,
) -> Dict[str, Any]:
    """Run a session on a batch of images (clean + adversarial), return
    aggregated per-step metrics."""
    n = images.shape[0]
    device = session.device

    # Per-step accumulators
    step_data: Dict[str, List[List[float]]] = defaultdict(lambda: defaultdict(list))

    clean_caps_all: List[List[StepCapture]] = []
    adv_caps_all: List[List[StepCapture]] = []
    clean_results: List[ForwardResult] = []
    adv_results: List[ForwardResult] = []

    for i in range(n):
        img = images[i]
        gt = int(labels[i])

        # Clean run
        res_c, caps_c = run_captures(session, img, ground_truth=gt)
        clean_caps_all.append(caps_c)
        clean_results.append(res_c)

        # Adversarial run
        adv_img = session.pgd(img, eps=eps, steps=pgd_steps)
        res_a, caps_a = run_captures(session, adv_img[0], ground_truth=gt)
        adv_caps_all.append(caps_a)
        adv_results.append(res_a)

        # Collect per-step metrics
        for step_i in range(max(len(caps_c), len(caps_a))):
            c = caps_c[step_i] if step_i < len(caps_c) else None
            a = caps_a[step_i] if step_i < len(caps_a) else None

            if c is not None:
                if c.pi_d is not None:
                    step_data[step_i]["pi_d_clean"].append(c.pi_d)
                if c.recon_error is not None:
                    step_data[step_i]["recon_clean"].append(c.recon_error)
                if c.hpc_error is not None:
                    step_data[step_i]["hpc_clean"].append(c.hpc_error)
                if c.continuation is not None:
                    step_data[step_i]["cont_clean"].append(c.continuation)
                if c.gate_alpha is not None:
                    step_data[step_i]["gate_alpha_clean"].append(c.gate_alpha)
                if c.uncertainty is not None:
                    step_data[step_i]["uncertainty_clean"].append(c.uncertainty)
                if c.error_mag is not None:
                    step_data[step_i]["error_mag_clean"].append(c.error_mag)

            if a is not None:
                if a.pi_d is not None:
                    step_data[step_i]["pi_d_adv"].append(a.pi_d)
                if a.recon_error is not None:
                    step_data[step_i]["recon_adv"].append(a.recon_error)
                if a.hpc_error is not None:
                    step_data[step_i]["hpc_adv"].append(a.hpc_error)
                if a.continuation is not None:
                    step_data[step_i]["cont_adv"].append(a.continuation)
                if a.gate_alpha is not None:
                    step_data[step_i]["gate_alpha_adv"].append(a.gate_alpha)
                if a.uncertainty is not None:
                    step_data[step_i]["uncertainty_adv"].append(a.uncertainty)
                if a.error_mag is not None:
                    step_data[step_i]["error_mag_adv"].append(a.error_mag)

    # Belief drift
    drift_summaries = []
    for caps_c, caps_a in zip(clean_caps_all, adv_caps_all):
        ds = belief_drift_summary(caps_c, caps_a)
        drift_summaries.append(ds)

    # Per-step aggregation
    max_step = max(step_data.keys()) + 1 if step_data else 0
    per_step = []
    for s in range(max_step):
        row = {"step": s}
        for key, vals in step_data[s].items():
            arr = np.array(vals)
            row[f"{key}_mean"] = float(arr.mean())
            row[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            row[f"{key}_n"] = len(arr)
        per_step.append(row)

    # Aggregate belief drift
    cos_all = [ds["mean_drift_cosine"] for ds in drift_summaries
               if ds["mean_drift_cosine"] is not None]
    l2_all = [ds["mean_drift_l2"] for ds in drift_summaries
              if ds["mean_drift_l2"] is not None]

    # Clean/adv accuracy
    clean_correct = sum(1 for r in clean_results if r.correct)
    adv_correct = sum(1 for r in adv_results if r.correct)

    return {
        "per_step": per_step,
        "n_images": n,
        "clean_acc": clean_correct / n * 100,
        "adv_acc": adv_correct / n * 100,
        "belief_drift": {
            "mean_cosine": float(np.mean(cos_all)) if cos_all else None,
            "std_cosine": float(np.std(cos_all, ddof=1)) if len(cos_all) > 1 else None,
            "mean_l2": float(np.mean(l2_all)) if l2_all else None,
            "std_l2": float(np.std(l2_all, ddof=1)) if len(l2_all) > 1 else None,
            "n_images": len(cos_all),
        },
        "steps_effective_clean": [r.steps_effective for r in clean_results],
        "steps_effective_adv": [r.steps_effective for r in adv_results],
        "frac_halting_clean": [r.frac_halting for r in clean_results if r.frac_halting is not None],
        "frac_halting_adv": [r.frac_halting for r in adv_results if r.frac_halting is not None],
    }


def print_per_step_table(per_step: List[Dict], label: str) -> None:
    """Pretty-print per-step metrics."""
    print(f"\n{'='*90}")
    print(f"  PER-STEP METRICS: {label}")
    print(f"{'='*90}")

    header = (
        f"{'Step':>4} | "
        f"{'Π_D c':>7} {'Π_D a':>7} | "
        f"{'Recon c':>8} {'Recon a':>8} | "
        f"{'HPC c':>8} {'HPC a':>8} | "
        f"{'Cont c':>7} {'Cont a':>7} | "
        f"{'Gateα c':>7} {'Gateα a':>7} | "
        f"{'Uncert c':>8} {'Uncert a':>8}"
    )
    print(header)
    print("-" * 90)

    for row in per_step:
        s = row["step"]
        def v(key, fmt=".4f"):
            return f"{row.get(key, 0.0):{fmt}}" if f"{key}_n" in row and row[f"{key}_n"] > 0 else "  —   "

        print(
            f"{s:4d} | "
            f"{v('pi_d_clean', '.3f'):>7} {v('pi_d_adv', '.3f'):>7} | "
            f"{v('recon_clean', '.4f'):>8} {v('recon_adv', '.4f'):>8} | "
            f"{v('hpc_clean', '.4f'):>8} {v('hpc_adv', '.4f'):>8} | "
            f"{v('cont_clean', '.3f'):>7} {v('cont_adv', '.3f'):>7} | "
            f"{v('gate_alpha_clean', '.3f'):>7} {v('gate_alpha_adv', '.3f'):>7} | "
            f"{v('uncertainty_clean', '.4f'):>8} {v('uncertainty_adv', '.4f'):>8}"
        )


def main():
    parser = argparse.ArgumentParser(description="Lens analysis for E1")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of test images to analyze")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for test image selection")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--pgd-steps", type=int, default=100,
                        help="PGD steps for adversarial images")
    parser.add_argument("--eps", type=float, default=0.094,
                        help="PGD epsilon in norm space")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[WARN] CUDA not available, falling back to CPU")

    print(f"\n{'='*90}")
    print(f"  LENS ANALYSIS — Stage 4-E1 (recon-mod vs D)")
    print(f"  Device: {device} | Images: {args.n_images} | Seed: {args.seed}")
    print(f"  PGD: ε={args.eps}, steps={args.pgd_steps}")
    print(f"{'='*90}")

    # Load test images
    print("\nLoading test images...")
    xs, ys = load_test_images(args.n_images, seed=args.seed)
    print(f"  Loaded {xs.shape[0]} images, classes: {[int(y) for y in ys[:10]]}...")

    # Load D checkpoint
    print(f"\nLoading D checkpoint: {D_CKPT}")
    t0 = time.time()
    sess_d = LensSession(D_CKPT, device=device,
                         label="D (AIS+HPC)")
    print(f"  Loaded in {time.time()-t0:.1f}s — ais={sess_d.ais_active}, hpc={sess_d.hpc_active}")

    # Load E1 checkpoint
    print(f"\nLoading E1 checkpoint: {E1_CKPT}")
    t0 = time.time()
    sess_e1 = LensSession(E1_CKPT, device=device,
                          label="E1 (AIS+HPC+recon)")
    print(f"  Loaded in {time.time()-t0:.1f}s — ais={sess_e1.ais_active}, hpc={sess_e1.hpc_active}")

    # Run D
    print(f"\nRunning D on {args.n_images} images (clean + PGD)...")
    t0 = time.time()
    results_d = run_session_batch(sess_d, xs, ys,
                                  eps=args.eps, pgd_steps=args.pgd_steps)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Clean acc: {results_d['clean_acc']:.1f}% | Adv acc: {results_d['adv_acc']:.1f}%")

    # Run E1
    print(f"\nRunning E1 on {args.n_images} images (clean + PGD)...")
    t0 = time.time()
    results_e1 = run_session_batch(sess_e1, xs, ys,
                                   eps=args.eps, pgd_steps=args.pgd_steps)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Clean acc: {results_e1['clean_acc']:.1f}% | Adv acc: {results_e1['adv_acc']:.1f}%")

    # ── Print per-step tables ────────────────────────────────────────────────
    print_per_step_table(results_d["per_step"], "D (AIS+HPC)")
    print_per_step_table(results_e1["per_step"], "E1 (AIS+HPC+recon)")

    # ── Print summary comparison ─────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*90}")

    def fmt_drift(d):
        if d is None:
            return "—"
        return f"{d:.4f}"

    def fmt_list_mean(lst):
        if not lst:
            return "—"
        return f"{np.mean(lst):.4f}"

    print(f"\n{'Metric':<35} {'D (AIS+HPC)':<20} {'E1 (AIS+HPC+recon)':<20} {'Δ(E1-D)':<15}")
    print("-" * 90)

    # Clean accuracy
    d_clean = results_d["clean_acc"]
    e1_clean = results_e1["clean_acc"]
    print(f"{'Clean accuracy (%)':<35} {d_clean:<20.1f} {e1_clean:<20.1f} {e1_clean-d_clean:>+14.1f}")

    # Adversarial accuracy
    d_adv = results_d["adv_acc"]
    e1_adv = results_e1["adv_acc"]
    print(f"{'Adv accuracy (%)':<35} {d_adv:<20.1f} {e1_adv:<20.1f} {e1_adv-d_adv:>+14.1f}")

    # Belief drift
    d_cos = results_d["belief_drift"]["mean_cosine"]
    e1_cos = results_e1["belief_drift"]["mean_cosine"]
    print(f"{'Belief drift (cosine)':<35} {fmt_drift(d_cos):<20} {fmt_drift(e1_cos):<20} "
          f"{(e1_cos-d_cos) if d_cos and e1_cos else 0:>+14.4f}")

    d_l2 = results_d["belief_drift"]["mean_l2"]
    e1_l2 = results_e1["belief_drift"]["mean_l2"]
    print(f"{'Belief drift (L2)':<35} {fmt_drift(d_l2):<20} {fmt_drift(e1_l2):<20} "
          f"{(e1_l2-d_l2) if d_l2 and e1_l2 else 0:>+14.4f}")

    # Effective steps
    d_se = results_d["steps_effective_clean"]
    e1_se = results_e1["steps_effective_clean"]
    print(f"{'Steps effective (clean)':<35} {fmt_list_mean(d_se):<20} {fmt_list_mean(e1_se):<20}")

    d_se_a = results_d["steps_effective_adv"]
    e1_se_a = results_e1["steps_effective_adv"]
    print(f"{'Steps effective (adv)':<35} {fmt_list_mean(d_se_a):<20} {fmt_list_mean(e1_se_a):<20}")

    # Halting
    d_halt = results_d["frac_halting_clean"]
    e1_halt = results_e1["frac_halting_clean"]
    print(f"{'Frac halted (clean)':<35} {fmt_list_mean(d_halt):<20} {fmt_list_mean(e1_halt):<20}")

    d_halt_a = results_d["frac_halting_adv"]
    e1_halt_a = results_e1["frac_halting_adv"]
    print(f"{'Frac halted (adv)':<35} {fmt_list_mean(d_halt_a):<20} {fmt_list_mean(e1_halt_a):<20}")

    # Per-step comparison (final step only)
    print(f"\n{'='*90}")
    print(f"  FINAL STEP (T=3) COMPARISON")
    print(f"{'='*90}")

    d_final = results_d["per_step"][-1] if results_d["per_step"] else {}
    e1_final = results_e1["per_step"][-1] if results_e1["per_step"] else {}

    def cmp(key):
        dv = d_final.get(f"{key}_mean")
        ev = e1_final.get(f"{key}_mean")
        ds = f"{dv:.4f}" if dv is not None else "—"
        es = f"{ev:.4f}" if ev is not None else "—"
        delta = ""
        if dv is not None and ev is not None:
            delta = f"{ev-dv:>+.4f}"
        return ds, es, delta

    print(f"{'Metric':<30} {'D':<12} {'E1':<12} {'Δ':<12}")
    print("-" * 66)
    for key, label in [
        ("pi_d", "Π_D"),
        ("recon", "Recon error"),
        ("hpc", "HPC error"),
        ("cont", "Continuation"),
        ("gate_alpha", "Gate α"),
        ("uncertainty", "Uncertainty"),
        ("error_mag", "Error magnitude"),
    ]:
        ds, es, delta = cmp(key)
        print(f"{label:<30} {ds:<12} {es:<12} {delta:<12}")

    # ── Per-step Π_D trajectory table ────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  Π_D TRAJECTORY (clean)")
    print(f"{'='*90}")
    print(f"{'Step':>4} | {'D mean±std':>15} {'E1 mean±std':>15} {'Δ(E1-D)':>12}")
    print("-" * 50)
    for i, (d_row, e_row) in enumerate(zip(
        results_d["per_step"], results_e1["per_step"])):
        d_val = d_row.get("pi_d_clean_mean")
        d_std = d_row.get("pi_d_clean_std", 0)
        e_val = e_row.get("pi_d_clean_mean")
        e_std = e_row.get("pi_d_clean_std", 0)
        d_s = f"{d_val:.3f}±{d_std:.3f}" if d_val is not None else "—"
        e_s = f"{e_val:.3f}±{e_std:.3f}" if e_val is not None else "—"
        delta = f"{e_val-d_val:>+.3f}" if d_val and e_val else "—"
        print(f"{i:4d} | {d_s:>15} {e_s:>15} {delta:>12}")

    # Π_D trajectory (adversarial)
    print(f"\n{'='*90}")
    print(f"  Π_D TRAJECTORY (adversarial)")
    print(f"{'='*90}")
    print(f"{'Step':>4} | {'D mean±std':>15} {'E1 mean±std':>15} {'Δ(E1-D)':>12}")
    print("-" * 50)
    for i, (d_row, e_row) in enumerate(zip(
        results_d["per_step"], results_e1["per_step"])):
        d_val = d_row.get("pi_d_adv_mean")
        d_std = d_row.get("pi_d_adv_std", 0)
        e_val = e_row.get("pi_d_adv_mean")
        e_std = e_row.get("pi_d_adv_std", 0)
        d_s = f"{d_val:.3f}±{d_std:.3f}" if d_val is not None else "—"
        e_s = f"{e_val:.3f}±{e_std:.3f}" if e_val is not None else "—"
        delta = f"{e_val-d_val:>+.3f}" if d_val and e_val else "—"
        print(f"{i:4d} | {d_s:>15} {e_s:>15} {delta:>12}")

    # ── Reconstruction error trajectory ──────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  RECONSTRUCTION ERROR TRAJECTORY")
    print(f"{'='*90}")
    print(f"{'Step':>4} | {'D clean':>10} {'D adv':>10} | {'E1 clean':>10} {'E1 adv':>10} | {'Δ E1-D':>10}")
    print("-" * 65)
    for i, (d_row, e_row) in enumerate(zip(
        results_d["per_step"], results_e1["per_step"])):
        dc = d_row.get("recon_clean_mean")
        da = d_row.get("recon_adv_mean")
        ec = e_row.get("recon_clean_mean")
        ea = e_row.get("recon_adv_mean")
        dc_s = f"{dc:.4f}" if dc is not None else "—"
        da_s = f"{da:.4f}" if da is not None else "—"
        ec_s = f"{ec:.4f}" if ec is not None else "—"
        ea_s = f"{ea:.4f}" if ea is not None else "—"
        # Δ is E1-D on adversarial (the relevant comparison)
        delta = f"{ea-da:>+.4f}" if da is not None and ea is not None else "—"
        print(f"{i:4d} | {dc_s:>10} {da_s:>10} | {ec_s:>10} {ea_s:>10} | {delta:>10}")

    # ── HPC error trajectory ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  HPC ERROR TRAJECTORY")
    print(f"{'='*90}")
    print(f"{'Step':>4} | {'D clean':>10} {'D adv':>10} | {'E1 clean':>10} {'E1 adv':>10} | {'Δ E1-D':>10}")
    print("-" * 65)
    for i, (d_row, e_row) in enumerate(zip(
        results_d["per_step"], results_e1["per_step"])):
        dc = d_row.get("hpc_clean_mean")
        da = d_row.get("hpc_adv_mean")
        ec = e_row.get("hpc_clean_mean")
        ea = e_row.get("hpc_adv_mean")
        dc_s = f"{dc:.4f}" if dc is not None else "—"
        da_s = f"{da:.4f}" if da is not None else "—"
        ec_s = f"{ec:.4f}" if ec is not None else "—"
        ea_s = f"{ea:.4f}" if ea is not None else "—"
        delta = f"{ea-da:>+.4f}" if da is not None and ea is not None else "—"
        print(f"{i:4d} | {dc_s:>10} {da_s:>10} | {ec_s:>10} {ea_s:>10} | {delta:>10}")

    # ── Halting / continuation trajectory ─────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  HALTING / CONTINUATION TRAJECTORY")
    print(f"{'='*90}")
    print(f"{'Step':>4} | {'D clean':>10} {'D adv':>10} | {'E1 clean':>10} {'E1 adv':>10} | {'Δ E1-D':>10}")
    print("-" * 65)
    for i, (d_row, e_row) in enumerate(zip(
        results_d["per_step"], results_e1["per_step"])):
        dc = d_row.get("cont_clean_mean")
        da = d_row.get("cont_adv_mean")
        ec = e_row.get("cont_clean_mean")
        ea = e_row.get("cont_adv_mean")
        dc_s = f"{dc:.4f}" if dc is not None else "—"
        da_s = f"{da:.4f}" if da is not None else "—"
        ec_s = f"{ec:.4f}" if ec is not None else "—"
        ea_s = f"{ea:.4f}" if ea is not None else "—"
        delta = f"{ea-da:>+.4f}" if da is not None and ea is not None else "—"
        print(f"{i:4d} | {dc_s:>10} {da_s:>10} | {ec_s:>10} {ea_s:>10} | {delta:>10}")

    # ── Gate α trajectory ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  GATE α TRAJECTORY (foveal/parafoveal fusion)")
    print(f"{'='*90}")
    print(f"{'Step':>4} | {'D clean':>10} {'D adv':>10} | {'E1 clean':>10} {'E1 adv':>10} | {'Δ E1-D':>10}")
    print("-" * 65)
    for i, (d_row, e_row) in enumerate(zip(
        results_d["per_step"], results_e1["per_step"])):
        dc = d_row.get("gate_alpha_clean_mean")
        da = d_row.get("gate_alpha_adv_mean")
        ec = e_row.get("gate_alpha_clean_mean")
        ea = e_row.get("gate_alpha_adv_mean")
        dc_s = f"{dc:.4f}" if dc is not None else "—"
        da_s = f"{da:.4f}" if da is not None else "—"
        ec_s = f"{ec:.4f}" if ec is not None else "—"
        ea_s = f"{ea:.4f}" if ea is not None else "—"
        delta = f"{ea-da:>+.4f}" if da is not None and ea is not None else "—"
        print(f"{i:4d} | {dc_s:>10} {da_s:>10} | {ec_s:>10} {ea_s:>10} | {delta:>10}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    output_dir = os.path.join(_ROOT, "report", "lens_e1_analysis")
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "d": results_d,
        "e1": results_e1,
        "config": {
            "n_images": args.n_images,
            "seed": args.seed,
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "device": device,
        },
    }

    json_path = os.path.join(output_dir, "lens_e1_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  ✓ Results saved to {json_path}")

    print(f"\n{'='*90}")
    print(f"  LENS ANALYSIS COMPLETE")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
