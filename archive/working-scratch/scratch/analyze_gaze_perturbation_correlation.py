"""
analyze_gaze_perturbation_correlation.py
========================================
Gaze–perturbation correlation analysis for AIS-v1.

For 50–100 test images, runs LensSession on both the clean image and its
PGD-ε=0.094 perturbed version.  Extracts the per-step gaze trajectory from
both runs and computes:

  1. Gaze displacement (pixels) under attack at each step.
  2. Local perturbation magnitude |δ| at the clean-image gaze position.
  3. Pearson r and Spearman ρ between displacement and local |δ|.

The falsifiable test:
  * A genuinely "seeking" mechanism (AIS-v2) should show **positive
    correlation** — gaze displaces MORE in high-|δ| regions.
  * A mechanically iterating mechanism (AIS-v1) should show **~zero
    correlation** — displacement is uniform regardless of perturbation
    location.

Usage:
    python scratch/analyze_gaze_perturbation_correlation.py \
        [--n-images 50] [--eps 0.094] [--pgd-steps 50] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Repo-root importability.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT,
           os.path.join(_ROOT, "phase1_training"),
           os.path.join(_ROOT, "phase2_attacks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_full_epsilon_sweep as _sweep  # noqa: E402
from rhan_core.lens.session import LensSession, run_captures  # noqa: E402
from rhan_core.lens.capture import (  # noqa: E402
    gaze_perturbation_correlation,
    aggregate_gaze_correlation,
)

CKPT_AIS = os.path.join(_ROOT, "checkpoints",
                         "rhan_next_ais_v1_halting_only_best.pth")


def main():
    parser = argparse.ArgumentParser(
        description="Gaze–perturbation correlation for AIS-v1")
    parser.add_argument("--n-images", type=int, default=50,
                        help="Number of test images (default: 50)")
    parser.add_argument("--eps", type=float, default=0.094,
                        help="PGD epsilon in norm space (default: 0.094)")
    parser.add_argument("--pgd-steps", type=int, default=50,
                        help="PGD steps (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for test-set sampling (default: 42)")
    parser.add_argument("--output", type=str,
                        default="report/gaze_perturbation_correlation.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading {args.n_images} test images (seed={args.seed})...")
    xs, ys = _sweep.load_test_samples(n_samples=args.n_images, seed=args.seed)
    print(f"  Loaded {len(xs)} images, labels: {ys.tolist()}")

    print(f"Loading AIS-v1 checkpoint...")
    sess = LensSession(CKPT_AIS, arch="next", device="cpu",
                       label="AIS-v1 (halting-only)")
    print(f"  ais_active={sess.ais_active}, hpc_active={sess.hpc_active}")

    all_rows = []
    t0 = time.time()
    for idx in range(len(xs)):
        x_img = xs[idx]
        gt = int(ys[idx])

        # Clean run.
        _, clean_caps = run_captures(sess, x_img, gt)

        # PGD run.
        adv_img = sess.pgd(x_img, eps=args.eps, steps=args.pgd_steps)
        _, adv_caps = run_captures(sess, adv_img[0], gt)

        # Gaze–perturbation correlation.
        rows = gaze_perturbation_correlation(
            clean_caps, adv_caps, x_img, adv_img[0])
        all_rows.append(rows)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(xs) - idx - 1) / rate
            print(f"  [{idx + 1}/{len(xs)}] {elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(xs)} images in {elapsed:.1f}s "
          f"({len(xs) / elapsed:.2f} img/s)")

    # Aggregate.
    agg = aggregate_gaze_correlation(all_rows)

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GAZE–PERTURBATION CORRELATION — AIS-v1 (halting-only)")
    print("=" * 70)
    print(f"  Images:           {len(xs)}")
    print(f"  ε (norm space):   {args.eps}")
    print(f"  PGD steps:        {args.pgd_steps}")
    print(f"  Sample pairs:     {agg['n_samples']}")

    print("\n  ── Aggregate correlation ──")
    if agg["pearson_r"] is not None:
        sig_p = "***" if agg["pearson_p"] < 0.001 else (
            "**" if agg["pearson_p"] < 0.01 else (
            "*" if agg["pearson_p"] < 0.05 else "ns"))
        print(f"  Pearson r:        {agg['pearson_r']:+.4f}  "
              f"(p={agg['pearson_p']:.4g}) {sig_p}")
        sig_s = "***" if agg["spearman_p"] < 0.001 else (
            "**" if agg["spearman_p"] < 0.01 else (
            "*" if agg["spearman_p"] < 0.05 else "ns"))
        print(f"  Spearman ρ:       {agg['spearman_rho']:+.4f}  "
              f"(p={agg['spearman_p']:.4g}) {sig_s}")
    else:
        print("  Pearson r:        N/A (insufficient data)")

    print(f"\n  Mean gaze displacement: {agg['mean_displacement']:.2f} px")
    print(f"  Mean local |δ|:         {agg['mean_local_delta']:.6f}")

    print("\n  ── Per-step breakdown ──")
    print(f"  {'Step':<6} {'Disp (px)':<12} {'Local |δ|':<14} {'N':<6}")
    print(f"  {'-'*40}")
    for ps in agg["per_step"]:
        d_str = (f"{ps['mean_disp_px']:.2f}±{ps['std_disp_px']:.2f}"
                 if ps["std_disp_px"] is not None
                 else f"{ps['mean_disp_px']:.2f}")
        l_str = (f"{ps['mean_local_delta']:.6f}"
                 if ps["mean_local_delta"] is not None else "—")
        print(f"  T={ps['step']:<4} {d_str:<12} {l_str:<14} {ps['n']:<6}")

    # ── Interpretation ──────────────────────────────────────────────────────
    print("\n  ── Interpretation ──")
    if agg["pearson_r"] is not None:
        r = agg["pearson_r"]
        if abs(r) < 0.1:
            verdict = ("NO CORRELATION — gaze displacement is independent of "
                       "perturbation location. AIS-v1 is mechanically iterating, "
                       "not information-seeking.")
        elif r > 0.2:
            verdict = (f"POSITIVE CORRELATION (r={r:+.4f}) — gaze is drawn "
                       "toward high-|δ| regions. Some evidence of seeking "
                       "behaviour, but test at larger scale needed.")
        elif r < -0.2:
            verdict = (f"NEGATIVE CORRELATION (r={r:+.4f}) — gaze avoids "
                       "high-|δ| regions. Anti-seeking: the perturbation "
                       "repels the gaze.")
        else:
            verdict = (f"WEAK CORRELATION (r={r:+.4f}) — marginal signal, "
                       "inconclusive at this sample size.")
        print(f"  {verdict}")
    print("=" * 70)

    # ── Save JSON ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "n_images": len(xs),
        "eps": args.eps,
        "pgd_steps": args.pgd_steps,
        "seed": args.seed,
        "checkpoint": CKPT_AIS,
        "aggregate": agg,
        "per_image_rows": all_rows,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
