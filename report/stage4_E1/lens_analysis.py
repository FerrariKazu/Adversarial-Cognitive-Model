#!/usr/bin/env python3
"""
Stage 4-E1 Lens Analysis: D vs E1 belief drift, gaze trajectory, Pi_D
====================================================================

Runs the Lens session on D and E1 checkpoints with the same image set,
computing:
  1. Belief drift (clean vs PGD-ε) for D and E1
  2. Gaze trajectory comparison
  3. Π_D per-class trajectory (including H1b reordering check)
  4. Per-step activation hooks

Usage:
    python3 report/stage4_E1/lens_analysis.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _p in (_REPO, os.path.join(_REPO, 'phase1_training'),
           os.path.join(_REPO, 'phase2_attacks')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rhan_core.lens.session import LensSession, batch_belief_drift, run_captures

# ── Config ──────────────────────────────────────────────────────────────
D_CKPT = os.path.join(_REPO, 'checkpoints', 'rhan_next_ais_hpc_best.pth')
E1_CKPT = os.path.join(_REPO, 'checkpoints', 'rhan_next_ais_hpc_recon_best.pth')
OUTPUT_DIR = os.path.join(_REPO, 'report', 'stage4_E1', 'lens_results')
EPS = 0.094
PGD_STEPS = 50
N_IMAGES = 30  # images for belief drift analysis
SEED = 42

# STL-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_test_images(n_images=30, seed=42):
    """Load n normalized STL-10 test images."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))])

    testset = torchvision.datasets.STL10(
        os.path.join(_REPO, 'data', 'stl10'), split='test', download=True)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(testset), size=n_images, replace=False)

    images = []
    labels = []
    for idx in indices:
        img, label = testset[idx]
        images.append(transform(img))
        labels.append(label)

    return images, labels


def analyze_pi_d_trajectory(session, images, labels, step_labels):
    """Compute per-class Π_D trajectory across recurrent steps."""
    per_class_pi_d = {c: {t: [] for t in range(4)} for c in range(10)}

    for img, label in zip(images, labels):
        result = None
        for item in session.run(img, step_by_step=True, ground_truth=label):
            if hasattr(item, 'captures'):
                result = item
                break
            if hasattr(item, 'pi_d') and item.pi_d is not None:
                step_idx = item.step
                per_class_pi_d[label][step_idx].append(item.pi_d)

    # Average per class per step
    summary = {}
    for c in range(10):
        summary[CLASSES[c]] = {}
        for t in range(4):
            vals = per_class_pi_d[c][t]
            if vals:
                summary[CLASSES[c]][step_labels[t]] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)) if len(vals) > 1 else 0.0,
                    'n': len(vals)
                }
            else:
                summary[CLASSES[c]][step_labels[t]] = {'mean': None, 'std': None, 'n': 0}
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"{'═' * 60}")
    print(f"  Stage 4-E1 Lens Analysis: D vs E1")
    print(f"  Images: {N_IMAGES} | ε: {EPS} | PGD steps: {PGD_STEPS}")
    print(f"{'═' * 60}")

    # ── Load test images ────────────────────────────────────────────────
    print("\nLoading test images...")
    images, labels = load_test_images(N_IMAGES, SEED)
    print(f"  Loaded {len(images)} images")

    # ── Load sessions ───────────────────────────────────────────────────
    results = {}

    for name, ckpt_path in [("D", D_CKPT), ("E1", E1_CKPT)]:
        if not os.path.exists(ckpt_path):
            print(f"\n  Skipping {name}: checkpoint not found ({ckpt_path})")
            continue

        print(f"\n─── Loading {name} session ───")
        t0 = time.time()
        session = LensSession(
            checkpoint_path=ckpt_path,
            label=name,
            beta_base=2.5  # ε=0.094 phase convention
        )
        print(f"  Loaded in {time.time() - t0:.1f}s")
        print(f"  Config: {session.config_summary}")
        print(f"  AIS active: {session.ais_active}")
        print(f"  HPC active: {session.hpc_active}")

        # ── Belief drift ────────────────────────────────────────────────
        print(f"\n  Computing belief drift (clean vs PGD-{EPS})...")
        t0 = time.time()
        drift = batch_belief_drift(
            [session], images, eps=EPS, pgd_steps=PGD_STEPS)
        print(f"  Done in {time.time() - t0:.1f}s")

        drift_summary = drift['per_checkpoint'].get(name, {}).get('summary', {})
        print(f"  Mean drift (cosine): {drift_summary.get('mean_drift_cosine', 'N/A')}")
        print(f"  Max drift (cosine): {drift_summary.get('max_drift_cosine', 'N/A')}")
        print(f"  Mean drift (L2): {drift_summary.get('mean_drift_l2', 'N/A')}")

        # ── Π_D trajectory ──────────────────────────────────────────────
        print(f"\n  Computing Π_D per-class trajectory...")
        step_labels = [f"T={t}" for t in range(4)]
        pi_d_traj = analyze_pi_d_trajectory(session, images, labels, step_labels)

        # H1b check: truck rank in top-2 Π_D
        for t_idx, t_label in enumerate(step_labels):
            ranked = sorted(
                [(c, pi_d_traj[c][t_label]['mean'])
                 for c in CLASSES if pi_d_traj[c][t_label]['mean'] is not None],
                key=lambda x: -x[1])
            top2 = [r[0] for r in ranked[:2]]
            truck_rank = next((i + 1 for i, (c, _) in enumerate(ranked)
                               if c == 'truck'), None)
            print(f"  {t_label} Π_D top-2: {top2} | truck rank: {truck_rank}")

        results[name] = {
            'config': session.config_summary,
            'drift_summary': drift_summary,
            'drift_per_step': drift['per_checkpoint'].get(name, {}).get('rows', []),
            'pi_d_trajectory': pi_d_traj,
        }

    # ── Save results ────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, 'lens_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── D vs E1 comparison ──────────────────────────────────────────────
    if 'D' in results and 'E1' in results:
        print(f"\n{'═' * 60}")
        print(f"  D vs E1 Comparison Summary")
        print(f"{'═' * 60}")

        d_drift = results['D']['drift_summary']
        e1_drift = results['E1']['drift_summary']
        print(f"\n  Belief drift (clean → PGD-{EPS}):")
        print(f"    D  mean cosine: {d_drift.get('mean_drift_cosine', 'N/A')}")
        print(f"    E1 mean cosine: {e1_drift.get('mean_drift_cosine', 'N/A')}")

        # H1b: does E1 reproduce car/airplane top-2?
        print(f"\n  H1b Π_D reordering check:")
        for t_idx, t_label in enumerate(step_labels):
            for name in ['D', 'E1']:
                if name not in results:
                    continue
                traj = results[name]['pi_d_trajectory']
                ranked = sorted(
                    [(c, traj[c][t_label]['mean'])
                     for c in CLASSES if traj[c][t_label]['mean'] is not None],
                    key=lambda x: -x[1])
                top2 = [r[0] for r in ranked[:2]]
                print(f"    {name} {t_label} top-2: {top2}")

    print(f"\n{'═' * 60}")
    print(f"  Lens analysis complete.")
    print(f"{'═' * 60}")


if __name__ == '__main__':
    main()
