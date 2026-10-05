#!/usr/bin/env python3
"""Per-class HPC diagnosis (Stage 2, 2026-08-15) — Step 1 of the HPC-specific
diagnosis plan.

Question: the HPC-only smoke's Π_D reordering (car/airplane top-2, truck
3rd) — does it come with a class-specific HPC prediction problem? Two parts:

  PART A (extractor-only, fast, STL-10 test): per ground-truth class, the
  edge-map TARGET statistics the HPC head must predict:
      E[t]         mean rescaled target ([-1,1])
      E[t^2]       predict-zero MSE floor (what a zero-predicting head scores)
      Var(t)       predict-mean floor
      edge_frac    fraction of pixels with Sobel magnitude > 0.1
      strong_frac  fraction with magnitude > 0.5 (edge-dense)
  Hypothesis under test: truck's large flat cargo-bed regions yield SPARSER
  edge maps -> lower information content. Note a sparser target is EASIER
  (lower floor), so "low info content" is not by itself an error driver.

  PART B (model path, HPC checkpoint, per-class-balanced sample): per-class
  ACTUAL HPC prediction error (mean over steps+samples), per-class
  classification CE + accuracy, and the w_hpc*L_hpc vs L_trades competing-
  signal ratio per class (Step 3 gradient-budget proxy — loss magnitudes,
  not true gradient norms, labeled as such).

NOTE ON THE CHECKPOINT: the local v3 HPC smoke is the pre-optimizer-fix run
whose head sat at the predict-zero floor, so PART B actual error ~= PART A
floor. The post-fix head (v4) lives on HF; pass --ckpt when it is local.

Usage:  python3 scratch/diag_hpc_perclass.py [--ckpt checkpoints/...pth]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'phase1_training')))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from rhan_core.predictive_coding.feature_targets import EdgeMapExtractor

NORM_MEAN = (0.4467, 0.4398, 0.4066)
NORM_STD = (0.2603, 0.2566, 0.2713)
FOVEA, IMG = 48, 96
CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
           'dog', 'horse', 'monkey', 'ship', 'truck']

# Gaze positions spanning the image the way _forage's loop does (starts near
# center, moves within [-0.9, 0.9]) — same grid as edge_map_floor_analysis.py.
POSITIONS = [(0.0, 0.0),
             (-0.75, -0.75), (-0.75, 0.0), (-0.75, 0.75),
             (0.0, -0.75), (0.0, 0.75),
             (0.75, -0.75), (0.75, 0.0), (0.75, 0.75)]


def foveal_crops(x: torch.Tensor, positions) -> torch.Tensor:
    """(N,3,96,96) -> (N*P,3,48,48) bilinear crops, pos-major ordering."""
    B = x.shape[0]
    out = []
    scale = FOVEA / IMG
    for ax, ay in positions:
        a = torch.tensor([[ax, ay]], dtype=x.dtype, device=x.device).expand(B, 2)
        scale_col = torch.full((B, 1), scale, dtype=x.dtype, device=x.device)
        zero_col = torch.zeros((B, 1), dtype=x.dtype, device=x.device)
        row0 = torch.cat([scale_col, zero_col, a[:, 0:1]], dim=1)
        row1 = torch.cat([zero_col, scale_col, a[:, 1:2]], dim=1)
        theta = torch.stack([row0, row1], dim=1)
        grid = F.affine_grid(theta, (B, 3, FOVEA, FOVEA), align_corners=False)
        out.append(F.grid_sample(x, grid, mode='bilinear',
                                 padding_mode='border', align_corners=False))
    return torch.cat(out, dim=0)                          # (N*P, 3, 48, 48)


def balanced_indices(ds, n_per_class, seed=0):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(10):
        ci = [i for i in range(len(ds)) if ds[i][1] == c]
        idx.extend(rng.choice(ci, min(n_per_class, len(ci)), replace=False))
    return np.array(idx)


def part_a_target_stats(ds, n_per_class, batch=64):
    """Per-class edge-map target statistics (extractor-only)."""
    idx = balanced_indices(ds, n_per_class)
    extractor = EdgeMapExtractor().eval()
    acc = {c: {'n': 0, 't': 0.0, 't2': 0.0, 'edge': 0.0, 'strong': 0.0}
           for c in range(10)}
    n_pix = 0
    with torch.no_grad():
        for i0 in range(0, len(idx), batch):
            sel = [int(j) for j in idx[i0:i0 + batch]]
            x = torch.stack([ds[j][0] for j in sel])       # (B,3,96,96)
            labs = torch.tensor([ds[j][1] for j in sel])
            crops = foveal_crops(x, POSITIONS)             # (B*P,3,48,48)
            raw = extractor(crops)                         # (B*P,1,48,48) [0,1]
            tgt = raw * 2.0 - 1.0                          # [-1,1]
            labs_rep = labs.repeat(len(POSITIONS))         # pos-major
            tflat = tgt.view(tgt.shape[0], -1)
            sflat = raw.view(raw.shape[0], -1)
            for c in range(10):
                m = (labs_rep == c)
                if not m.any():
                    continue
                tsel = tflat[m]
                ssel = sflat[m]
                acc[c]['n'] += int(m.sum())
                acc[c]['t'] += float(tsel.sum())
                acc[c]['t2'] += float((tsel ** 2).sum())
                acc[c]['edge'] += float((ssel > 0.1).sum())
                acc[c]['strong'] += float((ssel > 0.5).sum())
            n_pix += tflat.numel()
    rows = {}
    for c in range(10):
        a = acc[c]
        n = max(a['n'], 1) * FOVEA * FOVEA
        e_t = a['t'] / n
        e_t2 = a['t2'] / n
        rows[c] = {
            'n_crops': a['n'],
            'E[t]': e_t,
            'E[t^2]': e_t2,                       # predict-zero floor
            'Var(t)': e_t2 - e_t ** 2,            # predict-mean floor
            'edge_frac': a['edge'] / n,
            'strong_frac': a['strong'] / n,
        }
    return rows


def part_b_model_path(ckpt_path, ds, n_per_class, batch=8):
    """Per-class actual HPC error + CE + acc through the real model."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = RHANNextConfig.from_dict(ck['config'])
    m = RHANNext(config=cfg).eval()
    missing, unexpected = m.load_state_dict(ck['model'], strict=False)
    print(f"  model: {cfg} | load missing={len(missing)} "
          f"unexpected={len(unexpected)}")

    idx = balanced_indices(ds, n_per_class, seed=1)
    per = {c: {'n': 0, 'hpc': 0.0, 'ce': 0.0, 'acc': 0} for c in range(10)}
    with torch.no_grad():
        for i0 in range(0, len(idx), batch):
            sel = [int(j) for j in idx[i0:i0 + batch]]
            x = torch.stack([ds[j][0] for j in sel])
            labs = torch.tensor([ds[j][1] for j in sel])
            logits, traj = m(x, return_trajectory=True)
            errs = torch.stack(traj['hpc_errors'])         # (T,B)
            err_mean = errs.mean(dim=0)                    # (B,)
            logp = F.log_softmax(logits, dim=-1)
            ce = -logp[torch.arange(logits.shape[0]), labs]
            for c in range(10):
                mk = (labs == c)
                if not mk.any():
                    continue
                per[c]['n'] += int(mk.sum())
                per[c]['hpc'] += float(err_mean[mk].sum())
                per[c]['ce'] += float(ce[mk].sum())
                per[c]['acc'] += int((logits.argmax(-1) == labs)[mk].sum())
    out = {}
    for c in range(10):
        p = per[c]
        n = max(p['n'], 1)
        out[c] = {'n': p['n'],
                  'hpc_err': p['hpc'] / n,
                  'ce': p['ce'] / n,
                  'acc': p['acc'] / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/stl10')
    ap.add_argument('--ckpt',
                    default='checkpoints/rhan_next_hpc_only_smoke_v3_best.pth')
    ap.add_argument('--n-per-class-a', type=int, default=400,
                    help='images/class for the extractor-only floor (Part A)')
    ap.add_argument('--n-per-class-b', type=int, default=40,
                    help='images/class for the model-path errors (Part B)')
    args = ap.parse_args()

    torch.set_num_threads(max(4, os.cpu_count() or 4))
    dev = torch.device('cpu')

    ds = torchvision.datasets.STL10(
        args.root, split='test', download=False,
        transform=T.Compose([T.ToTensor(),
                             T.Normalize(NORM_MEAN, NORM_STD)]))

    print(f"PART A — per-class edge-map TARGET statistics "
          f"({args.n_per_class_a}/class, {len(POSITIONS)} gaze positions)")
    print("=" * 84)
    fa = part_a_target_stats(ds, args.n_per_class_a)
    print(f"  {'class':<10}{'crops':>8}{'E[t]':>8}{'E[t^2]':>9}"
          f"{'Var(t)':>9}{'edge%':>8}{'strong%':>9}   floor rank")
    rank = sorted(range(10), key=lambda c: -fa[c]['E[t^2]'])
    for c in rank:
        r = fa[c]
        tag = ' ◄' if CLASSES[c] in ('car', 'truck') else ''
        print(f"  {CLASSES[c]:<10}{r['n_crops']:>8}{r['E[t]']:>8.4f}"
              f"{r['E[t^2]']:>9.4f}{r['Var(t)']:>9.4f}"
              f"{100*r['edge_frac']:>7.2f}%{100*r['strong_frac']:>8.2f}%"
              f"{rank.index(c) + 1:>7}{tag}")
    for a, b in [('truck', 'car'), ('truck', 'airplane')]:
        d = fa[9]['E[t^2]'] - fa[CLASSES.index(b)]['E[t^2]']
        print(f"  >>> truck vs {b}: predict-zero floor delta = {d:+.4f} "
              f"({100*d/fa[CLASSES.index(b)]['E[t^2]']:+.1f}% of {b}'s)")

    print()
    print(f"PART B — per-class ACTUAL error through the model "
          f"({args.n_per_class_b}/class, checkpoint {os.path.basename(args.ckpt)})")
    print("=" * 84)
    fb = part_b_model_path(args.ckpt, ds, args.n_per_class_b)
    w_hpc = 0.10
    print(f"  {'class':<10}{'n':>5}{'hpc_err':>10}{'floor':>9}{'err/floor':>10}"
          f"{'CE':>8}{'acc%':>7}{'w_hpc*Lhpc/CE':>13}")
    for c in sorted(range(10), key=lambda c: -fb[c]['hpc_err']):
        b = fb[c]
        ratio = (w_hpc * b['hpc_err']) / max(b['ce'], 1e-9)
        tag = ' ◄' if CLASSES[c] in ('car', 'truck') else ''
        print(f"  {CLASSES[c]:<10}{b['n']:>5}{b['hpc_err']:>10.4f}"
              f"{fa[c]['E[t^2]']:>9.4f}{b['hpc_err']/max(fa[c]['E[t^2]'],1e-9):>10.3f}"
              f"{b['ce']:>8.3f}{100*b['acc']:>6.1f}%{ratio:>13.3f}{tag}")
    for a, b in [('truck', 'car'), ('truck', 'airplane')]:
        i, j = 9, CLASSES.index(b)
        print(f"  >>> truck vs {b}: actual hpc_err delta = "
              f"{fb[i]['hpc_err'] - fb[j]['hpc_err']:+.4f}; "
              f"w_hpc*Lhpc/CE = {w_hpc*fb[i]['hpc_err']/max(fb[i]['ce'],1e-9):.3f}"
              f" vs {w_hpc*fb[j]['hpc_err']/max(fb[j]['ce'],1e-9):.3f}")
    print("=" * 84)
    print("  NOTE: v3 is the pre-optimizer-fix smoke (head at the predict-zero")
    print("  floor), so hpc_err ~= floor above. Re-run with --ckpt pointed at")
    print("  the v4 checkpoint (HF) to see the LEARNED per-class errors.")


if __name__ == '__main__':
    main()
