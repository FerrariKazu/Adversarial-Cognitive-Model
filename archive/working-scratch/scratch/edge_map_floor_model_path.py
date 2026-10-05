#!/usr/bin/env python3
"""
Model-path floor check: measure the edge-map target distribution the smoke's
HPC head actually sees — i.e. extract_target() applied to the foveal crops at
the REAL model's gaze positions (RHANNext, HPC on, base checkpoint loaded),
on held-out STL-10 test images. No training, no GPU needed.

Answers: is the observed 0.6906 hpc_err really the predict-zero floor
(E[t^2]) for the model's own crop distribution, or is there headroom?
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1_training')))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

NORM_MEAN = (0.4467, 0.4398, 0.4066)
NORM_STD = (0.2603, 0.2566, 0.2713)
BASE = 'checkpoints/rhan_stl10_large_pseudolabel_best.pth'


def main():
    torch.manual_seed(0)
    torch.set_num_threads(max(4, os.cpu_count() or 4))
    dev = torch.device('cpu')

    print("Loading RHANNext (HPC on) + base checkpoint...")
    cfg = RHANNextConfig(enable_hpc=True, hpc_num_levels=1)
    m = RHANNext(config=cfg).to(dev).eval()

    from checkpoint_utils import compat_load
    ckpt = compat_load(BASE, map_location=dev)
    for k in ('model_state_dict', 'model', 'state_dict'):
        if isinstance(ckpt, dict) and k in ckpt:
            ckpt = ckpt[k]
            break
    missing, unexpected = m.load_state_dict(ckpt, strict=False)
    print(f"  loaded base: missing={len(missing)} unexpected={len(unexpected)}")

    ds = torchvision.datasets.STL10(
        'data/stl10', split='test', download=False,
        transform=T.Compose([T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]))

    n_img, B = 40, 4
    idx = np.random.RandomState(0).choice(len(ds), n_img, replace=False)
    lvl = m.hpc_stack.levels[0]

    t2_acc, t_acc, n_pix_acc = 0.0, 0.0, 0
    per_step_t2 = {t: 0.0 for t in range(m.max_steps)}
    per_step_n = {t: 0 for t in range(m.max_steps)}
    pred_errs = []

    with torch.no_grad():
        for i0 in range(0, n_img, B):
            x = torch.stack([ds[int(j)][0] for j in idx[i0:i0 + B]])
            logits, traj = m(x, return_trajectory=True)
            errs = torch.stack(traj['hpc_errors'])          # (T, B)
            pred_errs.append(float(errs.mean()))
            # Re-derive the targets from the recorded gaze actions.
            for t, a in enumerate(traj['actions']):
                # foveal_sample at the recorded action (same as _forage).
                scale = cfg.fovea_size / 96.0
                scale_col = torch.full((B, 1), scale, dtype=x.dtype)
                zero_col = torch.zeros((B, 1), dtype=x.dtype)
                row0 = torch.cat([scale_col, zero_col, a[:, 0:1]], dim=1)
                row1 = torch.cat([zero_col, scale_col, a[:, 1:2]], dim=1)
                theta = torch.stack([row0, row1], dim=1)
                grid = torch.nn.functional.affine_grid(
                    theta, (B, 3, cfg.fovea_size, cfg.fovea_size),
                    align_corners=False)
                crop = torch.nn.functional.grid_sample(
                    x, grid, mode='bilinear', padding_mode='border',
                    align_corners=False)
                tgt = lvl.extract_target(crop)              # (B, 1, 48, 48) [-1,1]
                per_step_t2[t] += float((tgt ** 2).sum())
                per_step_n[t] += tgt.numel()
                t2_acc += float((tgt ** 2).sum())
                t_acc += float(tgt.sum())
                n_pix_acc += tgt.numel()

    pred_zero = t2_acc / n_pix_acc
    t_mean = t_acc / n_pix_acc
    pred_mean = pred_zero - t_mean ** 2

    print("\n" + "=" * 74)
    print(f"  Model-path floor — {n_img} held-out STL-10 test images, "
          f"B={B}, {m.max_steps} gaze steps")
    print("=" * 74)
    print(f"  mean model hpc_err at init (pred~0): "
          f"{float(np.mean(pred_errs)):.4f}")
    print(f"  E[t] = {t_mean:+.4f}")
    print(f"  predict_zero_mse (E[t^2], model crops) = {pred_zero:.4f}")
    print(f"  predict_mean_mse (Var(t), model crops) = {pred_mean:.4f}")
    print(f"  observed smoke hpc_err (epoch 15)      = 0.6906")
    print(f"    0.6906 / predict_zero = {0.6906 / pred_zero:.3f}x")
    print(f"    0.6906 / predict_mean = {0.6906 / pred_mean:.3f}x")
    print("-" * 74)
    print("  Per-step predict_zero (E[t^2]) along the gaze trajectory:")
    for t in range(m.max_steps):
        print(f"    step {t}: {per_step_t2[t] / per_step_n[t]:.4f}")
    print("=" * 74)


if __name__ == '__main__':
    main()
