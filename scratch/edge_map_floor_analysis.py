#!/usr/bin/env python3
"""
Edge-map target floor analysis (Stage 2 HPC gate calibration).

Question: the Stage 2 smoke observed hpc_error_mean ~= 0.69 for the HPC
predictor. Is 0.69 near the *floor* for a sparse Sobel edge-map target
(predict-zero / predict-mean), or is there real room to learn?

Computes, on held-out (STL-10 test) images with the SAME preprocessing the
trainer uses and the SAME target pipeline (rhan_core EdgeMapExtractor +
extract_target's 2*t - 1 rescale, sampled as 48x48 foveal crops at a spread
of gaze positions):

    predict_zero_mse      = MSE(target, zeros)          = E[t^2]
    predict_mean_mse      = MSE(target, target.mean())  = Var(t)  (global)
    per_sample_mean_mse   = MSE(target, sample_mean)    = E[Var(t | sample)]
    predict_neg1_mse      = MSE(target, -1)  = 4*E[s^2]  ("all non-edge")

and compares them against the observed 0.6906. No GPU needed.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from rhan_core.predictive_coding.feature_targets import EdgeMapExtractor

NORM_MEAN = (0.4467, 0.4398, 0.4066)
NORM_STD = (0.2603, 0.2566, 0.2713)
FOVEA = 48
IMG = 96

# Gaze positions (normalized action coords in [-1, 1]) spanning the image the
# way the model's foraging loop does (starts near center, moves within
# [-0.9, 0.9]). A 5x5 grid + center approximates the crop distribution.
POSITIONS = [(0.0, 0.0),
             (-0.75, -0.75), (-0.75, 0.0), (-0.75, 0.75),
             (0.0, -0.75), (0.0, 0.75),
             (0.75, -0.75), (0.75, 0.0), (0.75, 0.75)]


def foveal_crops(x: torch.Tensor, positions) -> torch.Tensor:
    """(N, 3, 96, 96) -> (N*P, 3, 48, 48) bilinear crops at each gaze position
    (mirrors model_rhan_v10.foveal_sample: scale = fovea/96, border pad)."""
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
    return torch.cat(out, dim=0)                       # (B*P, 3, 48, 48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/stl10')
    ap.add_argument('--split', default='test', choices=['train', 'test'])
    ap.add_argument('--n-images', type=int, default=2000,
                    help='images to sample (test split has 8000, train 5000)')
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device('cpu')

    print(f"Loading STL-10 {args.split} split from {args.root} "
          f"({args.n_images} images)...")
    ds = torchvision.datasets.STL10(
        args.root, split=args.split, download=False,
        transform=T.Compose([T.ToTensor(),
                             T.Normalize(NORM_MEAN, NORM_STD)]))
    n = min(args.n_images, len(ds))
    idx = np.random.RandomState(0).choice(len(ds), n, replace=False)

    extractor = EdgeMapExtractor().to(dev).eval()

    # Accumulators over all crops/pixels.
    n_crops = 0
    s2_sum = 0.0          # sum of s^2  (raw Sobel magnitude squared)
    t2_sum = 0.0          # sum of t^2  (rescaled target squared)
    t_sum = 0.0           # sum of t
    per_sample = []       # per-crop: mean t, var t, frac-edge
    per_position = {p: [] for p in POSITIONS}

    with torch.no_grad():
        for i0 in range(0, n, args.batch):
            imgs = torch.stack([ds[int(j)][0] for j in idx[i0:i0 + args.batch]])
            crops = foveal_crops(imgs, POSITIONS)          # (B*P, 3, 48, 48)
            raw = extractor(crops)                         # (B*P, 1, 48, 48) [0,1]
            tgt = raw * 2.0 - 1.0                          # extract_target: [-1,1]
            Bp = crops.shape[0]
            n_crops += Bp
            s2_sum += float((raw ** 2).sum())
            t2_sum += float((tgt ** 2).sum())
            t_sum += float(tgt.sum())
            tflat = tgt.view(Bp, -1)
            sflat = raw.view(Bp, -1)
            per_sample.append(torch.stack([
                tflat.mean(dim=1),
                tflat.var(dim=1, unbiased=False),
                (sflat > 0.1).float().mean(dim=1),
            ], dim=1))
            for k, pos in enumerate(POSITIONS):
                tsel = tflat[k::len(POSITIONS)]
                per_position[pos].append(tsel)

    ps = torch.cat(per_sample)                             # (B*P, 3)
    n_pix = n_crops * FOVEA * FOVEA

    pred_zero = t2_sum / n_pix                             # E[t^2]
    t_mean = t_sum / n_pix
    pred_mean = pred_zero - t_mean ** 2                    # Var(t) = MSE(t, E[t])
    between = float((ps[:, 0] ** 2).mean()) - t_mean ** 2  # Var(E[t|sample])
    per_sample_mean = float(ps[:, 1].mean())               # E[Var(t|sample)]
    s2_mean = s2_sum / n_pix
    pred_neg1 = 4.0 * s2_mean                              # MSE(t, -1) = 4 E[s^2]
    edge_frac = float(ps[:, 2].mean())

    print("\n" + "=" * 74)
    print(f"  Edge-map target floors — STL-10 {args.split}, {n} images, "
          f"{n_crops} foveal crops")
    print("=" * 74)
    print(f"  target t = 2*s - 1 (s = per-sample max-normalized Sobel, [0,1])")
    print(f"  E[t] = {t_mean:+.4f}   E[s] = {(t_mean + 1) / 2:.4f}   "
          f"E[s^2] = {s2_mean:.4f}")
    print(f"  edge fraction (s > 0.1): {edge_frac:.4f}  "
          f"({edge_frac*100:.1f}% of pixels)")
    print(f"  Var(E[t|sample]) (between-crop): {between:.4f}")
    print("-" * 74)
    print(f"  predict_zero_mse    = MSE(t, 0)        = {pred_zero:.4f}")
    print(f"  predict_mean_mse    = MSE(t, mean(t))  = {pred_mean:.4f}")
    print(f"  per_sample_mean_mse = MSE(t, samp-mean)= {per_sample_mean:.4f}")
    print(f"  predict_neg1_mse    = MSE(t, -1)       = {pred_neg1:.4f} "
          f"('all non-edge' / predict the global mean)")
    print("-" * 74)
    print(f"  observed smoke hpc_err (epoch 15): 0.6906")
    for name, v in [("predict_zero", pred_zero), ("predict_mean", pred_mean),
                    ("per_sample_mean", per_sample_mean),
                    ("predict_neg1", pred_neg1)]:
        print(f"    0.6906 vs {name:>18}: "
              f"obs/floor = {0.6906 / v:.3f}x  "
              f"(floor is {100 * (1 - v / 0.6906):+5.1f}% of obs)")
    print("=" * 74)

    # Sensitivity: per gaze position (center vs periphery).
    print("\n  Per-position floors (predict_zero = E[t^2]):")
    for pos in POSITIONS:
        tcat = torch.cat(per_position[pos])                # (N, 48*48)
        e_t2 = float((tcat ** 2).mean())
        var = float(tcat.var(dim=1, unbiased=False).mean())
        print(f"    pos {str(pos):>14}: predict_zero={e_t2:.4f}  "
              f"per_sample_var={var:.4f}")


if __name__ == '__main__':
    main()
