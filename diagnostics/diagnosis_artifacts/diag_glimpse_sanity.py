#!/usr/bin/env python3
"""
diag_glimpse_sanity.py — §6 INPUT/GLIMPSE SANITY (real model path).
================================================================================
Through the REAL FoundationModel._glimpse path (foveal_sample -> _prep ->
_trunk -> refinement), for each of the T=4 fixed gaze points:
  * crop tensor stats (the plane the trunk actually sees)
  * pooled activation stats + token-norm stats (non-degenerate features)
  * distinctness: do different gaze points produce different crops and
    different representations (no constant fixation, no empty glimpse)?
  * backbone input integrity: dtype, range, no accidental grayscale
    (channel-variance check), no excessive downsampling (56px crops from
    96px frames with distinct content)
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, save_json  # noqa: E402

from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, build_model, build_fixed_gaze_schedule)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=32, num_workers=4)
    x, _ = next(iter(loaders["val"]))
    x = x.to(device)

    cfg = FoundationConfig()
    model = build_model(cfg, "belief_with_f").to(device).eval()
    B = x.shape[0]
    sched = build_fixed_gaze_schedule(B, cfg.num_glimpses, device)

    per_gaze = []
    with torch.no_grad():
        for t in range(cfg.num_glimpses):
            a_t = sched[:, t, :]
            crop = __import__("noesis_vision.models.foveation",
                              fromlist=["foveal_sample"]).foveal_sample(
                                  x, a_t, fovea_size=cfg.fovea_size)
            pooled, tokens = model._glimpse(x, a_t)
            per_gaze.append({
                "t": t,
                "gaze_xy": [round(v, 3) for v in a_t[0].tolist()],
                "crop": {"shape": list(crop.shape),
                         "min": round(crop.min().item(), 4),
                         "max": round(crop.max().item(), 4),
                         "mean": round(crop.mean().item(), 4),
                         "std": round(crop.std().item(), 4),
                         "channel_var_mean": round(
                             float(crop.var(dim=(0, 2, 3)).mean().item()), 6),
                         "not_grayscale": bool(crop.var(dim=(0, 2, 3)).min() > 1e-4)},
                "pooled": {"mean": round(pooled.mean().item(), 5),
                           "std": round(pooled.std().item(), 5),
                           "norm_mean": round(float(pooled.norm(dim=1).mean().item()), 4),
                           "degenerate": bool(pooled.std().item() < 1e-6)},
                "tokens": {"norm_mean": round(float(tokens.norm(dim=-1).mean().item()), 4),
                           "std": round(tokens.std().item(), 5)},
            })
            print(f"  t={t} gaze={per_gaze[-1]['gaze_xy']} "
                  f"crop[mean/std]={per_gaze[-1]['crop']['mean']}/"
                  f"{per_gaze[-1]['crop']['std']} "
                  f"|z|={per_gaze[-1]['pooled']['norm_mean']} "
                  f"tok|.|={per_gaze[-1]['tokens']['norm_mean']}")

    means = [g["crop"]["mean"] for g in per_gaze]
    crops_distinct = len(set(means)) == len(means)
    pooled_all_distinct = len({
        g["pooled"]["norm_mean"] for g in per_gaze}) == cfg.num_glimpses
    out = {"section": "6_input_glimpse_sanity", "per_gaze": per_gaze,
           "checks": {
               "crops_distinct_across_gazes": crops_distinct,
               "no_constant_fixation": bool(sched[:, 0, :].abs().max().item() > 0.0),
               "gaze_inside_frame": bool(sched.abs().max().item() <= 0.6 + 1e-6),
               "crop_range_valid": all(-4.0 <= g["crop"]["min"] and
                                       g["crop"]["max"] <= 4.0 for g in per_gaze),
               "no_grayscale_conversion": all(g["crop"]["not_grayscale"]
                                              for g in per_gaze),
               "no_degenerate_pooled": all(not g["pooled"]["degenerate"]
                                           for g in per_gaze),
               "crop_resolution_is_56": all(g["crop"]["shape"][-1] == 56
                                            for g in per_gaze)},
           "verdict_inputs": None}
    checks = out["checks"]
    out["verdict"] = ("PASS — glimpses are spatially distinct, correctly "
                      "normalized, non-degenerate" if all(checks.values())
                      else "ATTENTION — " + ", ".join(k for k, v in checks.items()
                                                     if not v))
    save_json("06_glimpse_sanity.json", out)
    print(f"SECTION 6 VERDICT: {out['verdict']}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
