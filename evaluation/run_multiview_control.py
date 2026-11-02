#!/usr/bin/env python3
"""Multi-View Control: Feedforward Trunk + Flagship Gaze Sampler & Recurrent Fusion
================================================================================
Protocol: multiview-control-112/1
Prereg: prereg/multiview_control_prereg_v1.json

Tests whether the robustness gain in flagship seed 42 comes from recurrent
dynamics or simply multi-view spatial sampling and feature integration.

Wraps each trained ff_112 seed in the flagship's spatial gaze sampler
(imported from noesis_vision.models.foveation.foveal_sample) and fixed-EMA
recurrent fusion (z_0 = g_0, z_t = 0.5*z_{t-1} + 0.5*g_t).
Attacked end-to-end on the canonical 1k slice.

Pre-declared reading:
  match (robust >= 37.0% on 1k) => multi-view averaging
  shortfall (robust < 37.0% on 1k) => inconclusive
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchattacks
from autoattack import AutoAttack

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import foveal_sample
from evaluation.harness import get_canonical_slice, load_model_from_checkpoint, NormalizedModel

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ROOT = "data/imagenet100/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_PIXEL = 5.417 / 255.0

FF_SEEDS = [
    {
        "arm_id": "ff_112_seed17",
        "seed": 17,
        "ckpt": "checkpoints/foundation_ff_112_seed17_final_d00fb750.pth",
    },
    {
        "arm_id": "ff_112_seed42",
        "seed": 42,
        "ckpt": "checkpoints/foundation_ff_112_seed42_final_2920df54.pth",
    },
    {
        "arm_id": "ff_112_seed99",
        "seed": 99,
        "ckpt": "checkpoints/foundation_ff_112_seed99_final_abde98a0.pth",
    },
]


class FlagshipGazeFusionWrapper(nn.Module):
    """Wraps feedforward model using flagship's gaze sampler + fusion."""
    def __init__(self, ff_model: Gen2FoundationModel):
        super().__init__()
        self.ff_model = ff_model
        self.fovea_size = ff_model.fovea_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        gaze_schedule = [
            torch.zeros(B, 2, device=device),
            torch.tensor([[0.3, 0.3]], device=device).expand(B, -1),
            torch.tensor([[-0.3, 0.3]], device=device).expand(B, -1),
            torch.tensor([[0.0, -0.3]], device=device).expand(B, -1),
        ]
        # Flagship gaze sampling via foveal_sample
        crops = [foveal_sample(x, gaze_schedule[t], fovea_size=self.fovea_size) for t in range(4)]
        # Pass each crop through ff_model's backbone trunk
        pooled = [self.ff_model.backbone(c)[0] for c in crops]
        # Flagship fixed-EMA fusion: z_0 = g_0; z_t = 0.5*z_{t-1} + 0.5*g_t
        z_t = pooled[0]
        for t in range(1, 4):
            z_t = 0.5 * z_t + 0.5 * pooled[t]
        return self.ff_model.cls_head(z_t)


def evaluate_multiview_control_for_seed(cfg, loader_1k):
    arm_id = cfg["arm_id"]
    ckpt_path = cfg["ckpt"]
    print(f"\n{'='*70}\n[MULTI-VIEW CONTROL] Running {arm_id}\n{'='*70}")

    # Load base ff model
    base_model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = sd.get("model", sd)
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()

    # Wrap with flagship gaze fusion and normalization
    mv_model = FlagshipGazeFusionWrapper(base_model)
    norm_model = NormalizedModel(mv_model).to(DEVICE)
    norm_model.eval()

    # Clean
    clean_correct = []
    all_x = []
    all_y = []
    with torch.no_grad():
        for x, y in loader_1k:
            all_x.append(x)
            all_y.append(y)
            xb, yb = x.to(DEVICE), y.to(DEVICE)
            preds = norm_model(xb).argmax(dim=-1)
            clean_correct.extend((preds == yb).cpu().numpy().tolist())

    clean_acc = float(np.mean(clean_correct))
    print(f"  Clean Accuracy: {clean_acc*100:.2f}%")

    x_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)

    # PGD-50
    print("  Running PGD-50 (alpha=2.5*eps/50)...")
    alpha = 2.5 * EPS_PIXEL / 50.0
    atk_pgd = torchattacks.PGD(norm_model, eps=EPS_PIXEL, alpha=alpha, steps=50, random_start=True)
    pgd_correct = []
    for x, y in loader_1k:
        xb, yb = x.to(DEVICE), y.to(DEVICE)
        x_adv = atk_pgd(xb, yb)
        with torch.no_grad():
            preds = norm_model(x_adv).argmax(dim=-1)
        pgd_correct.extend((preds == yb).cpu().numpy().tolist())
    pgd50_acc = float(np.mean(pgd_correct))
    print(f"  PGD-50 Accuracy: {pgd50_acc*100:.2f}%")

    # APGD-CE (via AutoAttack)
    print("  Running APGD-CE (AutoAttack, bs=16)...")
    adversary = AutoAttack(norm_model, norm="Linf", eps=EPS_PIXEL, version="standard", device=DEVICE)
    adversary.seed = 42
    adversary.attacks_to_run = ["apgd-ce"]
    dict_adv = adversary.run_standard_evaluation_individual(x_total, y_total, bs=16)

    apgd_correct = []
    x_apgd = dict_adv["apgd-ce"]
    with torch.no_grad():
        for i in range(0, len(x_apgd), 64):
            xb = x_apgd[i:i+64].to(DEVICE)
            yb = y_total[i:i+64].to(DEVICE)
            preds = norm_model(xb).argmax(dim=-1)
            apgd_correct.extend((preds == yb).cpu().numpy().tolist())
    apgd_ce_acc = float(np.mean(apgd_correct))
    print(f"  APGD-CE Accuracy: {apgd_ce_acc*100:.2f}%")

    clean_arr = np.array(clean_correct, dtype=np.uint8)
    pgd_arr = np.array(pgd_correct, dtype=np.uint8)
    apgd_arr = np.array(apgd_correct, dtype=np.uint8)
    worst_arr = clean_arr & pgd_arr & apgd_arr
    worst_acc = float(np.mean(worst_arr))
    print(f"  Worst-Case Robust Accuracy: {worst_acc*100:.2f}%")

    return {
        "arm_id": arm_id,
        "seed": cfg["seed"],
        "clean_1k": clean_acc,
        "pgd50_1k": pgd50_acc,
        "apgd_ce_1k": apgd_ce_acc,
        "worst_case_1k": worst_acc,
    }


def main():
    print("=" * 70)
    print("Protocol: multiview-control-112/1")
    print("Running Multi-View Control on ff_112 Seeds")
    print("=" * 70)

    indices_1k, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    full_ds = ImageFolder(VAL_ROOT, transform=tf_val)
    slice_ds = Subset(full_ds, indices_1k)
    loader_1k = DataLoader(slice_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    results = {
        "protocol_id": "multiview-control-112/1",
        "prereg": "prereg/multiview_control_prereg_v1.json",
        "slice_sha256": slice_hash,
        "n_samples": len(indices_1k),
        "eps_pixel": EPS_PIXEL,
        "eps_255": EPS_PIXEL * 255.0,
        "flagship_seed42_ref": {
            "clean_1k": 0.764,
            "worst_case_5k": 0.3772,
            "threshold_match": 0.3700,
        },
        "arms": {},
    }

    for cfg in FF_SEEDS:
        res = evaluate_multiview_control_for_seed(cfg, loader_1k)
        results["arms"][cfg["arm_id"]] = res

    # Reading check
    worst_vals = [v["worst_case_1k"] for v in results["arms"].values()]
    mean_worst = float(np.mean(worst_vals))
    results["summary"] = {
        "mean_worst_case_1k": mean_worst,
        "match_threshold": 0.3700,
        "declared_reading": "multi-view averaging" if mean_worst >= 0.3700 else "inconclusive",
    }
    print(f"\n[Multi-View Control Reading] Mean Worst-Case: {mean_worst*100:.2f}% | Verdict Reading: {results['summary']['declared_reading']}")

    out_file = OUT_DIR / "multiview_control_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_file}")


if __name__ == "__main__":
    main()

