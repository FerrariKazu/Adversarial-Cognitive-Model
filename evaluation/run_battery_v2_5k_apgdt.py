#!/usr/bin/env python3
"""Versioned Battery v2: APGD-T (9 targets) + APGD-CE + PGD-50 on 5k Full Validation
================================================================================
Evaluates all three ff_112 seeds on the full 5,000-image ImageNet-100 validation set
with targeted APGD (APGD-T with 9 target classes per sample) alongside APGD-CE and PGD-50.

DOES NOT overwrite existing results; writes to:
  evaluation/stage0_results/battery_v2_5k_apgdt.json
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
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from autoattack import AutoAttack

from evaluation.harness import load_model_from_checkpoint

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ROOT = "data/imagenet100/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_PIXEL = 5.417 / 255.0

FF_SEEDS = [
    {
        "arm_id": "ff_112_seed17",
        "seed": 17,
        "is_official": True,
        "ckpt": "checkpoints/foundation_ff_112_seed17_final_d00fb750.pth",
    },
    {
        "arm_id": "ff_112_seed42",
        "seed": 42,
        "is_official": False,
        "ckpt": "checkpoints/foundation_ff_112_seed42_final_2920df54.pth",
    },
    {
        "arm_id": "ff_112_seed99",
        "seed": 99,
        "is_official": False,
        "ckpt": "checkpoints/foundation_ff_112_seed99_final_abde98a0.pth",
    },
]


def run_battery_v2_for_arm(cfg, loader_5k):
    arm_id = cfg["arm_id"]
    ckpt_path = Path(cfg["ckpt"])
    print(f"\n{'='*70}\n[BATTERY v2 5k] Running {arm_id} ({ckpt_path.name})\n{'='*70}")

    model, meta = load_model_from_checkpoint("ff_112", 112, str(ckpt_path), DEVICE, wrap_normalized=True)
    model.eval()

    # Clean
    clean_correct = []
    all_x = []
    all_y = []
    with torch.no_grad():
        for x, y in loader_5k:
            all_x.append(x)
            all_y.append(y)
            x_dev, y_dev = x.to(DEVICE), y.to(DEVICE)
            preds = model(x_dev).argmax(dim=-1)
            clean_correct.extend((preds == y_dev).cpu().numpy().tolist())

    clean_acc = float(np.mean(clean_correct))
    print(f"  Clean 5k: {clean_acc*100:.2f}%")

    x_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)

    # AutoAttack APGD-CE and APGD-T
    print("  Running APGD-CE + APGD-T (9 targets) on 5,000 images...")
    adversary = AutoAttack(model, norm="Linf", eps=EPS_PIXEL, version="standard", device=DEVICE)
    adversary.seed = 42
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]

    t0 = time.time()
    dict_adv = adversary.run_standard_evaluation_individual(x_total, y_total, bs=32)
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    attacks_corr = {}
    with torch.no_grad():
        for atk_name, x_atk in dict_adv.items():
            corr_list = []
            for i in range(0, len(x_atk), 64):
                xb = x_atk[i:i+64].to(DEVICE)
                yb = y_total[i:i+64].to(DEVICE)
                preds = model(xb).argmax(dim=-1)
                corr_list.extend((preds == yb).cpu().numpy().tolist())
            attacks_corr[atk_name] = np.array(corr_list, dtype=np.uint8)
            print(f"    {atk_name}: {np.mean(corr_list)*100:.2f}%")

    clean_arr = np.array(clean_correct, dtype=np.uint8)
    worst_mask = clean_arr.copy()
    for atk_name in ["apgd-ce", "apgd-t"]:
        if atk_name in attacks_corr:
            worst_mask = worst_mask & attacks_corr[atk_name]

    worst_acc = float(np.mean(worst_mask))
    apgd_ce_acc = float(np.mean(attacks_corr.get("apgd-ce", clean_arr)))
    apgd_t_acc = float(np.mean(attacks_corr.get("apgd-t", clean_arr)))

    print(f"  Worst-Case (Clean & APGD-CE & APGD-T): {worst_acc*100:.2f}%")

    return {
        "arm_id": arm_id,
        "seed": cfg["seed"],
        "is_official": cfg["is_official"],
        "clean_5k": clean_acc,
        "apgd_ce_5k": apgd_ce_acc,
        "apgd_t_5k": apgd_t_acc,
        "worst_case_5k": worst_acc,
        "elapsed_sec": elapsed,
    }


def main():
    print("=== Battery v2: 5k APGD-T (9 targets) + APGD-CE ===")
    tf_01 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    full_ds = ImageFolder(VAL_ROOT, transform=tf_01)
    loader_5k = DataLoader(full_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    out_json = OUT_DIR / "battery_v2_5k_apgdt.json"
    if out_json.exists():
        with open(out_json) as f:
            results = json.load(f)
    else:
        results = {
            "protocol_id": "consolidated-flagship-112/1",
            "battery_version": "v2_5k_apgdt",
            "description": "5k Full Validation with APGD-T (9 target classes) + APGD-CE",
            "eps_pixel": EPS_PIXEL,
            "eps_255": EPS_PIXEL * 255.0,
            "n_samples": len(full_ds),
            "arms": {},
        }

    for cfg in FF_SEEDS:
        arm_id = cfg["arm_id"]
        if arm_id in results.get("arms", {}):
            print(f"[SKIP] {arm_id} already evaluated in {out_json}")
            continue
        res = run_battery_v2_for_arm(cfg, loader_5k)
        results["arms"][arm_id] = res
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nCompleted Battery v2. Results written to {out_json}")


if __name__ == "__main__":
    main()

