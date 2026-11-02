#!/usr/bin/env python3
"""APGD-CE + APGD-T Evaluation on Flagship Seed 42 (T=4, 1k canonical slice)
================================================================================
Evaluates foundation_recurrence_112_seed42_final_eb40bf84.pth on the canonical
1,000-image ImageNet-100 validation slice under:
  1. Clean accuracy
  2. APGD-CE (100 steps, restarts=1)
  3. APGD-T (100 steps, 9 target classes)
  4. Worst-case robust accuracy (clean & APGD-CE & APGD-T)

Saves results to evaluation/stage0_results/flagship_seed42_apgd_1k.json
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
from autoattack import AutoAttack

from evaluation.harness import get_canonical_slice, load_model_from_checkpoint

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ROOT = "data/imagenet100/val"
CKPT_PATH = "checkpoints/foundation_recurrence_112_seed42_final_eb40bf84.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_PIXEL = 5.417 / 255.0  # 0.0212431


def main():
    print("=" * 70)
    print("Flagship Seed 42 APGD-T + APGD-CE (T=4, 1k Canonical Slice)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Epsilon: {EPS_PIXEL:.6f} ({EPS_PIXEL*255:.3f}/255)")

    # 1. Load 1k canonical slice
    indices_1k, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    full_ds = ImageFolder(VAL_ROOT, transform=tf_val)
    slice_ds = Subset(full_ds, indices_1k)
    loader_1k = DataLoader(slice_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load model (recurrence_only, fovea_size=112, T=4)
    model, meta = load_model_from_checkpoint("recurrence_only", 112, CKPT_PATH, DEVICE, wrap_normalized=True)
    model.eval()

    # 3. Clean Evaluation
    print("\n[Step 1] Running Clean Evaluation (1,000 images)...")
    clean_correct = []
    all_x = []
    all_y = []
    with torch.no_grad():
        for x, y in loader_1k:
            all_x.append(x)
            all_y.append(y)
            x_dev, y_dev = x.to(DEVICE), y.to(DEVICE)
            preds = model(x_dev).argmax(dim=-1)
            clean_correct.extend((preds == y_dev).cpu().numpy().tolist())

    clean_acc = float(np.mean(clean_correct))
    print(f"  Clean 1k Accuracy: {clean_acc*100:.2f}%")

    x_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)

    # 4. AutoAttack running APGD-CE and APGD-T
    print("\n[Step 2] Running APGD-CE and APGD-T (AutoAttack, bs=16)...")
    adversary = AutoAttack(model, norm="Linf", eps=EPS_PIXEL, version="standard", device=DEVICE)
    adversary.seed = 42
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]

    t0 = time.time()
    dict_adv = adversary.run_standard_evaluation_individual(x_total, y_total, bs=16)
    elapsed = time.time() - t0
    print(f"  Attacks finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # 5. Evaluate robust accuracy for each attack
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

    # Worst-case: clean AND apgd-ce AND apgd-t
    clean_arr = np.array(clean_correct, dtype=np.uint8)
    worst_mask = clean_arr.copy()
    for atk_name in ["apgd-ce", "apgd-t"]:
        if atk_name in attacks_corr:
            worst_mask = worst_mask & attacks_corr[atk_name]

    worst_acc = float(np.mean(worst_mask))
    apgd_ce_acc = float(np.mean(attacks_corr.get("apgd-ce", clean_arr)))
    apgd_t_acc = float(np.mean(attacks_corr.get("apgd-t", clean_arr)))

    print(f"\n[Summary] Flagship Seed 42 (T=4, 1k slice):")
    print(f"  Clean 1k:        {clean_acc*100:.2f}%")
    print(f"  APGD-CE 1k:      {apgd_ce_acc*100:.2f}%")
    print(f"  APGD-T 1k:       {apgd_t_acc*100:.2f}%")
    print(f"  Worst-Case 1k:   {worst_acc*100:.2f}%")

    results = {
        "protocol_id": "consolidated-flagship-112/1",
        "diagnostic": "Flagship Seed 42 APGD-T + APGD-CE (T=4, 1k canonical slice)",
        "arm_id": "recurrence_112_seed42",
        "checkpoint": CKPT_PATH,
        "slice_sha256": slice_hash,
        "n_samples": len(indices_1k),
        "eps_pixel": EPS_PIXEL,
        "eps_255": EPS_PIXEL * 255.0,
        "t_eval": 4,
        "clean_1k": clean_acc,
        "apgd_ce_1k": apgd_ce_acc,
        "apgd_t_1k": apgd_t_acc,
        "worst_case_1k": worst_acc,
        "attacks_corr": {k: v.tolist() for k, v in attacks_corr.items()},
        "elapsed_sec": elapsed,
    }

    out_file = OUT_DIR / "flagship_seed42_apgd_1k.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
