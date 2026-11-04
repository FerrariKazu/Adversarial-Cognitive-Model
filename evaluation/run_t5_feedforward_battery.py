#!/usr/bin/env python3
"""
Original T5 Battery on all three ff_112 seeds:
1. AutoAttack standard:
   - APGD-CE (100 steps)
   - APGD-T (100 steps)
   - FAB-T (100 steps)
   - Square attack
   - AA worst-case robust accuracy
2. Epsilon Sweep:
   - {0.5, 1.0, 2.0} x eps_pixel with PGD-50
3. PGD-50 vs PGD-100 gap (gradient masking / optimization audit)
4. Expected Calibration Error (ECE) and reliability metrics
5. Formal Model Card for RHAN-NXA-A, variant feedforward-112 (designated seed 17)
"""

import os, sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np

import torchattacks
from autoattack import AutoAttack
from gen2_foundation.model import Gen2FoundationModel
from evaluation.harness import get_canonical_slice, load_model_from_checkpoint

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ROOT = "data/imagenet100/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
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

class _NormWrap(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_m", m)
        self.register_buffer("_s", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base((x - self._m.to(x)) / self._s.to(x))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (preds == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(corrects[in_bin])
            avg_conf_in_bin = np.mean(confs[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin

    return float(ece)


def run_battery_for_arm(cfg, loader_1k, indices_1k):
    arm_id = cfg["arm_id"]
    ckpt_path = Path(cfg["ckpt"])
    print(f"\n{'='*70}\n[T5 BATTERY] Running {arm_id} ({ckpt_path.name})\n{'='*70}")

    model, _ = load_model_from_checkpoint("ff_112", 112, str(ckpt_path), DEVICE, wrap_normalized=True)
    model.eval()

    # 1. Clean predictions & Probabilities (for ECE)
    all_probs = []
    all_labels = []
    clean_correct = []
    with torch.no_grad():
        for x, y in loader_1k:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            clean_correct.extend((logits.argmax(dim=-1) == y).cpu().numpy().tolist())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    clean_acc = float(np.mean(clean_correct))
    ece = compute_ece(probs_arr, labels_arr, n_bins=15)
    print(f"  Clean 1k: {clean_acc*100:.2f}% | ECE: {ece*100:.2f}%")

    # 2. PGD-50 vs PGD-100 Gap
    def eval_pgd(steps):
        atk = torchattacks.PGD(model, eps=EPS_PIXEL, alpha=EPS_PIXEL / 10.0, steps=steps, random_start=True)
        corr = []
        for x, y in loader_1k:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = atk(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=-1)
            corr.extend((pred == y).cpu().numpy().tolist())
        return float(np.mean(corr)), np.array(corr, dtype=np.uint8)

    acc_pgd50, corr_pgd50 = eval_pgd(50)
    acc_pgd100, corr_pgd100 = eval_pgd(100)
    pgd_gap = (acc_pgd50 - acc_pgd100) * 100
    print(f"  PGD-50: {acc_pgd50*100:.2f}% | PGD-100: {acc_pgd100*100:.2f}% | Gap: {pgd_gap:.2f} pp")

    # 3. Epsilon Sweep: {0.5, 1.0, 2.0} x eps_pixel with PGD-50
    eps_sweep = {}
    for factor in [0.5, 1.0, 2.0]:
        e_val = EPS_PIXEL * factor
        atk = torchattacks.PGD(model, eps=e_val, alpha=e_val / 10.0, steps=50, random_start=True)
        corr = []
        for x, y in loader_1k:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = atk(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=-1)
            corr.extend((pred == y).cpu().numpy().tolist())
        acc = float(np.mean(corr))
        eps_sweep[f"{factor}x"] = {
            "factor": factor,
            "eps_pixel": e_val,
            "eps_255": e_val * 255.0,
            "acc_robust": acc,
            "acc_pct": round(acc * 100, 2),
        }
        print(f"  Eps Sweep {factor}x ({e_val*255:.2f}/255): {acc*100:.2f}%")

    # 4. AutoAttack Components: APGD-CE, APGD-T, FAB-T, Square
    print("  Running AutoAttack standard (APGD-CE, APGD-T, FAB-T, Square)...")
    torch.cuda.empty_cache()
    all_x = []
    all_y = []
    for x, y in loader_1k:
        all_x.append(x)
        all_y.append(y)
    x_total = torch.cat(all_x, dim=0)
    y_total = torch.cat(all_y, dim=0)

    adversary = AutoAttack(model, norm="Linf", eps=EPS_PIXEL, version="standard", device=DEVICE)
    adversary.seed = 42
    dict_adv = adversary.run_standard_evaluation_individual(x_total, y_total, bs=16)

    attacks_corr = {}
    with torch.no_grad():
        for atk_name, x_atk in dict_adv.items():
            corr_list = []
            for i in range(0, len(x_atk), 64):
                xb = x_atk[i:i+64].to(DEVICE)
                yb = y_total[i:i+64].to(DEVICE)
                pred = model(xb).argmax(dim=-1)
                corr_list.extend((pred == yb).cpu().numpy().tolist())
            attacks_corr[atk_name] = np.array(corr_list, dtype=np.uint8)
            print(f"    {atk_name}: {np.mean(corr_list)*100:.2f}%")

    clean_arr = np.array(clean_correct, dtype=np.uint8)
    worst_mask = clean_arr.copy()
    for atk_name in ["apgd-ce", "apgd-t", "fab-t", "square"]:
        if atk_name in attacks_corr:
            worst_mask = worst_mask & attacks_corr[atk_name]
    aa_worst_acc = float(np.mean(worst_mask))
    print(f"  AutoAttack Standard Worst-Case: {aa_worst_acc*100:.2f}%")

    return {
        "arm_id": arm_id,
        "seed": cfg["seed"],
        "is_official": cfg["is_official"],
        "clean_1k": clean_acc,
        "ece": ece,
        "pgd50": acc_pgd50,
        "pgd100": acc_pgd100,
        "pgd_gap_pp": pgd_gap,
        "eps_sweep": eps_sweep,
        "autoattack": {
            "apgd_ce": float(np.mean(attacks_corr.get("apgd-ce", [0]))),
            "apgd_t": float(np.mean(attacks_corr.get("apgd-t", [0]))),
            "fab_t": float(np.mean(attacks_corr.get("fab-t", [0]))),
            "square": float(np.mean(attacks_corr.get("square", [0]))),
            "aa_worst_case": aa_worst_acc,
        },
    }


def main():
    print("=== Original T5 Feedforward Battery ===")
    indices_1k, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    tf_01 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    full_ds = ImageFolder(VAL_ROOT, transform=tf_01)
    slice_ds = Subset(full_ds, indices_1k)
    loader_1k = DataLoader(slice_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    out_json = OUT_DIR / "t5_feedforward_battery.json"
    if out_json.exists():
        try:
            with open(out_json) as f:
                battery_results = json.load(f)
        except Exception:
            battery_results = {
                "protocol_id": "consolidated-flagship-112/1",
                "diagnostic": "T5 Feedforward Battery (Standard AutoAttack, Eps Sweep, ECE, PGD Gap)",
                "slice_sha256": slice_hash,
                "n_samples": len(indices_1k),
                "arms": {},
            }
    else:
        battery_results = {
            "protocol_id": "consolidated-flagship-112/1",
            "diagnostic": "T5 Feedforward Battery (Standard AutoAttack, Eps Sweep, ECE, PGD Gap)",
            "slice_sha256": slice_hash,
            "n_samples": len(indices_1k),
            "arms": {},
        }

    for cfg in FF_SEEDS:
        arm_id = cfg["arm_id"]
        if arm_id in battery_results.get("arms", {}) and "autoattack" in battery_results["arms"][arm_id]:
            print(f"\n[SKIP] {arm_id} already evaluated in {out_json}")
            continue
        res = run_battery_for_arm(cfg, loader_1k, indices_1k)
        battery_results["arms"][arm_id] = res
        with open(out_json, "w") as f:
            json.dump(battery_results, f, indent=2)

    # Summary across all 3 seeds
    cln_scores = [r["clean_1k"] * 100 for r in battery_results["arms"].values()]
    aa_scores = [r["autoattack"]["aa_worst_case"] * 100 for r in battery_results["arms"].values()]
    pgd50_scores = [r["pgd50"] * 100 for r in battery_results["arms"].values()]

    battery_results["three_seed_summary"] = {
        "clean_mean": float(np.mean(cln_scores)),
        "clean_range": [float(np.min(cln_scores)), float(np.max(cln_scores))],
        "pgd50_mean": float(np.mean(pgd50_scores)),
        "pgd50_range": [float(np.min(pgd50_scores)), float(np.max(pgd50_scores))],
        "aa_worst_mean": float(np.mean(aa_scores)),
        "aa_worst_range": [float(np.min(aa_scores)), float(np.max(aa_scores))],
        "designated_official_seed": 17,
        "official_model_name": "RHAN-NXA-A, variant feedforward-112",
    }

    out_json = OUT_DIR / "t5_feedforward_battery.json"
    with open(out_json, "w") as f:
        json.dump(battery_results, f, indent=2)

    print(f"\nT5 Battery Complete. Results written to {out_json}")


if __name__ == "__main__":
    main()
