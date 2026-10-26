#!/usr/bin/env python3
"""G2-D5: Glimpse Scaling (Gate L) — Test-time glimpse budget sweep
=====================================================================
Tests how clean and robust accuracy varies with number of glimpses
T ∈ {1, 2, 4, 6, 8} for the recurrence_only model.

This requires temporarily overriding num_glimpses at test time.
The model was trained with T=4.

Protocol:
  - 500 images, seed=42
  - Clean + PGD-10 @ eps=0.094
  - Both models evaluated (recurrence_only_final, gen1_core_best_clean)
  - Time per T recorded

Registration: G2-D5
Dependency: G2-D3 (final checkpoints must exist)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from training.train_gen2_foundation import run_pgd_ce_eval

SEED = 42
SAMPLE_SIZE = 500
BATCH_SIZE = 32
EPS = 0.094
PGD_STEPS = 10
GLIMPSE_BUDGETS = [1, 2, 4, 6, 8]

CHECKPOINTS = [
    {"name": "recurrence_only_final", "path": "checkpoints/foundation_recurrence_only_final.pth",
     "phase": "recurrence_only", "arm": "v1"},
    {"name": "gen1_core_best_clean", "path": "checkpoints/foundation_gen1_core_best_clean.pth",
     "phase": "gen1_core", "arm": "v1"},
]


def get_loader(data_root: str, seed: int, sample_size: int, batch_size: int) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_ds), generator=g)[:sample_size].tolist()
    return DataLoader(Subset(val_ds, indices), batch_size=batch_size, shuffle=False, num_workers=2)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             eps: float = 0.0, pgd_steps: int = 0) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if eps > 0.0 and pgd_steps > 0:
            x = run_pgd_ce_eval(model, x, y, eps=eps, steps=pgd_steps)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return float(correct) / float(max(total, 1))


def main():
    parser = argparse.ArgumentParser(description="G2-D5 Glimpse Scaling")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--output-dir", type=str, default="report/d5_glimpse_scaling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-D5] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    loader = get_loader(args.data_root, SEED, SAMPLE_SIZE, BATCH_SIZE)
    all_results = {}

    for ckpt_info in CHECKPOINTS:
        name = ckpt_info["name"]
        ckpt_path = ckpt_info["path"]
        phase = ckpt_info["phase"]
        arm = ckpt_info["arm"]

        if not os.path.exists(ckpt_path):
            print(f"[G2-D5] BLOCKER: {ckpt_path} not found — skipping {name}")
            all_results[name] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        print(f"\n{'='*70}")
        print(f"[G2-D5] Evaluating: {name} (phase={phase})")
        ckpt_results: Dict = {"phase": phase, "arm": arm, "budgets": {}}

        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved

        for T in GLIMPSE_BUDGETS:
            print(f"\n  --- T={T} glimpses ---")
            model = Gen2FoundationModel(phase=phase, arm=arm, num_glimpses=T).to(device)
            model.load_state_dict(sd, strict=False)
            model.eval()

            t0 = time.time()
            clean = evaluate(model, loader, device, eps=0.0, pgd_steps=0)
            t_clean = time.time() - t0
            print(f"    Clean: {clean*100:.2f}% ({t_clean:.1f}s)")

            t0 = time.time()
            robust = evaluate(model, loader, device, eps=EPS, pgd_steps=PGD_STEPS)
            t_robust = time.time() - t0
            print(f"    PGD-{PGD_STEPS}@{EPS}: {robust*100:.2f}% ({t_robust:.1f}s)")

            ckpt_results["budgets"][T] = {
                "clean": clean,
                "robust": robust,
                "runtime_clean_s": t_clean,
                "runtime_robust_s": t_robust,
            }

            del model
            torch.cuda.empty_cache()

        all_results[name] = ckpt_results

        out_path = os.path.join(args.output_dir, f"g2_d5_{name}.json")
        with open(out_path, "w") as f:
            json.dump(ckpt_results, f, indent=2)
        print(f"\n  ✓ Saved: {out_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("G2-D5 GLIMPSE SCALING SUMMARY")
    print("=" * 80)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: {res['error']}")
            continue
        print(f"\n{name} (phase={res['phase']}):")
        print(f"  {'T':>4} | {'Clean':>8} | {'PGD-10':>8}")
        print(f"  {'-'*28}")
        for T, metrics in res["budgets"].items():
            print(f"  {T:>4} | {metrics['clean']*100:>7.2f}% | {metrics['robust']*100:>7.2f}%")

    combined_path = os.path.join(args.output_dir, "g2_d5_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined: {combined_path}")


if __name__ == "__main__":
    main()
