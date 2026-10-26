#!/usr/bin/env python3
"""G2-D3: Final Checkpoint Evaluation (recurrence_only vs gen1_core)
=====================================================================
Runs the registered final-checkpoint protocol on both reference models:
  - foundation_recurrence_only_final.pth   (control)
  - foundation_gen1_core_best_clean.pth    (full Gen-2)

Protocol per checkpoint:
  1. 8 seeds × 300 images — clean + PGD-10 at eps∈{0.031, 0.062, 0.094}
  2. 500 images seed=42 — Clean, PGD-50 @ eps=0.094, EOT-PGD (n_eot=5, steps=20) @ eps=0.094

All outputs are written to report/d3_final_eval/ as JSON.
Comparison table printed to stdout.

Registration: G2-D3
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from gen2_foundation.eot import eot_pgd_attack
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD, MEAN_STD
from training.train_gen2_foundation import run_pgd_ce_eval

DEFAULT_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108]
EVAL_SAMPLE_SIZE = 300
FINAL_EVAL_SAMPLE_SIZE = 500


CHECKPOINTS = [
    {
        "name": "recurrence_only_final",
        "path": "checkpoints/foundation_recurrence_only_final.pth",
        "phase": "recurrence_only",
        "arm": "v1",
        "role": "control",
    },
    {
        "name": "gen1_core_best_clean",
        "path": "checkpoints/foundation_gen1_core_best_clean.pth",
        "phase": "gen1_core",
        "arm": "v1",
        "role": "full_gen2",
    },
]


def get_val_dataset(data_root: str) -> datasets.ImageFolder:
    tf = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)


def evaluate_clean_and_pgd(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.0,
    pgd_steps: int = 10,
) -> float:
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


def evaluate_eot(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.094,
    steps: int = 20,
    n_eot: int = 5,
) -> float:
    model.eval()
    correct = 0
    total = 0
    alpha = (eps * 2.0) / float(max(steps, 1))
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = eot_pgd_attack(model, x, y, eps=eps, alpha=alpha,
                                steps=steps, n_eot=n_eot, clamp_min=-4.0, clamp_max=4.0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_adv)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return float(correct) / float(max(total, 1))


def run_sweep(model: nn.Module, val_ds: datasets.ImageFolder, device: torch.device,
              batch_size: int = 48) -> Dict[str, Any]:
    eps_pixel_2 = (2.0 / 255.0) / MEAN_STD
    eps_pixel_4 = (4.0 / 255.0) / MEAN_STD
    conditions = {
        "clean": 0.0,
        "norm_eps_0031": 0.031,
        "norm_eps_0062": 0.062,
        "norm_eps_0094": 0.094,
        "pix_eps_2_255": eps_pixel_2,
        "pix_eps_4_255": eps_pixel_4,
    }
    per_cond: Dict[str, List[float]] = {k: [] for k in conditions}

    for s_idx, seed in enumerate(DEFAULT_SEEDS):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(val_ds), generator=g)[:EVAL_SAMPLE_SIZE].tolist()
        loader = DataLoader(Subset(val_ds, indices), batch_size=batch_size,
                            shuffle=False, num_workers=2)
        sys.stdout.write(f"  Seed {s_idx+1}/{len(DEFAULT_SEEDS)} (seed={seed}): ")
        sys.stdout.flush()
        for cond, eps in conditions.items():
            acc = evaluate_clean_and_pgd(model, loader, device, eps=eps,
                                          pgd_steps=10 if eps > 0 else 0)
            per_cond[cond].append(acc)
            sys.stdout.write(f"{cond.split('_')[0]}={acc*100:.1f}% ")
            sys.stdout.flush()
        print()

    summary = {}
    for cond, accs in per_cond.items():
        t = torch.tensor(accs)
        summary[cond] = {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "per_seed": accs,
        }
    return summary


def run_final_protocol(model: nn.Module, val_ds: datasets.ImageFolder,
                        device: torch.device, batch_size: int = 32) -> Dict[str, float]:
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(val_ds), generator=g)[:FINAL_EVAL_SAMPLE_SIZE].tolist()
    loader = DataLoader(Subset(val_ds, indices), batch_size=batch_size,
                        shuffle=False, num_workers=2)

    print("  [Final] Clean (500 imgs)...")
    t0 = time.time()
    clean = evaluate_clean_and_pgd(model, loader, device, eps=0.0, pgd_steps=0)
    print(f"    Clean: {clean*100:.2f}% ({time.time()-t0:.1f}s)")

    print("  [Final] PGD-50 @ eps=0.094 (500 imgs)...")
    t0 = time.time()
    pgd50 = evaluate_clean_and_pgd(model, loader, device, eps=0.094, pgd_steps=50)
    print(f"    PGD-50: {pgd50*100:.2f}% ({time.time()-t0:.1f}s)")

    print("  [Final] EOT-PGD n_eot=5 steps=20 @ eps=0.094 (500 imgs)...")
    t0 = time.time()
    eot = evaluate_eot(model, loader, device, eps=0.094, steps=20, n_eot=5)
    print(f"    EOT-PGD: {eot*100:.2f}% ({time.time()-t0:.1f}s)")

    return {"clean": clean, "pgd50": pgd50, "eot_pgd": eot}


def main():
    parser = argparse.ArgumentParser(description="G2-D3 Final Checkpoint Evaluation")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--output-dir", type=str, default="report/d3_final_eval")
    parser.add_argument("--batch-size-sweep", type=int, default=48)
    parser.add_argument("--batch-size-final", type=int, default=32)
    parser.add_argument("--skip-sweep", action="store_true", help="Skip 8-seed sweep, only run final protocol")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-D3] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    val_ds = get_val_dataset(args.data_root)
    print(f"[G2-D3] Val dataset: {len(val_ds)} images")

    all_results = {}

    for ckpt_info in CHECKPOINTS:
        name = ckpt_info["name"]
        ckpt_path = ckpt_info["path"]
        phase = ckpt_info["phase"]
        arm = ckpt_info["arm"]

        if not os.path.exists(ckpt_path):
            print(f"[G2-D3] BLOCKER: {ckpt_path} not found — skipping {name}")
            all_results[name] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        ckpt_size = os.path.getsize(ckpt_path)
        print(f"\n{'='*70}")
        print(f"[G2-D3] Evaluating: {name}")
        print(f"  path={ckpt_path} ({ckpt_size:,} bytes)")
        print(f"  phase={phase}, arm={arm}")

        model = Gen2FoundationModel(phase=phase, arm=arm).to(device)
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved
        model.load_state_dict(sd, strict=False)
        model.eval()

        # Print checkpoint metadata if available
        meta_keys = [k for k in saved if k != "model"]
        print(f"  Checkpoint metadata keys: {meta_keys}")
        for k in ["epoch", "phase", "best_clean_acc", "best_robust_acc", "step"]:
            if k in saved:
                print(f"  {k}: {saved[k]}")

        result = {
            "checkpoint_path": ckpt_path,
            "checkpoint_size_bytes": ckpt_size,
            "phase": phase,
            "arm": arm,
            "role": ckpt_info["role"],
        }

        if not args.skip_sweep:
            print(f"\n--- 8-Seed Sweep ({len(DEFAULT_SEEDS)} seeds × {EVAL_SAMPLE_SIZE} imgs) ---")
            t0 = time.time()
            sweep = run_sweep(model, val_ds, device, batch_size=args.batch_size_sweep)
            result["sweep"] = sweep
            result["sweep_runtime_s"] = time.time() - t0
            print(f"Sweep complete in {result['sweep_runtime_s']:.1f}s")

        print(f"\n--- Final Protocol (seed=42, n={FINAL_EVAL_SAMPLE_SIZE}) ---")
        t0 = time.time()
        final = run_final_protocol(model, val_ds, device, batch_size=args.batch_size_final)
        result["final"] = final
        result["final_runtime_s"] = time.time() - t0
        print(f"Final protocol complete in {result['final_runtime_s']:.1f}s")

        all_results[name] = result

        del model
        torch.cuda.empty_cache()

        # Save per-checkpoint JSON
        out_path = os.path.join(args.output_dir, f"g2_d3_{name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  ✓ Saved: {out_path}")

    # Comparison table
    print("\n" + "=" * 80)
    print("G2-D3 FINAL CHECKPOINT COMPARISON")
    print("=" * 80)
    print(f"{'Checkpoint':<30} | {'Clean':>8} | {'PGD-50':>8} | {'EOT-PGD':>8}")
    print("-" * 65)
    for name, res in all_results.items():
        if "final" in res:
            f = res["final"]
            print(f"{name:<30} | {f['clean']*100:>7.2f}% | {f['pgd50']*100:>7.2f}% | {f['eot_pgd']*100:>7.2f}%")
        else:
            print(f"{name:<30} | {'ERROR':>8} | {'ERROR':>8} | {'ERROR':>8}")

    # If both have sweep results, print clean comparison
    if all(("sweep" in r) for r in all_results.values() if "error" not in r):
        print("\n--- 8-Seed Sweep Clean Mean Acc ---")
        for name, res in all_results.items():
            if "sweep" in res:
                c = res["sweep"]["clean"]
                r = res["sweep"]["norm_eps_0094"]
                print(f"  {name}: clean={c['mean']*100:.2f}±{c['std']*100:.2f}%  "
                      f"pgd10@0094={r['mean']*100:.2f}±{r['std']*100:.2f}%")

    print("=" * 80)

    # Save combined
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(args.output_dir, f"g2_d3_combined_{ts}.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined report: {combined_path}")


if __name__ == "__main__":
    main()
