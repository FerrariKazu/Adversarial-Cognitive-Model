#!/usr/bin/env python3
"""Gen-2 Foundation Evaluation Protocol — training/eval_gen2_foundation.py
================================================================================

Implements the S7 Per-Phase Evaluation Protocol:
  * 8 seeds × 300 validation images per seed.
  * Perturbation grid:
      - Clean (eps = 0.0)
      - Norm-space: eps in {0.031, 0.062, 0.094}
      - Pixel-space: {2/255, 4/255} (mapped to normalized space via dataset std)
  * Final checkpoint protocol:
      - PGD-50 (50 steps) on 500 images
      - EOT-PGD (n_eot=5) on 500 images
  * Outputs detailed per-seed and aggregated statistics (mean, std).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def get_eval_dataset(data_root: str) -> datasets.ImageFolder:
    tf_val = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_dir = os.path.join(data_root, "val")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    return datasets.ImageFolder(val_dir, transform=tf_val)


def evaluate_batch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.0,
    pgd_steps: int = 10,
    use_autocast: bool = True,
) -> float:
    """Evaluate accuracy on loader. If eps > 0, attacks with PGD-ce."""
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if eps > 0.0 and pgd_steps > 0:
            x_eval = run_pgd_ce_eval(model, x, y, eps=eps, steps=pgd_steps)
        else:
            x_eval = x

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(use_autocast and device.type == "cuda")):
                logits = model(x_eval)
            preds = logits.argmax(dim=-1)
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
    use_autocast: bool = True,
) -> float:
    """Evaluate accuracy under EOT-PGD attack."""
    model.eval()
    correct = 0
    total = 0
    alpha = (eps * 2.0) / float(max(steps, 1))

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # EOT-PGD evaluated outside autocast in fp32
        x_adv = eot_pgd_attack(
            model,
            x,
            y,
            eps=eps,
            alpha=alpha,
            steps=steps,
            n_eot=n_eot,
            clamp_min=-4.0,
            clamp_max=4.0,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(use_autocast and device.type == "cuda")):
                logits = model(x_adv)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return float(correct) / float(max(total, 1))


def run_per_phase_eval(
    model: nn.Module,
    val_ds: datasets.ImageFolder,
    device: torch.device,
    seeds: List[int] = DEFAULT_SEEDS,
    sample_size: int = EVAL_SAMPLE_SIZE,
    batch_size: int = 48,
    pgd_steps: int = 10,
) -> Dict[str, Any]:
    """Runs 8 seeds × 300 images across the standardized perturbation grid."""
    # Define test conditions: name -> norm_eps
    eps_pixel_2_255 = (2.0 / 255.0) / MEAN_STD
    eps_pixel_4_255 = (4.0 / 255.0) / MEAN_STD

    conditions = {
        "clean (eps=0.0)": 0.0,
        "norm_eps=0.031": 0.031,
        "norm_eps=0.062": 0.062,
        "norm_eps=0.094": 0.094,
        "pix_eps=2/255": eps_pixel_2_255,
        "pix_eps=4/255": eps_pixel_4_255,
    }

    per_condition_seed_results: Dict[str, List[float]] = {cond: [] for cond in conditions}

    print(f"\n--- Running Per-Phase Evaluation ({len(seeds)} seeds × {sample_size} images) ---")
    for s_idx, seed in enumerate(seeds):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(val_ds), generator=g)[:sample_size].tolist()
        sub_ds = Subset(val_ds, indices)
        loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        sys.stdout.write(f"Seed {s_idx+1}/{len(seeds)} (seed={seed}): ")
        sys.stdout.flush()

        for cond_name, eps_val in conditions.items():
            acc = evaluate_batch(
                model=model,
                loader=loader,
                device=device,
                eps=eps_val,
                pgd_steps=pgd_steps if eps_val > 0.0 else 0,
            )
            per_condition_seed_results[cond_name].append(acc)
            sys.stdout.write(f"{cond_name.split()[0]}={acc*100:.1f}% ")
            sys.stdout.flush()
        print()

    # Aggregate statistics
    summary = {}
    print("\n" + "=" * 70)
    print(f"{'Condition':<22} | {'Mean Acc':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 70)
    for cond_name, accs in per_condition_seed_results.items():
        t_accs = torch.tensor(accs)
        mean_acc = float(t_accs.mean().item())
        std_acc = float(t_accs.std().item()) if len(accs) > 1 else 0.0
        min_acc = float(t_accs.min().item())
        max_acc = float(t_accs.max().item())

        summary[cond_name] = {
            "mean": mean_acc,
            "std": std_acc,
            "min": min_acc,
            "max": max_acc,
            "per_seed": accs,
        }
        print(f"{cond_name:<22} | {mean_acc*100:>9.2f}% | {std_acc*100:>7.2f}% | {min_acc*100:>7.2f}% | {max_acc*100:>7.2f}%")
    print("=" * 70)

    return summary


def run_final_checkpoint_eval(
    model: nn.Module,
    val_ds: datasets.ImageFolder,
    device: torch.device,
    seed: int = 42,
    sample_size: int = FINAL_EVAL_SAMPLE_SIZE,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Runs high-strength verification on final checkpoint: PGD-50 and EOT-PGD."""
    print(f"\n--- Running Final Checkpoint Verification (500 images) ---")
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_ds), generator=g)[:sample_size].tolist()
    loader = DataLoader(Subset(val_ds, indices), batch_size=batch_size, shuffle=False, num_workers=2)

    # 1. Clean
    print("  Evaluating Clean (500 imgs)...")
    clean_acc = evaluate_batch(model, loader, device, eps=0.0, pgd_steps=0)
    print(f"    -> Clean Acc: {clean_acc*100:.2f}%")

    # 2. PGD-50 @ eps=0.094
    print("  Evaluating PGD-50 @ eps=0.094 (500 imgs)...")
    pgd50_acc = evaluate_batch(model, loader, device, eps=0.094, pgd_steps=50)
    print(f"    -> PGD-50 Acc: {pgd50_acc*100:.2f}%")

    # 3. EOT-PGD (n_eot=5, steps=20) @ eps=0.094
    print("  Evaluating EOT-PGD (n_eot=5, steps=20) @ eps=0.094 (500 imgs)...")
    eot_acc = evaluate_eot(model, loader, device, eps=0.094, steps=20, n_eot=5)
    print(f"    -> EOT-PGD Acc: {eot_acc*100:.2f}%")

    return {
        "final_clean_acc": clean_acc,
        "final_pgd50_acc": pgd50_acc,
        "final_eot_acc": eot_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Gen-2 Foundation Evaluation Protocol")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--phase", type=str, default="g2_gist_only", help="Architecture phase")
    parser.add_argument("--arm", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--report-dir", type=str, default="report/eval")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--final-verify", action="store_true", help="Also run PGD-50 and EOT-PGD (500 images)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Gen-2 Eval] Evaluating '{args.ckpt_path}' (phase={args.phase}, arm={args.arm}, device={device})")

    # Build model
    model = Gen2FoundationModel(phase=args.phase, arm=args.arm).to(device)
    saved = torch.load(args.ckpt_path, map_location=device)
    sd = saved["model"] if "model" in saved else saved
    model.load_state_dict(sd, strict=False)
    model.eval()

    val_ds = get_eval_dataset(args.data_root)

    # 1. 8-seed sweep
    sweep_summary = run_per_phase_eval(
        model=model,
        val_ds=val_ds,
        device=device,
        batch_size=args.batch_size,
    )

    final_results = {}
    if args.final_verify:
        final_results = run_final_checkpoint_eval(
            model=model,
            val_ds=val_ds,
            device=device,
            batch_size=args.batch_size,
        )

    # Save report
    os.makedirs(args.report_dir, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.report_dir, f"eval_{args.phase}_{ts}.json")
    report_data = {
        "timestamp": ts,
        "checkpoint": args.ckpt_path,
        "phase": args.phase,
        "arm": args.arm,
        "sweep_summary": sweep_summary,
        "final_verification": final_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n✓ Evaluation complete. Full report saved to: {out_file}")


if __name__ == "__main__":
    main()
