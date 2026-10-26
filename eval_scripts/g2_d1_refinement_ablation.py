#!/usr/bin/env python3
"""G2-D1: Refinement Block Ablation Experiment
=================================================
Tests whether the within-glimpse TiedRecurrence refinement block is functionally
active in recurrence_only checkpoints by comparing:

  (A) ACTIVE   — normal forward pass (refinement block engaged)
  (B) BYPASSED — backbone.refinement replaced with identity (no refinement)
  (C) ZEROED   — refinement LayerScale gammas zeroed (soft ablation)

Evaluation on fixed 500-image seed=42 validation split.
Reports accuracy delta and effect size.

Usage:
  python eval_scripts/g2_d1_refinement_ablation.py \
      --ckpt-path checkpoints/foundation_recurrence_only_final.pth \
      --data-root data/imagenet100 \
      --output-dir report/d1_refinement_ablation

Registration: G2-D1
Protocol: 500 images, seed=42, clean + PGD-10 @ eps=0.094
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
PGD_STEPS = 10
EPS = 0.094


class IdentityRefinement(nn.Module):
    """Drop-in replacement for TiedRecurrence that is a pure identity."""
    def forward(self, x: torch.Tensor, num_iters: int = 2) -> torch.Tensor:
        return x


def load_model(ckpt_path: str, device: torch.device) -> Gen2FoundationModel:
    model = Gen2FoundationModel(phase="recurrence_only").to(device)
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = saved["model"] if "model" in saved else saved
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_loader(data_root: str, seed: int, sample_size: int, batch_size: int) -> DataLoader:
    tf_val = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_dir = os.path.join(data_root, "val")
    val_ds = datasets.ImageFolder(val_dir, transform=tf_val)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_ds), generator=g)[:sample_size].tolist()
    sub = Subset(val_ds, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             eps: float = 0.0, pgd_steps: int = 0) -> Tuple[float, int]:
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
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return float(correct) / float(max(total, 1)), total


def zero_refinement_gammas(model: Gen2FoundationModel) -> int:
    """Zero all LayerScale gammas in the refinement block. Returns number of params zeroed."""
    count = 0
    for name, param in model.backbone.refinement.named_parameters():
        if "gamma" in name:
            with torch.no_grad():
                param.fill_(0.0)
            count += param.numel()
    return count


def main():
    parser = argparse.ArgumentParser(description="G2-D1 Refinement Ablation")
    parser.add_argument("--ckpt-path", type=str,
                        default="checkpoints/foundation_recurrence_only_final.pth")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--output-dir", type=str, default="report/d1_refinement_ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-D1] Device: {device}")

    # Verify checkpoint
    ckpt_size = os.path.getsize(args.ckpt_path)
    print(f"[G2-D1] Checkpoint: {args.ckpt_path} ({ckpt_size:,} bytes)")

    # Build fixed data loader
    loader = build_loader(args.data_root, SEED, SAMPLE_SIZE, BATCH_SIZE)
    print(f"[G2-D1] Data loader: seed={SEED}, n={SAMPLE_SIZE}, batch_size={BATCH_SIZE}")

    results: Dict = {
        "checkpoint": args.ckpt_path,
        "checkpoint_size_bytes": ckpt_size,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "pgd_steps": PGD_STEPS,
        "eps": EPS,
        "conditions": {}
    }

    # ─── Condition A: ACTIVE (normal) ───────────────────────────────────────
    print("\n=== Condition A: Refinement ACTIVE (normal forward) ===")
    model_a = load_model(args.ckpt_path, device)

    # Inspect gamma norms before running
    gamma_norms = {}
    for name, param in model_a.backbone.refinement.named_parameters():
        if "gamma" in name:
            gamma_norms[name] = {
                "norm": float(param.norm().item()),
                "mean": float(param.mean().item()),
                "std": float(param.std().item()),
                "min": float(param.min().item()),
                "max": float(param.max().item()),
            }
    print(f"  Refinement LayerScale gammas: {gamma_norms}")

    t0 = time.time()
    clean_a, n_a = evaluate(model_a, loader, device, eps=0.0, pgd_steps=0)
    t_clean_a = time.time() - t0
    print(f"  Clean acc: {clean_a*100:.2f}% (n={n_a}, {t_clean_a:.1f}s)")

    t0 = time.time()
    robust_a, _ = evaluate(model_a, loader, device, eps=EPS, pgd_steps=PGD_STEPS)
    t_robust_a = time.time() - t0
    print(f"  PGD-{PGD_STEPS} @ eps={EPS}: {robust_a*100:.2f}% ({t_robust_a:.1f}s)")

    results["conditions"]["A_active"] = {
        "clean_acc": clean_a,
        "robust_acc": robust_a,
        "refinement_gamma_norms": gamma_norms,
        "runtime_clean_s": t_clean_a,
        "runtime_robust_s": t_robust_a,
    }
    del model_a
    torch.cuda.empty_cache()

    # ─── Condition B: BYPASSED (identity replacement) ────────────────────────
    print("\n=== Condition B: Refinement BYPASSED (identity) ===")
    model_b = load_model(args.ckpt_path, device)
    model_b.backbone.refinement = IdentityRefinement().to(device)

    t0 = time.time()
    clean_b, n_b = evaluate(model_b, loader, device, eps=0.0, pgd_steps=0)
    t_clean_b = time.time() - t0
    print(f"  Clean acc: {clean_b*100:.2f}% (n={n_b}, {t_clean_b:.1f}s)")

    t0 = time.time()
    robust_b, _ = evaluate(model_b, loader, device, eps=EPS, pgd_steps=PGD_STEPS)
    t_robust_b = time.time() - t0
    print(f"  PGD-{PGD_STEPS} @ eps={EPS}: {robust_b*100:.2f}% ({t_robust_b:.1f}s)")

    results["conditions"]["B_bypassed"] = {
        "clean_acc": clean_b,
        "robust_acc": robust_b,
        "runtime_clean_s": t_clean_b,
        "runtime_robust_s": t_robust_b,
    }
    del model_b
    torch.cuda.empty_cache()

    # ─── Condition C: ZEROED gammas (soft ablation) ──────────────────────────
    print("\n=== Condition C: Refinement ZEROED (LayerScale gammas = 0) ===")
    model_c = load_model(args.ckpt_path, device)
    n_zeroed = zero_refinement_gammas(model_c)
    print(f"  Zeroed {n_zeroed} gamma parameters in refinement block")

    t0 = time.time()
    clean_c, n_c = evaluate(model_c, loader, device, eps=0.0, pgd_steps=0)
    t_clean_c = time.time() - t0
    print(f"  Clean acc: {clean_c*100:.2f}% (n={n_c}, {t_clean_c:.1f}s)")

    t0 = time.time()
    robust_c, _ = evaluate(model_c, loader, device, eps=EPS, pgd_steps=PGD_STEPS)
    t_robust_c = time.time() - t0
    print(f"  PGD-{PGD_STEPS} @ eps={EPS}: {robust_c*100:.2f}% ({t_robust_c:.1f}s)")

    results["conditions"]["C_zeroed"] = {
        "clean_acc": clean_c,
        "robust_acc": robust_c,
        "n_zeroed_params": n_zeroed,
        "runtime_clean_s": t_clean_c,
        "runtime_robust_s": t_robust_c,
    }
    del model_c
    torch.cuda.empty_cache()

    # ─── Delta Analysis ───────────────────────────────────────────────────────
    delta_clean_AB = clean_a - clean_b
    delta_robust_AB = robust_a - robust_b
    delta_clean_AC = clean_a - clean_c
    delta_robust_AC = robust_a - robust_c

    print("\n" + "=" * 70)
    print("G2-D1 REFINEMENT ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<22} | {'Clean Acc':>10} | {'Robust Acc':>10}")
    print("-" * 50)
    print(f"{'A: Active':22} | {clean_a*100:>9.2f}% | {robust_a*100:>9.2f}%")
    print(f"{'B: Bypassed (identity)':22} | {clean_b*100:>9.2f}% | {robust_b*100:>9.2f}%")
    print(f"{'C: Zeroed (LS gamma=0)':22} | {clean_c*100:>9.2f}% | {robust_c*100:>9.2f}%")
    print("-" * 50)
    print(f"  Δ(A-B) clean={delta_clean_AB*100:+.2f}%  robust={delta_robust_AB*100:+.2f}%  [identity vs active]")
    print(f"  Δ(A-C) clean={delta_clean_AC*100:+.2f}%  robust={delta_robust_AC*100:+.2f}%  [zeroed gamma vs active]")

    if abs(delta_clean_AB) < 0.005 and abs(delta_robust_AB) < 0.005:
        verdict = "INERT — refinement contributes <0.5pp to both clean and robust acc. Can be removed."
    elif abs(delta_clean_AB) < 0.01:
        verdict = "MARGINAL — refinement contributes <1pp clean but may differ on robust."
    else:
        verdict = "ACTIVE — refinement contributes measurably to model performance."

    print(f"\nVERDICT: {verdict}")
    print("=" * 70)

    results["deltas"] = {
        "A_minus_B_clean": delta_clean_AB,
        "A_minus_B_robust": delta_robust_AB,
        "A_minus_C_clean": delta_clean_AC,
        "A_minus_C_robust": delta_robust_AC,
    }
    results["verdict"] = verdict

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "g2_d1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
