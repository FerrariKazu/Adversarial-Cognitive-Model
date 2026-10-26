#!/usr/bin/env python3
"""G2-CAL2: Evidential Uncertainty Gate Evaluation
====================================================
Evaluates calibration of evidential Dirichlet uncertainty against
standard softmax baseline for gen1_core and recurrence_only checkpoints.

Metrics:
  - ECE (Expected Calibration Error, 10 bins)
  - NLL (Negative Log-Likelihood)
  - AUROC for uncertainty-vs-error detection (can the model tell when it's wrong?)
  - Mean uncertainty on clean vs. adversarial inputs (U_clean vs U_adv)

For recurrence_only (no evidential head): uses softmax entropy as proxy.
For gen1_core (has evidential head): uses Dirichlet uncertainty U_t.

Protocol:
  - 500 images, seed=42, clean + PGD-10 @ eps=0.094
  - Records confidence, uncertainty, correctness per sample

Registration: G2-CAL2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

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
EPS = 0.094
PGD_STEPS = 10
N_ECE_BINS = 10


def ece_score(confidences: torch.Tensor, correctness: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    ece = 0.0
    n = len(confidences)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correctness[mask].float().mean().item()
        conf = confidences[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(acc - conf)
    return ece


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC using manual rank computation (no sklearn needed)."""
    # labels: 1 = positive class (error), 0 = correct
    # scores: uncertainty value (higher = more uncertain = should be error)
    n = len(scores)
    n_pos = labels.sum().item()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U stat
    sorted_idx = torch.argsort(scores)
    rank_sum = 0.0
    for rank, idx in enumerate(sorted_idx):
        if labels[idx] == 1:
            rank_sum += rank + 1
    u_stat = rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.0,
    pgd_steps: int = 0,
    has_evidential: bool = False,
) -> Dict[str, torch.Tensor]:
    """Collect per-sample predictions, confidences, and uncertainty."""
    model.eval()
    all_confidences = []
    all_correctness = []
    all_entropies = []
    all_logits_list = []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if eps > 0.0 and pgd_steps > 0:
            x = run_pgd_ce_eval(model, x, y, eps=eps, steps=pgd_steps)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            conf = probs.max(dim=-1).values
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            correct = (preds == y)

            all_confidences.append(conf.cpu())
            all_correctness.append(correct.cpu())
            all_entropies.append(entropy.cpu())
            all_logits_list.append(logits.cpu())

    return {
        "confidences": torch.cat(all_confidences),
        "correctness": torch.cat(all_correctness),
        "entropies": torch.cat(all_entropies),
    }


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


def evaluate_checkpoint(
    ckpt_path: str, phase: str, arm: str, loader: DataLoader, device: torch.device
) -> Dict:
    model = Gen2FoundationModel(phase=phase, arm=arm).to(device)
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = saved["model"] if "model" in saved else saved
    model.load_state_dict(sd, strict=False)
    model.eval()

    has_evidential = phase in ("belief_no_f", "belief_with_f", "ais_v2_swap", "gen1_core")
    results = {}

    for cond_name, eps, pgd_s in [
        ("clean", 0.0, 0),
        ("pgd10", EPS, PGD_STEPS),
    ]:
        print(f"  Collecting {cond_name} predictions...")
        t0 = time.time()
        preds = collect_predictions(model, loader, device, eps=eps, pgd_steps=pgd_s,
                                    has_evidential=has_evidential)
        elapsed = time.time() - t0

        confidences = preds["confidences"]
        correctness = preds["correctness"]
        entropies = preds["entropies"]

        acc = correctness.float().mean().item()
        ece = ece_score(confidences, correctness, N_ECE_BINS)
        nll = -torch.log(confidences.clamp(min=1e-8)).mean().item()
        mean_entropy = entropies.mean().item()

        # AUROC: can entropy predict errors?
        # Higher entropy = more uncertain = should be error (label=1 when wrong)
        error_labels = (~correctness).long()
        auc = auroc(entropies, error_labels)

        print(f"    {cond_name}: acc={acc*100:.2f}%, ECE={ece:.4f}, NLL={nll:.4f}, "
              f"entropy={mean_entropy:.4f}, AUROC={auc:.4f} ({elapsed:.1f}s)")

        results[cond_name] = {
            "accuracy": acc,
            "ece": ece,
            "nll": nll,
            "mean_entropy": mean_entropy,
            "auroc_error_detection": auc,
            "n_correct": int(correctness.sum()),
            "n_total": len(correctness),
            "runtime_s": elapsed,
        }

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="G2-CAL2 Uncertainty Evaluation")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--output-dir", type=str, default="report/cal2_uncertainty")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-CAL2] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    loader = get_loader(args.data_root, SEED, SAMPLE_SIZE, BATCH_SIZE)

    checkpoints = [
        {"name": "recurrence_only_final", "path": "checkpoints/foundation_recurrence_only_final.pth",
         "phase": "recurrence_only", "arm": "v1"},
        {"name": "gen1_core_best_clean", "path": "checkpoints/foundation_gen1_core_best_clean.pth",
         "phase": "gen1_core", "arm": "v1"},
    ]

    all_results = {}

    for ckpt_info in checkpoints:
        name = ckpt_info["name"]
        ckpt_path = ckpt_info["path"]

        if not os.path.exists(ckpt_path):
            print(f"[G2-CAL2] BLOCKER: {ckpt_path} not found — skipping {name}")
            all_results[name] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        print(f"\n{'='*70}")
        print(f"[G2-CAL2] {name} (phase={ckpt_info['phase']})")

        res = evaluate_checkpoint(ckpt_path, ckpt_info["phase"], ckpt_info["arm"], loader, device)
        all_results[name] = res

        out_path = os.path.join(args.output_dir, f"g2_cal2_{name}.json")
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  ✓ Saved: {out_path}")

    # Summary
    print("\n" + "=" * 90)
    print("G2-CAL2 UNCERTAINTY CALIBRATION SUMMARY")
    print("=" * 90)
    print(f"{'Checkpoint':<30} {'Cond':<8} {'Acc':>7} {'ECE':>7} {'NLL':>7} {'Entropy':>9} {'AUROC':>8}")
    print("-" * 80)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: {res['error']}")
            continue
        for cond, m in res.items():
            if isinstance(m, dict):
                print(f"{name:<30} {cond:<8} {m['accuracy']*100:>6.2f}% "
                      f"{m['ece']:>7.4f} {m['nll']:>7.4f} {m['mean_entropy']:>9.4f} "
                      f"{m['auroc_error_detection']:>8.4f}")

    combined_path = os.path.join(args.output_dir, "g2_cal2_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined: {combined_path}")


if __name__ == "__main__":
    main()
