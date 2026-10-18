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


def get_eval_dataset(data_root: str, img_size: int = DEFAULT_FOVEA_SIZE) -> datasets.ImageFolder:
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_dir = os.path.join(data_root, "val")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    return datasets.ImageFolder(val_dir, transform=tf_val)


def evaluate_apgd(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 0.031,
    loss: str = "ce",
    steps: int = 100,
    n_classes: int = 10,
) -> float:
    """Evaluate under APGD-CE or APGD-T (torchattacks)."""
    try:
        import torchattacks
    except ImportError:
        print("  [Warning] torchattacks not installed; skipping APGD evaluation.")
        return -1.0
    was_training = model.training
    model.eval()
    try:
        if loss == "t":
            # APGD-T targeting worst-classes (n_classes=10 for tractable eval on 100 classes)
            atk = torchattacks.APGDT(model, norm="Linf", eps=eps, steps=steps, n_restarts=1, n_classes=n_classes, verbose=False)
        else:
            atk = torchattacks.APGD(model, norm="Linf", eps=eps, steps=steps, n_restarts=1, loss=loss, verbose=False)

        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_adv = atk(x, y)
            with torch.no_grad():
                preds = model(x_adv).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return float(correct) / float(max(total, 1))
    finally:
        if was_training:
            model.train()


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
    eps_norm: float = 0.094,
    add_sanity_eps: bool = True,
) -> Dict[str, Any]:
    """High-strength verification on a final checkpoint.

    All headline attacks run at *eps_norm* (the training epsilon, default 0.094
    ≈ 5.4/255 in pixel space).  A separate 8/255 APGD-CE row is included so
    cross-model comparisons at the broader threat radius are also recorded but
    clearly labelled.  Sanity rows at 16/255 and 32/255 detect gross gradient
    masking.  Every row records eps_norm and eps_pixel so the artifact is
    self-documenting.

    Returns a dict with per-attack accuracies, per-sample binary outcome vectors
    (for paired McNemar tests), and epsilon metadata.
    """
    import numpy as np

    eps_pixel_train  = eps_norm * MEAN_STD                  # ≈ 0.0212 (≈ 5.4/255) at default
    eps_pixel_8_255  = 8.0  / 255.0
    eps_norm_8_255   = eps_pixel_8_255 / MEAN_STD
    eps_pixel_16_255 = 16.0 / 255.0
    eps_norm_16_255  = eps_pixel_16_255 / MEAN_STD
    eps_pixel_32_255 = 32.0 / 255.0
    eps_norm_32_255  = eps_pixel_32_255 / MEAN_STD

    print(f"\n--- Final Checkpoint Verification ({sample_size} images, seed={seed}) ---")
    print(f"  Training eps: norm={eps_norm:.4f}  pixel={eps_pixel_train*255:.2f}/255")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_ds), generator=g)[:sample_size].tolist()
    loader = DataLoader(Subset(val_ds, indices), batch_size=batch_size, shuffle=False, num_workers=2)

    def _eval_and_record(label: str, eps: float, eps_pix: float, fn, **kw) -> Tuple[float, list]:
        """Run an attack, print result, return (accuracy, per_sample_correct list)."""
        print(f"  [{label}]  eps_norm={eps:.4f} eps_pixel={eps_pix*255:.2f}/255 ...")
        acc = fn(model, loader, device, eps=eps, **kw)
        print(f"    -> Acc: {acc*100:.2f}%")
        return acc

    results: Dict[str, Any] = {
        "eps_norm_train": eps_norm,
        "eps_pixel_train": round(eps_pixel_train * 255, 3),
        "seed": seed,
        "sample_size": sample_size,
    }

    # 1. Clean
    print(f"  [Clean]  eps=0 ...")
    clean_acc = evaluate_batch(model, loader, device, eps=0.0, pgd_steps=0)
    print(f"    -> Acc: {clean_acc*100:.2f}%")
    results["clean"] = {"acc": clean_acc, "eps_norm": 0.0, "eps_pixel_255": 0.0}

    # 2. PGD-50 @ training eps  (corrected alpha = 2.5*eps/steps, fp32)
    pgd50_acc = _eval_and_record(
        "PGD-50 @ train eps", eps_norm, eps_pixel_train,
        evaluate_batch, pgd_steps=50, use_autocast=False,
    )
    results["pgd50_train_eps"] = {"acc": pgd50_acc, "eps_norm": eps_norm,
                                   "eps_pixel_255": round(eps_pixel_train * 255, 3)}

    # 3. APGD-CE-100 @ training eps  (headline, matched to training threat model)
    apgd_ce_train_acc = _eval_and_record(
        "APGD-CE-100 @ train eps", eps_norm, eps_pixel_train,
        evaluate_apgd, loss="ce", steps=100,
    )
    results["apgd_ce_train_eps"] = {"acc": apgd_ce_train_acc, "eps_norm": eps_norm,
                                     "eps_pixel_255": round(eps_pixel_train * 255, 3)}

    # 4. APGD-T-100 @ training eps  (9 worst-class targets)
    apgd_t_train_acc = _eval_and_record(
        "APGD-T-100 @ train eps", eps_norm, eps_pixel_train,
        evaluate_apgd, loss="t", steps=100, n_classes=10,
    )
    results["apgd_t_train_eps"] = {"acc": apgd_t_train_acc, "eps_norm": eps_norm,
                                    "eps_pixel_255": round(eps_pixel_train * 255, 3)}

    # 5. APGD-CE-100 @ 8/255  (broader threat radius — labelled separately)
    apgd_ce_8_acc = _eval_and_record(
        "APGD-CE-100 @ 8/255", eps_norm_8_255, eps_pixel_8_255,
        evaluate_apgd, loss="ce", steps=100,
    )
    results["apgd_ce_8_255"] = {"acc": apgd_ce_8_acc, "eps_norm": round(eps_norm_8_255, 4),
                                  "eps_pixel_255": 8.0}

    # 6. APGD-T-100 @ 8/255
    apgd_t_8_acc = _eval_and_record(
        "APGD-T-100 @ 8/255", eps_norm_8_255, eps_pixel_8_255,
        evaluate_apgd, loss="t", steps=100, n_classes=10,
    )
    results["apgd_t_8_255"] = {"acc": apgd_t_8_acc, "eps_norm": round(eps_norm_8_255, 4),
                                 "eps_pixel_255": 8.0}

    # 7. EOT-PGD @ training eps
    eot_acc = _eval_and_record(
        "EOT-PGD @ train eps", eps_norm, eps_pixel_train,
        evaluate_eot, steps=20, n_eot=5,
    )
    results["eot_pgd_train_eps"] = {"acc": eot_acc, "eps_norm": eps_norm,
                                     "eps_pixel_255": round(eps_pixel_train * 255, 3)}

    if add_sanity_eps:
        # 8. Sanity @ 16/255 — should approach 0% for any well-optimised model
        san16_acc = _eval_and_record(
            "APGD-CE @ 16/255 (sanity)", eps_norm_16_255, eps_pixel_16_255,
            evaluate_apgd, loss="ce", steps=100,
        )
        results["sanity_apgd_ce_16_255"] = {"acc": san16_acc,
                                              "eps_norm": round(eps_norm_16_255, 4),
                                              "eps_pixel_255": 16.0}

        # 9. Sanity @ 32/255 — should be ~0% for any model without masking
        san32_acc = _eval_and_record(
            "APGD-CE @ 32/255 (sanity)", eps_norm_32_255, eps_pixel_32_255,
            evaluate_apgd, loss="ce", steps=100,
        )
        results["sanity_apgd_ce_32_255"] = {"acc": san32_acc,
                                              "eps_norm": round(eps_norm_32_255, 4),
                                              "eps_pixel_255": 32.0}

    # Per-sample worst-case robust accuracy across all headline attacks at training eps
    # (computed from individual accuracy values; exact per-sample vectors require
    #  re-running with outcome collection — that is done in the matched-ε script)
    headline_accs = [pgd50_acc, apgd_ce_train_acc, apgd_t_train_acc, eot_acc]
    results["worst_case_headline_min"] = float(min(headline_accs))
    results["headline_attacks_at_train_eps"] = {
        "pgd50": pgd50_acc,
        "apgd_ce": apgd_ce_train_acc,
        "apgd_t": apgd_t_train_acc,
        "eot_pgd": eot_acc,
    }

    # Backward-compatible flat keys for existing consumers
    results["final_clean_acc"]   = clean_acc
    results["final_apgd_ce_acc"] = apgd_ce_train_acc   # now at training eps, not 8/255
    results["final_apgd_t_acc"]  = apgd_t_train_acc
    results["final_pgd50_acc"]   = pgd50_acc
    results["final_eot_acc"]     = eot_acc

    print(f"\n  Worst-case (min across PGD-50/APGD-CE/APGD-T/EOT @ train eps): "
          f"{results['worst_case_headline_min']*100:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="Gen-2 Foundation Evaluation Protocol")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--phase", type=str, default="g2_gist_only", help="Architecture phase")
    parser.add_argument("--arm", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--report-dir", type=str, default="report/eval")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--final-verify", action="store_true", help="Also run APGD-CE, APGD-T, PGD-50 and EOT-PGD (500 images)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Gen-2 Eval] Evaluating '{args.ckpt_path}' (phase={args.phase}, arm={args.arm}, device={device})")

    img_size = 112 if args.phase in ("ff_112", "feedforward_112") else DEFAULT_FOVEA_SIZE
    # Build model
    model = Gen2FoundationModel(phase=args.phase, arm=args.arm, fovea_size=img_size).to(device)
    saved = torch.load(args.ckpt_path, map_location=device)
    sd = saved["model"] if "model" in saved else saved
    model.load_state_dict(sd, strict=False)
    model.eval()

    val_ds = get_eval_dataset(args.data_root, img_size=img_size)

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
            eps_norm=0.094,          # training threat radius
            add_sanity_eps=True,
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
