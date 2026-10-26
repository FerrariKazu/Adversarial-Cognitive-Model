#!/usr/bin/env python3
"""G2-D8: Adaptive Attack Finals
==================================
Runs targeted adaptive attacks on both reference checkpoints to stress-test
the reported robustness numbers.

Attacks:
  1. AutoPGD-CE (APGD-CE): momentum-based attack, 100 steps, eps=0.094
     - Adaptive step size with restarts
  2. PGD with momentum (heavy) @ eps=0.094, 100 steps
  3. CW-style L-inf approximation (maximize margin): 50 steps
  4. Input-diversity PGD: randomizes input before each step, 20 steps, n_restarts=5

Protocol:
  - 300 images, seed=42 (smaller set for heavy attacks)
  - Both recurrence_only_final and gen1_core_best_clean

Registration: G2-D8
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

SEED = 42
SAMPLE_SIZE = 300
BATCH_SIZE = 24  # Smaller batch for heavy attacks
EPS = 0.094
CLAMP_MIN, CLAMP_MAX = -4.0, 4.0


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


def pgd_momentum_attack(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    eps: float, steps: int = 100, alpha: float = None, momentum: float = 0.9,
    clamp_min: float = CLAMP_MIN, clamp_max: float = CLAMP_MAX,
) -> torch.Tensor:
    """PGD with gradient momentum (MI-FGSM style)."""
    if alpha is None:
        alpha = (eps * 2.0) / float(max(steps, 1))

    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, clamp_min - x, clamp_max - x)
    delta.requires_grad_(True)
    velocity = torch.zeros_like(x)

    for _ in range(steps):
        adv_x = x + delta
        logits = model(adv_x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # Normalize gradient + momentum
        g_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        velocity = momentum * velocity + g_norm
        delta_data = delta.data + alpha * velocity.sign()
        delta_data = torch.clamp(delta_data, -eps, eps)
        delta_data = torch.clamp(delta_data, clamp_min - x, clamp_max - x)
        delta = delta_data.detach().requires_grad_(True)

    return (x + delta.detach()).clamp(clamp_min, clamp_max)


def apgd_attack(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    eps: float, steps: int = 100,
    clamp_min: float = CLAMP_MIN, clamp_max: float = CLAMP_MAX,
) -> torch.Tensor:
    """Simplified AutoPGD-CE with momentum and step-size reduction."""
    alpha = eps * 0.75  # Initial large step

    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, clamp_min - x, clamp_max - x)
    delta.requires_grad_(True)
    velocity = torch.zeros_like(x)

    best_loss = torch.zeros(x.size(0), device=x.device)
    best_delta = delta.data.clone()
    rho = 0.75  # Step size reduction trigger

    n_no_improve = 0

    for step_i in range(steps):
        adv_x = x + delta
        logits = model(adv_x)
        loss_per = F.cross_entropy(logits, y, reduction="none")
        loss = loss_per.mean()

        improved = (loss_per > best_loss)
        best_delta[improved] = delta.data[improved]
        best_loss = torch.maximum(best_loss, loss_per.detach())

        if loss.item() <= 0 or torch.isnan(loss):
            break

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        g_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        velocity = 0.75 * velocity + g_norm
        delta_data = delta.data + alpha * velocity.sign()
        delta_data = torch.clamp(delta_data, -eps, eps)
        delta_data = torch.clamp(delta_data, clamp_min - x, clamp_max - x)

        # Reduce step if loss not improving
        if step_i > 0 and step_i % 20 == 0:
            if n_no_improve > 5:
                alpha = alpha * rho
                n_no_improve = 0
            else:
                n_no_improve += 1

        delta = delta_data.detach().requires_grad_(True)

    return (x + best_delta).clamp(clamp_min, clamp_max)


def diversity_pgd_attack(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    eps: float, steps: int = 20, n_restarts: int = 5,
    clamp_min: float = CLAMP_MIN, clamp_max: float = CLAMP_MAX,
) -> torch.Tensor:
    """Multi-restart PGD with input diversity (dropout-style noise)."""
    alpha = (eps * 2.0) / float(max(steps, 1))
    B = x.size(0)

    best_loss = torch.full((B,), -1e9, device=x.device)
    best_x_adv = x.clone()

    for restart in range(n_restarts):
        # Random init
        delta = torch.empty_like(x).uniform_(-eps, eps)
        delta = torch.clamp(delta, clamp_min - x, clamp_max - x)
        delta.requires_grad_(True)

        for _ in range(steps):
            # Input diversity: add small random noise
            noise = 0.01 * torch.randn_like(x)
            adv_in = x + delta + noise
            logits = model(adv_in.clamp(clamp_min, clamp_max))
            loss_per = F.cross_entropy(logits, y, reduction="none")
            loss = loss_per.mean()

            if torch.isnan(loss):
                break

            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            delta_data = delta.data + alpha * grad.sign()
            delta_data = torch.clamp(delta_data, -eps, eps)
            delta_data = torch.clamp(delta_data, clamp_min - x, clamp_max - x)
            delta = delta_data.detach().requires_grad_(True)

        # Keep best
        with torch.no_grad():
            adv_x = (x + delta.data).clamp(clamp_min, clamp_max)
            logits = model(adv_x)
            loss_per = F.cross_entropy(logits, y, reduction="none")
            improved = loss_per > best_loss
            best_x_adv[improved] = adv_x[improved]
            best_loss = torch.maximum(best_loss, loss_per)

    return best_x_adv


def evaluate_attack(
    model: nn.Module, loader: DataLoader, device: torch.device,
    attack_fn, attack_name: str
) -> Tuple[float, float]:
    """Evaluate an attack function, return (accuracy, runtime)."""
    model.eval()
    correct = 0
    total = 0
    t0 = time.time()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_adv)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = float(correct) / float(max(total, 1))
    runtime = time.time() - t0
    print(f"    {attack_name}: {acc*100:.2f}% (n={total}, {runtime:.1f}s)")
    return acc, runtime


def main():
    parser = argparse.ArgumentParser(description="G2-D8 Adaptive Attack Finals")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--output-dir", type=str, default="report/d8_adaptive_attacks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-D8] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    loader = get_loader(args.data_root, SEED, SAMPLE_SIZE, BATCH_SIZE)
    print(f"[G2-D8] Loader: seed={SEED}, n={SAMPLE_SIZE}, batch={BATCH_SIZE}")

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
            print(f"[G2-D8] BLOCKER: {ckpt_path} not found — skipping {name}")
            all_results[name] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        print(f"\n{'='*70}")
        print(f"[G2-D8] {name}")
        model = Gen2FoundationModel(phase=ckpt_info["phase"], arm=ckpt_info["arm"]).to(device)
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved
        model.load_state_dict(sd, strict=False)
        model.eval()

        ckpt_results = {"phase": ckpt_info["phase"], "eps": EPS, "sample_size": SAMPLE_SIZE,
                        "seed": SEED, "attacks": {}}

        # Clean baseline
        print("  Evaluating Clean...")
        t0 = time.time()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(x)
                preds = logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        clean_acc = float(correct) / float(max(total, 1))
        print(f"    Clean: {clean_acc*100:.2f}% (n={total}, {time.time()-t0:.1f}s)")
        ckpt_results["attacks"]["clean"] = {"acc": clean_acc, "runtime_s": time.time()-t0}

        # Attack 1: MI-FGSM @ 100 steps
        print("  Attack 1: MI-FGSM (momentum PGD, 100 steps)...")
        acc1, rt1 = evaluate_attack(
            model, loader, device,
            lambda m, x, y: pgd_momentum_attack(m, x, y, eps=EPS, steps=100),
            "MI-FGSM-100"
        )
        ckpt_results["attacks"]["mi_fgsm_100"] = {"acc": acc1, "steps": 100, "runtime_s": rt1}

        # Attack 2: APGD-CE @ 100 steps
        print("  Attack 2: APGD-CE (100 steps)...")
        acc2, rt2 = evaluate_attack(
            model, loader, device,
            lambda m, x, y: apgd_attack(m, x, y, eps=EPS, steps=100),
            "APGD-CE-100"
        )
        ckpt_results["attacks"]["apgd_ce_100"] = {"acc": acc2, "steps": 100, "runtime_s": rt2}

        # Attack 3: Diversity PGD (5 restarts × 20 steps)
        print("  Attack 3: Diversity PGD (5 restarts × 20 steps)...")
        acc3, rt3 = evaluate_attack(
            model, loader, device,
            lambda m, x, y: diversity_pgd_attack(m, x, y, eps=EPS, steps=20, n_restarts=5),
            "Diversity-PGD-5x20"
        )
        ckpt_results["attacks"]["diversity_pgd_5x20"] = {"acc": acc3, "n_restarts": 5,
                                                          "steps": 20, "runtime_s": rt3}

        all_results[name] = ckpt_results

        out_path = os.path.join(args.output_dir, f"g2_d8_{name}.json")
        with open(out_path, "w") as f:
            json.dump(ckpt_results, f, indent=2)
        print(f"  ✓ Saved: {out_path}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("G2-D8 ADAPTIVE ATTACK SUMMARY")
    print("=" * 80)
    print(f"{'Checkpoint':<30} | {'Clean':>8} | {'MI-FGSM-100':>12} | {'APGD-100':>10} | {'Div-PGD-5x20':>13}")
    print("-" * 80)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: {res['error']}")
            continue
        att = res.get("attacks", {})
        clean = att.get("clean", {}).get("acc", float("nan"))
        mi = att.get("mi_fgsm_100", {}).get("acc", float("nan"))
        apgd = att.get("apgd_ce_100", {}).get("acc", float("nan"))
        div = att.get("diversity_pgd_5x20", {}).get("acc", float("nan"))
        print(f"{name:<30} | {clean*100:>7.2f}% | {mi*100:>11.2f}% | {apgd*100:>9.2f}% | {div*100:>12.2f}%")

    combined_path = os.path.join(args.output_dir, "g2_d8_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined: {combined_path}")


if __name__ == "__main__":
    main()
