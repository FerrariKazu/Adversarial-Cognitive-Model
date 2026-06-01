#!/usr/bin/env python3
"""
Split-Stream Ventral/Dorsal RHAN vs. Epsilon Analysis (Trial 3 Evaluation)
========================================================================

Evaluates the trained Ventral/Dorsal Split-Stream RHAN model under PGD-100 
attacks at multiple epsilon levels and compares its performance with RHAN-adv.
Saves diagnostic plot to:
  phase4_analysis/figures/combined/split_stream_accuracy_vs_epsilon.png
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Silence TF32 warning
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model_rhan_split import RHANSplit
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Checkpoint paths
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    split_ckpt = os.path.join(ckpt_dir, 'rhan_split_best.pth')

    if not os.path.exists(split_ckpt):
        print(f"ERROR: Split RHAN checkpoint not found at {split_ckpt}")
        print("Please run train_rhan_split.py first!")
        return

    # 1. Load model
    print("Loading Split-Stream RHAN model...")
    model = RHANSplit(head_type='cosine').to(device)
    model.load_state_dict(torch.load(split_ckpt, map_location=device))
    model.eval()

    # Disable parameters gradients
    for p in model.parameters():
        p.requires_grad = False

    # 2. Setup DataLoaders (500 test images)
    _, testloader_raw = get_dataloaders(
        batch_size=128, num_workers=4, model_name='resnet'
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
    )

    # CIFAR bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    accuracies = []
    print(f"\n{'='*70}")
    print(f"Split-Stream Ventral/Dorsal RHAN Evaluation (PGD-100)")
    print(f"{'='*70}")

    for eps in epsilons:
        eps_start = time.time()
        print(f"Evaluating ε = {eps:.2f}...", end=' ', flush=True)
        total_samples = 0
        correct_samples = 0

        alpha = max(eps / 10, 0.001) if eps > 0 else 0

        for images, labels in testloader:
            if total_samples >= max_samples:
                break

            images = images.to(device)
            labels = labels.to(device)

            if eps > 0:
                adv_images, _ = pgd_attack(
                    model, images, labels,
                    epsilon=eps, alpha=alpha, steps=100,
                    device=device, clip_min=cifar_min, clip_max=cifar_max,
                    random_start=True
                )
            else:
                adv_images = images

            with torch.no_grad():
                logits = model(adv_images)
                _, preds = logits.max(dim=1)
                is_correct = preds.eq(labels)
                correct_samples += is_correct.sum().item()
                total_samples += labels.size(0)

        eps_accuracy = 100. * correct_samples / max(total_samples, 1)
        accuracies.append(eps_accuracy)
        elapsed = time.time() - eps_start
        print(f"Acc: {eps_accuracy:.2f}% | Time: {elapsed:.1f}s")

    # =========================================================================
    # Comparison table vs RHAN-adv
    # =========================================================================
    rhan_adv_pgd = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}

    print(f"\n{'='*70}")
    print(f"TRIAL 3 FINAL VERDICT")
    print(f"{'='*70}")
    print(f"{'ε':<8} | {'RHAN-adv':<12} | {'RHAN-split':<18} | {'Delta':<10}")
    print("-" * 55)
    for i, eps in enumerate(epsilons):
        adv_acc = rhan_adv_pgd[eps]
        split_acc = accuracies[i]
        delta = split_acc - adv_acc
        sign = "+" if delta >= 0 else ""
        print(f"{eps:<8.2f} | {adv_acc:<11.2f}% | {split_acc:<17.2f}% | {sign}{delta:<8.2f}%")
    print("=" * 55)

    # PLOT
    os.makedirs('phase4_analysis/figures/combined', exist_ok=True)
    fig_path = 'phase4_analysis/figures/combined/split_stream_accuracy_vs_epsilon.png'
    
    plt.figure(figsize=(7, 5))
    plt.plot(epsilons, [rhan_adv_pgd[e] for e in epsilons], '-o', label='RHAN-adv (Standard Feedback)', color='#00ff7f', linewidth=2)
    plt.plot(epsilons, accuracies, '-s', label='RHAN-Split (Ventral/Dorsal Pathways)', color='#ff4500', linewidth=2)
    plt.title("Adversarial Robustness: Standard vs. Ventral/Dorsal Split RHAN", fontsize=12, fontweight='bold')
    plt.xlabel("Adversarial Epsilon (ε)", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Diagnostic plot saved to: {fig_path}")

    total_elapsed = time.time() - total_start
    print(f"\nEvaluation complete in {total_elapsed:.1f}s.")


if __name__ == '__main__':
    main()
