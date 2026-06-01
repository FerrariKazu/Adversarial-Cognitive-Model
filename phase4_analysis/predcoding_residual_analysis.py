#!/usr/bin/env python3
"""
Predictive Coding Residual Analysis
=====================================
Analyzes the prediction error (residual/surprise) magnitudes in RHAN_PredCoding
under PGD-100 attacks at multiple epsilon levels.

Tests the hypothesis: adversarial perturbations that violate natural image
statistics produce larger prediction errors than clean images.

Saves figure to:
  phase4_analysis/figures/combined/predcoding_residuals_vs_epsilon.png
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

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_predcoding import RHAN_PredCoding
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    ckpt = os.path.join(ckpt_dir, 'rhan_predcoding_adv_best.pth')

    if not os.path.exists(ckpt):
        print(f"ERROR: Checkpoint not found at {ckpt}")
        print("Run train_rhan_predcoding_adv.py first!")
        return

    # Load model
    model = RHAN_PredCoding(num_classes=10, embed_dim=512, num_heads=8,
                            ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                            num_recurrent_steps=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    class LogitsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _ = self.m(x)
            return logits

    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
    )

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    # Storage
    mean_residuals_step1 = []
    mean_residuals_step2 = []
    std_residuals_step1 = []
    std_residuals_step2 = []
    accuracies = []

    print(f"\n{'='*70}")
    print(f"Predictive Coding Residual Analysis (PGD-100)")
    print(f"{'='*70}")

    for eps in epsilons:
        eps_start = time.time()
        print(f"Evaluating ε = {eps:.2f}...", end=' ', flush=True)

        all_res1 = []
        all_res2 = []
        correct = 0
        total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0

        for images, labels in testloader:
            if total >= max_samples:
                break
            images = images.to(device)
            labels = labels.to(device)

            if eps > 0:
                adv_images, _ = pgd_attack(
                    LogitsWrapper(model), images, labels,
                    epsilon=eps, alpha=alpha, steps=100,
                    device=device, clip_min=cifar_min, clip_max=cifar_max,
                    random_start=True
                )
            else:
                adv_images = images

            with torch.no_grad():
                logits, residuals = model(adv_images)
                _, preds = logits.max(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                if len(residuals) >= 1:
                    all_res1.append(residuals[0])
                if len(residuals) >= 2:
                    all_res2.append(residuals[1])

        acc = 100.0 * correct / max(total, 1)
        accuracies.append(acc)
        mean_r1 = np.mean(all_res1) if all_res1 else 0
        mean_r2 = np.mean(all_res2) if all_res2 else 0
        std_r1 = np.std(all_res1) if all_res1 else 0
        std_r2 = np.std(all_res2) if all_res2 else 0
        mean_residuals_step1.append(mean_r1)
        mean_residuals_step2.append(mean_r2)
        std_residuals_step1.append(std_r1)
        std_residuals_step2.append(std_r2)

        elapsed = time.time() - eps_start
        print(f"Acc: {acc:.2f}% | Res1: {mean_r1:.2f}±{std_r1:.2f} | Res2: {mean_r2:.2f}±{std_r2:.2f} | {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"PREDICTIVE CODING RESIDUAL ANALYSIS")
    print(f"{'='*70}")
    print(f"{'ε':<8} | {'Acc%':<8} | {'Res Step1':<14} | {'Res Step2':<14}")
    print("-" * 55)
    for i, eps in enumerate(epsilons):
        print(f"{eps:<8.2f} | {accuracies[i]:<7.2f}% | {mean_residuals_step1[i]:<14.3f} | {mean_residuals_step2[i]:<14.3f}")
    print("=" * 55)

    # Check monotonicity of residuals
    mono_s1 = all(x <= y for x, y in zip(mean_residuals_step1, mean_residuals_step1[1:]))
    mono_s2 = all(x <= y for x, y in zip(mean_residuals_step2, mean_residuals_step2[1:]))
    print(f"\nResidual Step 1 monotonic with ε: {'✓ YES' if mono_s1 else '✗ NO'}")
    print(f"Residual Step 2 monotonic with ε: {'✓ YES' if mono_s2 else '✗ NO'}")
    print("Interpretation: Larger residuals at higher ε = adversarial perturbations")
    print("produce more 'surprise' (prediction error) in the predictive coding loop.")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Residual magnitude vs epsilon
    axes[0].errorbar(epsilons, mean_residuals_step1, yerr=std_residuals_step1,
                     fmt='-o', color='#8a2be2', ecolor='#ba55d3', elinewidth=2,
                     capsize=4, linewidth=2.5, markersize=8, label='Step 1')
    axes[0].errorbar(epsilons, mean_residuals_step2, yerr=std_residuals_step2,
                     fmt='--s', color='#ff4500', ecolor='#ff7f50', elinewidth=2,
                     capsize=4, linewidth=2.5, markersize=8, label='Step 2')
    axes[0].set_title("Prediction Error (Surprise) vs. ε", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[0].set_ylabel("Mean Residual Magnitude", fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Right: Accuracy comparison
    axes[1].plot(epsilons, accuracies, '-o', color='#006400', linewidth=2.5,
                 markersize=8, label='RHAN-PredCoding')
    axes[1].set_title("Accuracy vs. ε", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'combined')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'predcoding_residuals_vs_epsilon.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {fig_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
