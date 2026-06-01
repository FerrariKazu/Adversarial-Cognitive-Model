#!/usr/bin/env python3
"""
Dual-Stream Attention Analysis
================================
Analyzes ventral vs dorsal stream contributions under PGD-100 attacks.

Tests hypotheses:
  1. Ventral stream dominates at low epsilon (identity features intact)
  2. Dorsal stream contributes more at high epsilon (spatial info preserved)
  3. Fusion gate shifts toward the more reliable stream as epsilon increases

Saves figure to:
  phase4_analysis/figures/combined/dualstream_analysis.png
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

from phase1_training.model_rhan_dualstream import RHAN_DualStream
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
    ckpt = os.path.join(ckpt_dir, 'rhan_dualstream_adv_best.pth')

    if not os.path.exists(ckpt):
        print(f"ERROR: Checkpoint not found at {ckpt}")
        print("Run train_rhan_dualstream_adv.py first!")
        return

    model = RHAN_DualStream(
        num_classes=10, embed_dim=512, num_heads=8,
        ff_dim=2048, dropout=0.1, num_layers_per_stream=2,
        num_recurrent_steps=2,
    ).to(device)
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

    accuracies = []
    fusion_gates = []

    print(f"\n{'='*70}")
    print(f"Dual-Stream Attention Analysis (PGD-100)")
    print(f"{'='*70}")

    for eps in epsilons:
        eps_start = time.time()
        print(f"Evaluating ε = {eps:.2f}...", end=' ', flush=True)

        correct = 0
        total = 0
        gates = []
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
                logits, aux = model(adv_images)
                _, preds = logits.max(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                gates.append(aux['fusion_gate'].mean().item())

        acc = 100.0 * correct / max(total, 1)
        accuracies.append(acc)
        fusion_gates.append(np.mean(gates))

        elapsed = time.time() - eps_start
        print(f"Acc: {acc:.2f}% | Gate: {fusion_gates[-1]:.3f} | {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"DUAL-STREAM ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"{'ε':<8} | {'Acc%':<8} | {'Fusion Gate':<12} | {'Interpretation'}")
    print("-" * 60)
    for i, eps in enumerate(epsilons):
        gate = fusion_gates[i]
        interp = "Ventral dominant" if gate < 0.4 else ("Balanced" if gate < 0.6 else "Dorsal dominant")
        print(f"{eps:<8.2f} | {accuracies[i]:<7.2f}% | {gate:<12.3f} | {interp}")
    print("=" * 60)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy vs epsilon
    axes[0].plot(epsilons, accuracies, '-o', color='#006400', linewidth=2.5,
                 markersize=8, label='RHAN-DualStream')
    axes[0].set_title("Accuracy vs. ε", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 100)

    # Right: Fusion gate vs epsilon
    axes[1].plot(epsilons, fusion_gates, '-s', color='#8a2be2', linewidth=2.5,
                 markersize=8, label='Fusion Gate (dorsal weight)')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    axes[1].set_title("Fusion Gate vs. ε", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[1].set_ylabel("Mean Fusion Gate Value", fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].fill_between(epsilons, 0.4, 0.6, alpha=0.1, color='gray', label='Balanced zone')

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'combined')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'dualstream_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {fig_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
