#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_training'))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


def compute_dprime(acc_pct):
    acc = acc_pct / 100.0
    hr = np.clip(acc, 1e-5, 1.0 - 1e-5)
    far = np.clip((1.0 - acc) / 9.0, 1e-5, 1.0 - 1e-5)
    return float(stats.norm.ppf(hr) - stats.norm.ppf(far))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'rhan_selfalign_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    model = RHANv5().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out
    wrapper = Wrapper(model)

    # Load test data
    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500
    selfalign_accs = []

    print("\nRunning PGD-100 evaluation on RHAN-SelfAlign...")
    for eps in epsilons:
        t0 = time.time()
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0.0
        
        for imgs, lbls in testloader:
            if total >= max_samples:
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)

            if eps > 0:
                x_adv, _ = pgd_attack(
                    wrapper, imgs, lbls, epsilon=eps, alpha=alpha, steps=100,
                    device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True
                )
            else:
                x_adv = imgs

            with torch.no_grad():
                preds = wrapper(x_adv).argmax(1)
                correct += (preds == lbls).sum().item()
                total += B

        acc = 100. * correct / total
        selfalign_accs.append(acc)
        print(f"  ε={eps:.2f} -> PGD-100 Accuracy: {acc:.2f}% (Time: {time.time()-t0:.1f}s)")

    # Compute d-primes
    selfalign_dprimes = [compute_dprime(a) for a in selfalign_accs]

    # Calculate eps_thresh (d' = 1.0)
    eps_thresh = None
    for i in range(len(selfalign_dprimes) - 1):
        d1, d2 = selfalign_dprimes[i], selfalign_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(selfalign_dprimes) > 0 and selfalign_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]

    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"
    print(f"\nCalculated eps_thresh (d'=1.0) for RHAN-SelfAlign: {thresh_str}")

    # Ensure output directories exist
    os.makedirs('figures', exist_ok=True)

    # ── PLOT 1: ACCURACY DECAY COMPARISON ──
    plt.figure(figsize=(10, 7), dpi=300)
    
    acc_data = {
        'Human (Visual Cognition)': ([73.33, np.nan, 69.17, 59.17, 62.22, 58.61], '#10B981', 'o', '-'),
        'RHAN-trades-curriculum': ([78.12, 75.00, 65.23, 52.93, 29.49, 10.16], '#D97706', 'D', '-'),
        'RHAN-v5-TRADES (Baseline)': ([87.30, 84.77, 65.82, 37.89, 5.47, 0.20], '#EC4899', '^', '--'),
        'RHAN-SelfAlign (PGD-100, masked)': (selfalign_accs, '#8B5CF6', 's', '-'),
        'ResNet-18 (Feedforward)': ([95.82, 75.57, 2.84, 0.21, 0.02, 0.00], '#EF4444', 'x', '-.'),
        'ViT-Small (Transformer)': ([97.80, 55.18, 8.80, 2.78, 1.12, 0.58], '#6B7280', '*', '-.')
    }

    # Plot lines
    for name, (y_vals, color, marker, ls) in acc_data.items():
        y_vals_arr = np.array(y_vals)
        mask = ~np.isnan(y_vals_arr)
        plt.plot(np.array(epsilons)[mask], y_vals_arr[mask], marker=marker, label=name, color=color, linestyle=ls, linewidth=2.5, markersize=8)

    # Plot single AutoAttack point for SelfAlign to show the gradient masking drop
    plt.scatter([0.031], [21.60], color='#EF4444', marker='X', s=150, zorder=5, label='RHAN-SelfAlign under AutoAttack (eps=0.031)')
    plt.annotate(
        "AutoAttack drop (21.6%)\ndue to gradient masking",
        xy=(0.031, 21.60),
        xytext=(0.06, 15.0),
        arrowprops=dict(facecolor='#EF4444', arrowstyle='->', lw=1.5),
        color='#EF4444', fontweight='bold', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#EF4444', boxstyle='round,pad=0.3')
    )

    plt.title("Adversarial Robustness: PGD-100 Accuracy Decay Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Adversarial Perturbation Budget (Epsilon, L_inf)", fontsize=11, labelpad=8)
    plt.ylabel("Classification Accuracy (%)", fontsize=11, labelpad=8)
    plt.xlim(-0.01, 0.31)
    plt.ylim(-2, 102)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig('figures/selfalign_accuracy_decay.png', bbox_inches='tight')
    plt.close()

    # ── PLOT 2: SENSITIVITY DECAY (d') COMPARISON ──
    plt.figure(figsize=(10, 7), dpi=300)
    plt.axhline(y=1.0, color='#374151', linestyle='--', alpha=0.7, linewidth=1.5, label="Near-chance Threshold (d'=1.0)")
    plt.fill_between([-0.02, 0.32], -3.5, 1.0, color='#F3F4F6', alpha=0.5, label='Chance-Performance Region')

    dprime_data = {
        'Human (Visual Cognition)': ([4.790, 4.567, 3.985, 3.368, 2.440, 1.769], '#10B981', 'o', '-'),
        'RHAN-trades-curriculum': ([2.748, 2.589, 2.159, 1.696, 0.877, 0.010], '#D97706', 'D', '-'),
        'RHAN-v5-TRADES (Baseline)': ([3.383, 3.186, 2.230, 1.231, -0.291, -1.602], '#EC4899', '^', '--'),
        'RHAN-SelfAlign (PGD-100, masked)': (selfalign_dprimes, '#8B5CF6', 's', '-'),
        'ResNet-18 (Feedforward)': ([4.426, 2.687, -0.771, -1.707, -1.913, -1.880], '#EF4444', 'x', '-.'),
        'ViT-Small (Transformer)': ([4.931, 1.814, -0.154, -0.909, -1.242, -1.469], '#6B7280', '*', '-.')
    }

    for name, (y_vals, color, marker, ls) in dprime_data.items():
        plt.plot(epsilons, y_vals, marker=marker, label=name, color=color, linestyle=ls, linewidth=2.5, markersize=8)

    # Plot single AutoAttack point for SelfAlign dprime
    aa_dprime = compute_dprime(21.60)
    plt.scatter([0.031], [aa_dprime], color='#EF4444', marker='X', s=150, zorder=5, label="Self-Align d' under AutoAttack")

    plt.title("Signal Detection Theory (SDT): Perceptual Sensitivity Decay Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Adversarial Perturbation Budget (Epsilon, L_inf)", fontsize=11, labelpad=8)
    plt.ylabel("Sensitivity Index (d')", fontsize=11, labelpad=8)
    plt.xlim(-0.01, 0.31)
    plt.ylim(-3.5, 5.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9, loc='lower left', frameon=True, facecolor='white', edgecolor='none')
    
    # Annotate Self-aligned PGD threshold vs True threshold
    if eps_thresh is not None:
        plt.axvline(x=eps_thresh, color='#8B5CF6', linestyle=':', alpha=0.8, linewidth=1.5)
        plt.text(eps_thresh + 0.003, -2.8, f"SelfAlign (PGD-100)\nε_thresh ≈ {eps_thresh:.3f}", color='#8B5CF6', fontweight='bold', fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig('figures/selfalign_sensitivity_decay.png', bbox_inches='tight')
    plt.close()

    print("Comparison figures saved successfully:")
    print("  - figures/selfalign_accuracy_decay.png")
    print("  - figures/selfalign_sensitivity_decay.png")


if __name__ == '__main__':
    main()
