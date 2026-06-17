#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_training'))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack
from autoattack import AutoAttack


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

    # Select 128 images for evaluation
    imgs_list, lbls_list = [], []
    total_images = 0
    for imgs, lbls in testloader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        total_images += imgs.size(0)
        if total_images >= 128:
            break
    x_test = torch.cat(imgs_list, dim=0)[:128].to(device)
    y_test = torch.cat(lbls_list, dim=0)[:128].to(device)
    print(f"Loaded {x_test.size(0)} test images.")

    epsilons_255 = [0, 2, 4, 8, 12, 16, 24]
    epsilons = [e / 255.0 for e in epsilons_255]

    epsilons_255 = [0, 2, 4, 8, 12, 16, 24]
    epsilons = [e / 255.0 for e in epsilons_255]

    # Use the precomputed accuracies from the GPU sweep to save time
    pgd_accs = [79.69, 76.56, 75.00, 70.31, 67.97, 65.62, 55.47]
    aa_accs  = [79.69, 24.22, 23.44, 21.88, 21.09, 20.31, 18.75]

    # Human Visual Cognition baseline from paper (at eps = 0.00, 0.05, 0.10)
    # eps=0.00 -> 0/255, eps=0.05 -> 12.75/255, eps=0.10 -> 25.5/255
    human_eps_255 = [0.0, 12.75, 25.5]
    human_accs = [73.33, 69.17, 59.17]

    # ── PLOT GRAPH ──
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 7), dpi=300)

    # Plot curves
    plt.plot(epsilons_255, pgd_accs, marker='o', color='#3B82F6', linestyle='-', linewidth=2.5, label='PGD-100 (Apparent Robustness — Masked)')
    plt.plot(epsilons_255, aa_accs, marker='s', color='#EF4444', linestyle='-', linewidth=2.5, label='AutoAttack (True Robustness — Bypassed)')
    plt.plot(human_eps_255, human_accs, marker='^', color='#10B981', linestyle='-', linewidth=2.5, label='Human Visual Cognition')

    # Fill the gap in red
    plt.fill_between(epsilons_255, aa_accs, pgd_accs, color='#FEE2E2', alpha=0.6, label='Gradient Masking / Artificial Robustness Gap')

    # Formatting
    plt.title("Visualizing Gradient Masking: RHAN-SelfAlign Apparent vs. True Robustness", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Adversarial Perturbation (Epsilon / 255)", fontsize=11, labelpad=8)
    plt.ylabel("Classification Accuracy (%)", fontsize=11, labelpad=8)
    plt.xlim(-0.5, 26.5)
    plt.ylim(-2, 102)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(fontsize=9.5, loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    # Mark the gap at eps=8
    idx_8 = epsilons_255.index(8)
    gap_8 = pgd_accs[idx_8] - aa_accs[idx_8]
    plt.annotate(
        f"Gap = {gap_8:.1f} pp\n(Gradient Masking)",
        xy=(8, (pgd_accs[idx_8] + aa_accs[idx_8]) / 2),
        xytext=(11, (pgd_accs[idx_8] + aa_accs[idx_8]) / 2 - 5),
        arrowprops=dict(facecolor='#DC2626', arrowstyle='->', lw=1.5),
        color='#DC2626', fontweight='bold', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#EF4444', boxstyle='round,pad=0.3')
    )

    plt.tight_layout()
    plt.savefig('figures/gradient_masking_comparison.png', bbox_inches='tight')
    plt.close()
    print("Graph saved to figures/gradient_masking_comparison.png")


if __name__ == '__main__':
    main()
