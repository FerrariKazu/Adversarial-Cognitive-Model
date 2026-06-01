#!/usr/bin/env python3
"""
Neural Representation Alignment Analysis (Improvement #3)
===========================================================
Evaluates alignment quality between RHAN and CORnet-S IT features
under clean and adversarial conditions.

Tests: Does alignment survive at high epsilon? Does forced alignment
with biological features improve robustness?

Saves figure to:
  phase4_analysis/figures/combined/alignment_analysis.png
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan import RHAN
from phase1_training.alignment_head import AlignmentHead
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
    
    # Support both rhan_aligned_best.pth (Trial 8) and rhan_alignment_adv_best.pth (fallback)
    rhan_aligned_ckpt = os.path.join(ckpt_dir, 'rhan_aligned_best.pth')
    rhan_alignment_adv_ckpt = os.path.join(ckpt_dir, 'rhan_alignment_adv_best.pth')

    if os.path.exists(rhan_aligned_ckpt):
        adv_ckpt = rhan_aligned_ckpt
        use_rhan_aligned = True
        print(f"Found Trial 8 (RHANAligned) checkpoint at: {adv_ckpt}")
    elif os.path.exists(rhan_alignment_adv_ckpt):
        adv_ckpt = rhan_alignment_adv_ckpt
        use_rhan_aligned = False
        print(f"Found fallback (RHAN + AlignmentHead) checkpoint at: {adv_ckpt}")
    else:
        print(f"ERROR: Alignment checkpoint not found. Tried paths:")
        print(f"  - {rhan_aligned_ckpt}")
        print(f"  - {rhan_alignment_adv_ckpt}")
        print("Please run train_rhan_aligned.py first!")
        return

    # Load CORnet-S teacher model dynamically for on-the-fly features
    print("Loading pre-trained CORnet-S teacher...")
    from phase1_training.model_cornets import CIFARCORnet
    from phase1_training.train_rhan_aligned import get_it_features

    teacher = CIFARCORnet().to(device)
    cornet_ckpt = os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'cornets_best.pth')
    if not os.path.exists(cornet_ckpt):
        # Alternative path
        cornet_ckpt = os.path.join(ckpt_dir, 'cornets_best.pth')
    
    if os.path.exists(cornet_ckpt):
        teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device))
        print(f"Loaded CORnet-S teacher from {cornet_ckpt}")
    else:
        print(f"ERROR: CORnet-S teacher checkpoint not found at {cornet_ckpt}")
        return
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load model and alignment head based on path
    if use_rhan_aligned:
        from phase1_training.model_rhan_aligned import RHANAligned
        model = RHANAligned(head_type='cosine').to(device)
        ckpt = torch.load(adv_ckpt, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        alignment_head = None
    else:
        model = RHAN(num_classes=10, embed_dim=512, num_heads=8,
                     ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                     num_recurrent_steps=2, head_type='linear').to(device)
        alignment_head = AlignmentHead(rhan_dim=512, bio_dim=512, hidden_dim=256).to(device)
        ckpt = torch.load(adv_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        alignment_head.load_state_dict(ckpt['alignment_head'])
        model.eval()
        alignment_head.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in alignment_head.parameters():
            p.requires_grad = False

    class LogitsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

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
    alignments = []

    print(f"\n{'='*70}")
    print(f"Alignment Quality Analysis (PGD-100)")
    print(f"{'='*70}")

    for eps in epsilons:
        eps_start = time.time()
        print(f"Evaluating ε = {eps:.2f}...", end=' ', flush=True)

        correct = 0
        total = 0
        cos_sims = []
        alpha = max(eps / 10, 0.001) if eps > 0 else 0

        for step, (images, labels) in enumerate(testloader):
            if total >= max_samples:
                break
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

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
                logits = model(adv_images)
                _, preds = logits.max(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                # Compute biological activations on the fly from original clean images (resized to 224x224)
                imgs_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                bio_batch = get_it_features(teacher, imgs_resized)
                bio_norm = F.normalize(bio_batch, dim=-1)

                # Compute model projected features
                if use_rhan_aligned:
                    _, rhan_proj = model.forward_with_features(adv_images)
                else:
                    cls_features = model.get_feature_vector(adv_images)
                    rhan_proj = alignment_head(cls_features)

                cos_sim = (rhan_proj * bio_norm).sum(dim=-1)
                cos_sims.extend(cos_sim.cpu().numpy())

        acc = 100.0 * correct / max(total, 1)
        mean_align = np.mean(cos_sims) if cos_sims else 0
        accuracies.append(acc)
        alignments.append(mean_align)

        elapsed = time.time() - eps_start
        print(f"Acc: {acc:.2f}% | Align: {mean_align:.4f} | {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"ALIGNMENT ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"{'ε':<8} | {'Acc%':<8} | {'CosSim':<10} | {'Alignment'}")
    print("-" * 50)
    for i, eps in enumerate(epsilons):
        interp = "Strong" if alignments[i] > 0.5 else ("Moderate" if alignments[i] > 0.3 else "Weak")
        print(f"{eps:<8.2f} | {accuracies[i]:<7.2f}% | {alignments[i]:<10.4f} | {interp}")
    print("=" * 50)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epsilons, alignments, '-o', color='#8a2be2', linewidth=2.5,
                 markersize=8, label='RHAN ↔ CORnet IT')
    axes[0].set_title("Representation Alignment vs. ε", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[0].set_ylabel("Mean Cosine Similarity", fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 1)

    axes[1].plot(epsilons, accuracies, '-o', color='#006400', linewidth=2.5,
                 markersize=8, label='RHAN-Alignment')
    axes[1].set_title("Accuracy vs. ε", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'combined')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'alignment_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {fig_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
