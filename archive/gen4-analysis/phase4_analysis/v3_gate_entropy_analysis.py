#!/usr/bin/env python3
"""
RHAN-v3 Gate Activation and Gating Entropy Analysis (Trial 2 Adaptive Recurrence Correlate)
=========================================================================================

Loads the trained RHAN-v3 model, runs PGD-100 attacks at multiple epsilon levels,
and extracts the gating activations during the two recurrent feedback steps.
Computes:
  1. Mean Gate Activation (Step 1 and Step 2)
  2. Effective Recurrent Steps (Sum of mean activations across steps)
  3. Gating Shannon Entropy (spatial distribution of gate maps)

Saves the diagnostic figure to:
  phase4_analysis/figures/combined/v3_gate_entropy_vs_epsilon.png
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

# Optimize matmul performance
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_split import RHANSplit
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_with_gates(model, x):
    """Run forward pass and return logits along with recurrent feedback gates."""
    # Stage 1: Conv Stem
    stem_features = model.stem(x)  # (B, 512, 8, 8)
    
    # Stage 2: Tokeniser
    tokens = model.tokeniser(stem_features)  # (B, 65, 512)
    
    # Stage 3: Split Ventral/Dorsal Attention
    attended = model.transformer(tokens)  # (B, 65, 512)
    
    # Stage 4: Recurrent Feedback
    gates = []
    current = attended
    for t in range(model.feedback.num_recurrent_steps):
        cls_token = current[:, :1, :]
        spatial = model.feedback.tokens_to_spatial(current)
        feedback = model.feedback.feedback_conv(spatial)
        g = model.feedback.gate(feedback)  # (B, 512, 8, 8)
        gates.append(g.detach().cpu())
        
        modulated = stem_features + g * feedback
        modulated_tokens = model.feedback.spatial_to_tokens(modulated, cls_token)
        current = model.transformer(modulated_tokens)
        
    refined = current
    cls_output = refined[:, 0, :]
    logits = model.head(cls_output)
    
    return logits, gates


def compute_shannon_entropy(gate_map):
    """Computes spatial Shannon entropy of the gating map distribution (averaged over channels)."""
    # gate_map is (8, 8)
    p = gate_map.flatten()
    p = p / (p.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
    return entropy


def main():
    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Checkpoint paths
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    v3_ckpt = os.path.join(ckpt_dir, 'rhan_v3_best.pth')

    if not os.path.exists(v3_ckpt):
        print(f"ERROR: RHAN-v3 checkpoint not found at {v3_ckpt}")
        return

    # Load model
    print("Loading RHAN-v3 model...")
    model = RHANSplit(num_classes=10, head_type='cosine').to(device)
    model.load_state_dict(torch.load(v3_ckpt, map_location=device))
    model.eval()

    # Disable parameters gradients
    for p in model.parameters():
        p.requires_grad = False

    # Setup DataLoader — 500 test images
    _, testloader_raw = get_dataloaders(
        batch_size=128, num_workers=4, model_name='resnet'
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
    )

    # CIFAR normalisation bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    # Storage for results
    accuracies = []
    mean_gate_act_step1 = []
    mean_gate_act_step2 = []
    effective_steps = []
    gate_entropy_step1 = []
    gate_entropy_step2 = []

    print(f"\n{'='*80}")
    print(f"RHAN-v3 Gating Activation and Entropy Analysis under PGD-100")
    print(f"{'='*80}")

    for eps in epsilons:
        eps_start = time.time()
        print(f"Evaluating ε = {eps:.2f}...", end=' ', flush=True)
        total_samples = 0
        correct_samples = 0
        
        # Temp accumulators for this epsilon
        eps_gate1_acts = []
        eps_gate2_acts = []
        eps_gate1_entropies = []
        eps_gate2_entropies = []

        alpha = max(eps / 10, 0.001) if eps > 0 else 0

        for images, labels in testloader:
            if total_samples >= max_samples:
                break

            images = images.to(device)
            labels = labels.to(device)

            # Generate adversarial examples using PGD-100
            if eps > 0:
                adv_images, _ = pgd_attack(
                    model, images, labels,
                    epsilon=eps, alpha=alpha, steps=100,
                    device=device, clip_min=cifar_min, clip_max=cifar_max,
                    random_start=True
                )
            else:
                adv_images = images

            # Forward pass with gates extraction
            with torch.no_grad():
                logits, gates = forward_with_gates(model, adv_images)
                _, preds = logits.max(dim=1)
                is_correct = preds.eq(labels)

                correct_samples += is_correct.sum().item()
                total_samples += labels.size(0)

                # gates is a list of [g1, g2], each of shape (B, 512, 8, 8)
                g1, g2 = gates[0], gates[1]

                # 1. Mean Gate Activation: mean over C, H, W for each sample in batch
                g1_mean = g1.mean(dim=(1, 2, 3)).numpy()  # (B,)
                g2_mean = g2.mean(dim=(1, 2, 3)).numpy()  # (B,)
                eps_gate1_acts.extend(g1_mean)
                eps_gate2_acts.extend(g2_mean)

                # 2. Shannon Entropy: compute per-sample
                for i in range(g1.shape[0]):
                    map1 = g1[i].mean(dim=0)  # (8, 8)
                    map2 = g2[i].mean(dim=0)  # (8, 8)
                    eps_gate1_entropies.append(compute_shannon_entropy(map1))
                    eps_gate2_entropies.append(compute_shannon_entropy(map2))

        eps_accuracy = 100. * correct_samples / max(total_samples, 1)
        
        # Calculate stats for this epsilon
        m_g1 = np.mean(eps_gate1_acts)
        m_g2 = np.mean(eps_gate2_acts)
        m_eff = m_g1 + m_g2
        ent_g1 = np.mean(eps_gate1_entropies)
        ent_g2 = np.mean(eps_gate2_entropies)

        accuracies.append(eps_accuracy)
        mean_gate_act_step1.append(m_g1)
        mean_gate_act_step2.append(m_g2)
        effective_steps.append(m_eff)
        gate_entropy_step1.append(ent_g1)
        gate_entropy_step2.append(ent_g2)

        elapsed = time.time() - eps_start
        print(f"Acc: {eps_accuracy:.2f}% | Eff Steps: {m_eff:.3f} (S1:{m_g1:.3f}, S2:{m_g2:.3f}) | Ent1: {ent_g1:.3f} | {elapsed:.1f}s")

    # =========================================================================
    # Print Verdict Table
    # =========================================================================
    print(f"\n{'='*95}")
    print(f"GATING ANALYSIS VERDICT (RHAN-v3)")
    print(f"{'='*95}")
    print(f"{'ε':<8} | {'Accuracy':<10} | {'Mean Gate S1':<13} | {'Mean Gate S2':<13} | {'Eff Steps':<11} | {'Entropy S1':<11} | {'Entropy S2':<11}")
    print("-" * 95)
    for i, eps in enumerate(epsilons):
        print(f"{eps:<8.2f} | {accuracies[i]:<9.2f}% | {mean_gate_act_step1[i]:<13.4f} | {mean_gate_act_step2[i]:<13.4f} | {effective_steps[i]:<11.4f} | {gate_entropy_step1[i]:<11.4f} | {gate_entropy_step2[i]:<11.4f}")
    print("=" * 95)

    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\nGenerating diagnostic plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: Effective steps vs Epsilon
    axes[0].plot(epsilons, effective_steps, '-o', color='#3b82f6', linewidth=2.5, markersize=8, label='Effective Steps (S1+S2)')
    axes[0].plot(epsilons, mean_gate_act_step1, '--s', color='#10b981', linewidth=2, markersize=6, label='Step 1 Gate Mean')
    axes[0].plot(epsilons, mean_gate_act_step2, '--d', color='#f59e0b', linewidth=2, markersize=6, label='Step 2 Gate Mean')
    axes[0].set_title("Effective Recurrent Steps vs. ε", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[0].set_ylabel("Recurrence Activation", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0.0, 2.1)

    # Plot B: Shannon Entropy of Gating Maps vs Epsilon
    axes[1].plot(epsilons, gate_entropy_step1, '-o', color='#8b5cf6', linewidth=2.5, markersize=8, label='Step 1 Shannon Entropy')
    axes[1].plot(epsilons, gate_entropy_step2, '-s', color='#ec4899', linewidth=2.5, markersize=8, label='Step 2 Shannon Entropy')
    axes[1].set_title("Gating Spatial Entropy vs. ε", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[1].set_ylabel("Shannon Entropy (nats)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9)

    # Plot C: Accuracy vs Epsilon
    axes[2].plot(epsilons, accuracies, '-o', color='#ef4444', linewidth=2.5, markersize=8)
    axes[2].set_title("Classification Accuracy vs. ε", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Adversarial Epsilon (ε)", fontsize=10)
    axes[2].set_ylabel("Accuracy (%)", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_ylim(-5, 105)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'combined')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'v3_gate_entropy_vs_epsilon.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved diagnostic plot to: {fig_path}")

    # =========================================================================
    # SUMMARY & INTERPRETATION
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"INTERPRETATION & CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Check monotonicity of effective steps
    is_mono_steps = all(x <= y for x, y in zip(effective_steps, effective_steps[1:]))
    # Check monotonicity of entropy
    is_mono_ent1 = all(x <= y for x, y in zip(gate_entropy_step1, gate_entropy_step1[1:]))
    
    print(f"Monotonic increase of Effective Steps with noise:  {'✓ CONFIRMED' if is_mono_steps else '❌ NOT MONOTONIC'}")
    print(f"Monotonic increase of Gating Spatial Entropy S1:   {'✓ CONFIRMED' if is_mono_ent1 else '❌ NOT MONOTONIC'}")
    print("\nNeuroscientific Implications:")
    print("  - Gating Entropy correlates with the level of representation distortion.")
    print("  - If gates open more at higher epsilons, this shows that the network recruits")
    print("    top-down feedback more intensely when input features are ambiguous or distorted.")
    print("  - This mirrors the increase in reaction times observed in human visual experiments.")

    total_elapsed = time.time() - total_start
    print(f"\nTotal evaluation completed in {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
