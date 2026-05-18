"""
RHAN Evaluation Script
======================
Evaluates the trained RHAN model against adversarial attacks and
compares performance to ResNet-18 and ViT-Small baselines.

Usage: python3 phase2_attacks/evaluate_rhan.py
"""

import sys
import os
import yaml
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model_rhan import RHAN
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack
from phase2_attacks.fgsm import fgsm_attack


def evaluate_clean(model, testloader, device):
    """Evaluate clean accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def evaluate_attack(model, testloader, device, attack_fn, cifar_min, cifar_max, **attack_kwargs):
    """Evaluate accuracy under adversarial attack."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        adv_images, _ = attack_fn(
            model, images, labels, device=device,
            clip_min=cifar_min, clip_max=cifar_max,
            **attack_kwargs
        )
        adv_images = torch.max(torch.min(adv_images, cifar_max), cifar_min)

        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        del images, labels, adv_images
        torch.cuda.empty_cache()

    return 100.0 * correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']

        # Load model
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'rhan_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Please train RHAN first: python3 phase1_training/train_rhan.py")
        sys.exit(1)

    model = RHAN(num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded RHAN checkpoint from: {ckpt_path}")

    # Load data — native 32×32
    _, testloader = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')

    # CIFAR normalisation bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.5141,  2.6078,  2.7537]).view(1, 3, 1, 1).to(device)

    pgd_steps = config.get('pgd_steps', 20)
    pgd_alpha = config.get('pgd_alpha', 0.01)

    # =========================================================================
    # Clean Accuracy
    # =========================================================================
    clean_acc = evaluate_clean(model, testloader, device)
    print(f"\nClean Accuracy: {clean_acc:.2f}%")

    # =========================================================================
    # PGD Attack Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PGD ACCURACY COLLAPSE — RHAN")
    print("=" * 80)
    print(f"{'Epsilon':<12} {'Accuracy':<12} {'Drop':<12}")
    print("-" * 36)

    pgd_results = {}
    for eps in epsilons:
        if eps == 0.0:
            acc = clean_acc
        else:
            acc = evaluate_attack(
                model, testloader, device, pgd_attack,
                cifar_min, cifar_max,
                epsilon=eps, alpha=pgd_alpha, steps=pgd_steps, random_start=True
            )
        pgd_results[eps] = acc
        drop = clean_acc - acc
        print(f"ε={eps:<10.2f} {acc:>6.2f}%      {drop:>6.2f}%")

    # =========================================================================
    # FGSM Attack Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("FGSM ACCURACY COLLAPSE — RHAN")
    print("=" * 80)
    print(f"{'Epsilon':<12} {'Accuracy':<12} {'Drop':<12}")
    print("-" * 36)

    fgsm_results = {}
    for eps in epsilons:
        if eps == 0.0:
            acc = clean_acc
        else:
            acc = evaluate_attack(
                model, testloader, device, fgsm_attack,
                cifar_min, cifar_max,
                epsilon=eps
            )
        fgsm_results[eps] = acc
        drop = clean_acc - acc
        print(f"ε={eps:<10.2f} {acc:>6.2f}%      {drop:>6.2f}%")

    # =========================================================================
    # Comparison Table
    # =========================================================================
    # Known baselines from our study
    resnet_pgd = {0.00: 95.82, 0.01: 75.57, 0.05: 2.76, 0.10: 0.17, 0.20: 0.00, 0.30: 0.01}
    vit_pgd = {0.00: 97.80, 0.01: 55.18, 0.05: 8.69, 0.10: 2.86, 0.20: 1.09, 0.30: 0.63}
    human = {0.00: 74.15, 0.01: float('nan'), 0.05: 68.54, 0.10: 59.02, 0.20: 63.90, 0.30: 60.00}

    print("\n" + "=" * 80)
    print("COMPARISON: PGD ACCURACY — RHAN vs ResNet-18 vs ViT-Small vs Human")
    print("=" * 80)
    header = f"{'ε':<8}"
    for name in ['RHAN', 'ResNet-18', 'ViT-Small', 'Human']:
        header += f" {name:>12}"
    print(header)
    print("-" * 56)

    for eps in epsilons:
        row = f"{eps:<8.2f}"
        row += f" {pgd_results.get(eps, 0):>11.2f}%"
        row += f" {resnet_pgd.get(eps, 0):>11.2f}%"
        row += f" {vit_pgd.get(eps, 0):>11.2f}%"
        
        hum_val = human.get(eps, 0)
        if hum_val is not None and not np.isnan(hum_val):
            row += f" {hum_val:>11.2f}%"
        else:
            row += f" {'N/A':>12}"
        print(row)

    print("=" * 80)

    # =========================================================================
    # The Moment of Truth
    # =========================================================================
    print("\n" + "=" * 80)
    print("THE MOMENT OF TRUTH")
    print("=" * 80)

    # Check Regime 1: Does RHAN beat ViT at ε=0.01?
    rhan_01 = pgd_results.get(0.01, 0)
    vit_01 = vit_pgd.get(0.01, 0)
    regime1 = "✅ YES" if rhan_01 > vit_01 else "❌ NO"
    print(f"Regime 1 (ε=0.01): RHAN ({rhan_01:.1f}%) > ViT ({vit_01:.1f}%)?  {regime1}")

    # Check Regime 2: Does RHAN beat ResNet at ε=0.10?
    rhan_10 = pgd_results.get(0.10, 0)
    resnet_10 = resnet_pgd.get(0.10, 0)
    regime2 = "✅ YES" if rhan_10 > resnet_10 else "❌ NO"
    print(f"Regime 2 (ε=0.10): RHAN ({rhan_10:.1f}%) > ResNet ({resnet_10:.1f}%)?  {regime2}")

    # Check: Does RHAN beat BOTH at ANY epsilon?
    beats_both = False
    for eps in epsilons:
        rhan_e = pgd_results.get(eps, 0)
        resnet_e = resnet_pgd.get(eps, 0)
        vit_e = vit_pgd.get(eps, 0)
        if rhan_e > resnet_e and rhan_e > vit_e and eps > 0:
            beats_both = True
            print(f"RHAN beats both at ε={eps:.2f}: RHAN={rhan_e:.1f}% vs ResNet={resnet_e:.1f}% vs ViT={vit_e:.1f}%")

    if not beats_both:
        print("RHAN does not beat both baselines simultaneously at any epsilon.")

    print("=" * 80)


if __name__ == '__main__':
    main()
