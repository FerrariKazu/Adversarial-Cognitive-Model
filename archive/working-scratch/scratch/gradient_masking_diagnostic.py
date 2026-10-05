#!/usr/bin/env python3
"""
Gradient Masking Diagnostic for RHAN-v3
=======================================
1. Loads checkpoints/rhan_v3_best.pth (RHANSplit model).
2. Measures accuracy under Random Noise at ε ∈ [0.01, 0.05, 0.10].
3. Measures accuracy under PGD-20 at ε ∈ [0.01, 0.05, 0.10].
4. Computes PGD-20 vs PGD-100 gap.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def eval_random_noise(model, loader, eps, device, cifar_min, cifar_max, max_samples=500):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            # Generate uniform random noise in [-eps, eps]
            noise = torch.empty_like(images).uniform_(-eps, eps)
            noisy_images = images + noise
            # Clip to valid normalized CIFAR-10 range
            noisy_images = torch.max(torch.min(noisy_images, cifar_max), cifar_min)
            
            outputs = model(noisy_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def eval_pgd(model, loader, eps, alpha, steps, device, cifar_min, cifar_max, max_samples=500):
    correct = 0
    total = 0
    for images, labels in loader:
        if total >= max_samples:
            break
        images, labels = images.to(device), labels.to(device)
        adv_images, _ = pgd_attack(
            model, images, labels,
            epsilon=eps, alpha=alpha, steps=steps,
            device=device, clip_min=cifar_min, clip_max=cifar_max,
            random_start=True
        )
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # Load model
    ckpt_path = "checkpoints/rhan_v3_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
        
    model = RHANSplit(head_type='cosine').to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    epsilons = [0.01, 0.05, 0.10]
    
    print("\nRunning Random Noise diagnostics...")
    random_accs = {}
    for eps in epsilons:
        acc = eval_random_noise(model, testloader, eps, device, cifar_min, cifar_max)
        random_accs[eps] = acc
        print(f"  Random Noise ε={eps:.2f}: {acc:.2f}%")

    print("\nRunning PGD-20 diagnostics...")
    pgd20_accs = {}
    for eps in epsilons:
        alpha = max(eps / 4.0, 0.001)
        acc = eval_pgd(model, testloader, eps, alpha, steps=20, device=device, cifar_min=cifar_min, cifar_max=cifar_max)
        pgd20_accs[eps] = acc
        print(f"  PGD-20 ε={eps:.2f}: {acc:.2f}%")

    pgd100_accs = {0.01: 85.35, 0.05: 60.74, 0.10: 26.17}

    print("\n" + "="*50)
    print("Gradient Masking Diagnostic Report for RHAN-v3")
    print("="*50)
    print(f"{'Attack':<12} | {'ε=0.01':<8} | {'ε=0.05':<8} | {'ε=0.10':<8}")
    print("-"*48)
    print(f"{'Random':<12} | {random_accs[0.01]:>6.2f}% | {random_accs[0.05]:>6.2f}% | {random_accs[0.10]:>6.2f}%")
    print(f"{'PGD-20':<12} | {pgd20_accs[0.01]:>6.2f}% | {pgd20_accs[0.05]:>6.2f}% | {pgd20_accs[0.10]:>6.2f}%")
    print(f"{'PGD-100':<12} | {pgd100_accs[0.01]:>6.2f}% | {pgd100_accs[0.05]:>6.2f}% | {pgd100_accs[0.10]:>6.2f}%")
    print("-"*48)
    gap_01 = pgd20_accs[0.01] - pgd100_accs[0.01]
    gap_05 = pgd20_accs[0.05] - pgd100_accs[0.05]
    gap_10 = pgd20_accs[0.10] - pgd100_accs[0.10]
    print(f"{'Gap (20v100)':<12} | {gap_01:>6.2f}% | {gap_05:>6.2f}% | {gap_10:>6.2f}%")
    print("="*50)

    # Verification checks
    print("\n--- Diagnostic Verdicts ---")
    
    # Test 1: Random vs PGD-100
    for eps in epsilons:
        r_acc = random_accs[eps]
        p_acc = pgd100_accs[eps]
        diff = r_acc - p_acc
        if diff > 10.0:
            print(f"✓ Test 1 (ε={eps:.2f}): PASS (Random Acc {r_acc:.2f}% >> PGD-100 {p_acc:.2f}%, diff={diff:.2f}%)")
        else:
            print(f"✗ Test 1 (ε={eps:.2f}): FAIL (Random Acc {r_acc:.2f}% ≈ PGD-100 {p_acc:.2f}%, diff={diff:.2f}% - potential gradient masking!)")

    # Test 2: PGD-20 vs PGD-100 gap at ε=0.05
    if gap_05 < 8.0:
        print(f"✓ Test 2 (ε=0.05): PASS (PGD-20 to PGD-100 gap is {gap_05:.2f}% < 8%)")
    else:
        print(f"✗ Test 2 (ε=0.05): FAIL (PGD-20 to PGD-100 gap is {gap_05:.2f}% >= 8% - indicates potential gradient masking or optimization lag)")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
