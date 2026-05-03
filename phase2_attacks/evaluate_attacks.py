"""
Adversarial Attack Evaluation Script
=====================================

PURPOSE:
    This script loads the Phase 1 checkpoint and systematically benchmarks all
    three attack methods (FGSM, PGD, C&W) across all epsilon levels defined in
    config/attack_config.yaml.

    It produces a comprehensive results table showing how model accuracy degrades
    as a function of perturbation strength — the core empirical result of Phase 2.

WHAT THE RESULTS TABLE TELLS US:
    - "Accuracy" column: How many images the model still classifies correctly
      after the attack. 95.82% → 0% means the attack has completely fooled
      the model at that epsilon.
    - "Avg L2": The average Euclidean distance between clean and adversarial
      images across all pixels. This measures "how much did the image change
      overall?" — directly comparable to psychophysical JND thresholds.
    - "Avg L∞": The maximum pixel change. For FGSM/PGD this equals epsilon
      (by construction). For C&W it varies per image.

COGNITIVE SCIENCE CONTEXT:
    The accuracy-vs-epsilon curve is the machine analog of a psychometric
    function in psychophysics. In human experiments, you plot "% correct
    identification" vs "stimulus degradation level" and fit a sigmoid.
    Our Phase 3 human study will produce the same curve, and Phase 5 (SDT)
    will formally compare the two.
"""

import sys
import os
import yaml
import torch
import numpy as np

# Add the project root and phase1 to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model import CIFARResNet
from phase1_training.dataset import get_dataloaders
from fgsm import fgsm_attack
from pgd import pgd_attack
from cw import cw_attack


def compute_distortions(clean, adversarial):
    """
    Compute L2 and L∞ distortion between clean and adversarial images.

    1. WHAT: Measures how much the adversarial image differs from the original.
    2. WHY: Distortion metrics tell us whether the attack is "imperceptible."
       - L2: Total energy of the perturbation. If L2 is small, the image looks
         almost identical to a human.
       - L∞: Worst-case pixel change. If L∞ is small, no single pixel stands out.
    3. OBSERVE: FGSM/PGD will have L∞ ≈ ε exactly. C&W will have varying L2.
    """
    delta = (adversarial - clean).view(clean.size(0), -1)
    l2 = torch.norm(delta, p=2, dim=1).mean().item()
    linf = torch.norm(delta, p=float('inf'), dim=1).mean().item()
    return l2, linf


def evaluate_attack(model, testloader, attack_fn, device, **attack_kwargs):
    """Run an attack on the full test set and return accuracy + distortions."""
    correct = 0
    total = 0
    all_l2 = []
    all_linf = []

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        adv_images, adv_preds = attack_fn(
            model, images, labels, device=device, **attack_kwargs
        )
        correct += (adv_preds == labels).sum().item()
        total += labels.size(0)

        l2, linf = compute_distortions(images, adv_images)
        all_l2.append(l2)
        all_linf.append(linf)

    accuracy = 100.0 * correct / total
    avg_l2 = np.mean(all_l2)
    avg_linf = np.mean(all_linf)
    return accuracy, avg_l2, avg_linf


def main():
    # Load config
    with open('../config/attack_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Load the trained model
    # -------------------------------------------------------------------------
    model = CIFARResNet().to(device)
    checkpoint_path = os.path.join('..', config['checkpoint_path'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}\n")

    # Get test data (no augmentation, just normalization)
    _, testloader = get_dataloaders(batch_size=64, num_workers=4)

    epsilons = config['epsilons']
    pgd_steps = config['pgd_steps']
    pgd_alpha = config['pgd_alpha']

    # -------------------------------------------------------------------------
    # Results Table Header
    # -------------------------------------------------------------------------
    print(f"{'Attack':<10} | {'Epsilon':>8} | {'Accuracy':>10} | {'Avg L2':>10} | {'Avg Linf':>10}")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # FGSM Evaluation
    # -------------------------------------------------------------------------
    for eps in epsilons:
        acc, l2, linf = evaluate_attack(
            model, testloader, fgsm_attack, device, epsilon=eps
        )
        print(f"{'FGSM':<10} | {eps:>8.2f} | {acc:>9.2f}% | {l2:>10.4f} | {linf:>10.4f}")

    print("-" * 60)

    # -------------------------------------------------------------------------
    # PGD Evaluation
    # -------------------------------------------------------------------------
    for eps in epsilons:
        acc, l2, linf = evaluate_attack(
            model, testloader, pgd_attack, device,
            epsilon=eps, alpha=pgd_alpha, steps=pgd_steps
        )
        print(f"{'PGD':<10} | {eps:>8.2f} | {acc:>9.2f}% | {l2:>10.4f} | {linf:>10.4f}")

    print("-" * 60)

    # -------------------------------------------------------------------------
    # C&W Evaluation (L2 attack — epsilon is not directly used)
    # -------------------------------------------------------------------------
    # NOTE: C&W minimizes L2 distortion, not L∞. We run it once (no epsilon
    # sweep) because the attack finds the minimum perturbation automatically.
    # We include it for comparison against the L∞ attacks at various epsilons.
    # -------------------------------------------------------------------------
    acc, l2, linf = evaluate_attack(
        model, testloader, cw_attack, device
    )
    print(f"{'C&W-L2':<10} | {'  auto':>8} | {acc:>9.2f}% | {l2:>10.4f} | {linf:>10.4f}")

    print("-" * 60)
    print("\nDone. Compare accuracy degradation across attacks and epsilons.")
    print("Key insight: PGD should be strictly stronger than FGSM at every epsilon.")
    print("C&W finds the minimum-distortion adversarial — closest to human JND.")


if __name__ == '__main__':
    main()
