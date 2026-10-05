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

        # Free GPU memory explicitly to prevent fragmentation/OOM on long runs
        del images, labels, adv_images, adv_preds
        torch.cuda.empty_cache()

    accuracy = 100.0 * correct / total
    avg_l2 = np.mean(all_l2)
    avg_linf = np.mean(all_linf)
    return accuracy, avg_l2, avg_linf


import argparse
from phase1_training.model_vit import CIFARViT
from phase1_training.dataset_vit import get_dataloaders_vit

import os

MODELS = {
    'resnet': {
        'ckpt': os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'best.pth'),
        'class': CIFARResNet,
        'loader_fn': get_dataloaders
    },
    'vit': {
        'ckpt': os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth'),
        'class': CIFARViT,
        'loader_fn': get_dataloaders_vit
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'vit'], required=True)
    args = parser.parse_args()

    cfg = MODELS[args.model]

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = cfg['class']().to(device)
    model.load_state_dict(torch.load(cfg['ckpt'], map_location=device))
    model.eval()
    print(f"Loaded {args.model} checkpoint from: {cfg['ckpt']}\n")

    _, testloader = cfg['loader_fn'](batch_size=32, num_workers=2)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    pgd_steps = config.get('pgd_steps', 10)
    pgd_alpha = config.get('pgd_alpha', 0.01)

    print(f"{'Attack':<10} | {'Epsilon':>8} | {'Accuracy':>10} | {'Avg L2':>10} | {'Avg Linf':>10}")
    print("-" * 60)

    for eps in epsilons:
        acc, l2, linf = evaluate_attack(model, testloader, fgsm_attack, device, epsilon=eps)
        print(f"{'FGSM':<10} | {eps:>8.2f} | {acc:>9.2f}% | {l2:>10.4f} | {linf:>10.4f}")

    print("-" * 60)

    for eps in epsilons:
        acc, l2, linf = evaluate_attack(model, testloader, pgd_attack, device, epsilon=eps, alpha=pgd_alpha, steps=pgd_steps)
        print(f"{'PGD':<10} | {eps:>8.2f} | {acc:>9.2f}% | {l2:>10.4f} | {linf:>10.4f}")

    print("-" * 60)

    if args.model != 'vit':
        acc, l2, linf = evaluate_attack(model, testloader, cw_attack, device)
        print(f"{'C&W-L2':<10} | {'  auto':>8} | {acc:>9.2f}% | {l2:>10.4f} | {linf:>10.4f}")
        print("-" * 60)

if __name__ == '__main__':
    main()
