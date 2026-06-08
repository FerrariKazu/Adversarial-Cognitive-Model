#!/usr/bin/env python3
"""
RHAN-v7 Quick Eval: PGD-100 + AutoAttack
Evaluates the v7 checkpoint saved at epoch 13 (best so far).
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_v7 import RHANv7
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

# ── AutoAttack import ──────────────────────────────────
try:
    from autoattack import AutoAttack as AA
    HAS_AA = True
except ImportError:
    HAS_AA = False
    print("WARNING: autoattack not installed. Skipping AutoAttack eval.")
    print("Install with: pip install autoattack")


def eval_pgd100(model, loader, eps_val, device, cifar_min, cifar_max, max_samples=512):
    """PGD-100 evaluation."""
    if eps_val == 0:
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                if total >= max_samples:
                    break
                images, labels = images.to(device), labels.to(device)
                logits, _, _, _ = model(images)
                correct += logits.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return 100. * correct / total if total > 0 else 0.0

    for p in model.parameters():
        p.requires_grad = False

    # Wrap model to return only logits for PGD attack
    class LogitsOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            logits, _, _, _ = self.base(x)
            return logits

    pgd_model = LogitsOnly(model).to(device).eval()

    alpha = max(eps_val / 10, 0.001)
    correct = total = 0
    for images, labels in loader:
        if total >= max_samples:
            break
        images, labels = images.to(device), labels.to(device)
        adv_images, _ = pgd_attack(
            pgd_model, images, labels,
            epsilon=eps_val, alpha=alpha, steps=100,
            device=device, clip_min=cifar_min, clip_max=cifar_max,
        )
        with torch.no_grad():
            logits, _, _, _ = model(adv_images)
            correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total if total > 0 else 0.0


def eval_autoattack(model, device, eps_val, n_samples=1000):
    """AutoAttack evaluation on a subset of test data."""
    if not HAS_AA:
        return None

    _, testloader_raw = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')

    # Collect n_samples
    all_images, all_labels = [], []
    for imgs, lbls in testloader_raw:
        all_images.append(imgs)
        all_labels.append(lbls)
        if sum(x.size(0) for x in all_images) >= n_samples:
            break
    x = torch.cat(all_images, dim=0)[:n_samples].to(device)
    y = torch.cat(all_labels, dim=0)[:n_samples].to(device)

    # Clean accuracy
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(x)
        clean_acc = 100. * logits.argmax(1).eq(y).sum().item() / y.size(0)
    print(f"  Clean acc on {n_samples} samples: {clean_acc:.2f}%")

    # AutoAttack
    # Normalize epsilon to [0,1] range for AutoAttack (it expects unnormalized)
    # Our data is normalized with mean/std, so eps=0.062 in normalized space
    # AutoAttack works in [0,1] space, so we pass eps/255 equivalent
    eps_aa = eps_val  # already in normalized space, AutoAttack handles this

    # Wrap model to return only logits for AutoAttack
    class LogitsWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
        def forward(self, x):
            logits, _, _, _ = self.base(x)
            return logits

    wrapped = LogitsWrapper(model).to(device).eval()

    adversary = AA(wrapped, norm='Linf', eps=eps_aa / 255.0 if eps_aa > 1 else eps_aa,
                   version='standard', device=device)

    print(f"  Running AutoAttack (ε={eps_val:.4f})...")
    t0 = time.time()
    adv_images = adversary.run_standard_evaluation(x, y, bs=256)
    elapsed = time.time() - t0

    with torch.no_grad():
        logits, _, _, _ = model(adv_images)
        aa_acc = 100. * logits.argmax(1).eq(y).sum().item() / y.size(0)
    print(f"  AutoAttack done in {elapsed:.0f}s")
    return aa_acc


def compute_dprime(acc, n_samples):
    """Compute d' from accuracy (assuming 10-class chance = 10%)."""
    from scipy.stats import norm
    if acc <= 0.10:
        return 0.0
    if acc >= 0.999:
        return 5.0
    hit_rate = acc
    fa_rate = (1 - acc) / 9  # distribute errors evenly across 9 wrong classes
    d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    return d_prime


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'rhan_v7_best.pth')

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load model
    model = RHANv7(head_type='cosine').to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # CIFAR bounds (normalized space)
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # Dataloader
    _, testloader_raw = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── PGD-100 Evaluation ──────────────────────────────
    print("\n" + "=" * 60)
    print("PGD-100 EVALUATION")
    print("=" * 60)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    results = []

    for eps_val in epsilons:
        t0 = time.time()
        acc = eval_pgd100(model, testloader, eps_val, device, cifar_min, cifar_max)
        elapsed = time.time() - t0
        d_prime = compute_dprime(acc / 100.0, 512)
        results.append((eps_val, acc, d_prime))
        print(f"  PGD-100 ε={eps_val:.3f}: {acc:.2f}%  d'={d_prime:.4f}  ({elapsed:.0f}s)")

    # Find ε_thresh (d' = 1.0)
    eps_thresh = None
    for i in range(len(results) - 1):
        eps1, acc1, d1 = results[i]
        eps2, acc2, d2 = results[i + 1]
        if d1 >= 1.0 and d2 < 1.0:
            # Linear interpolation
            frac = (1.0 - d1) / (d2 - d1) if d2 != d1 else 0.5
            eps_thresh = eps1 + frac * (eps2 - eps1)
            break

    if eps_thresh is not None:
        print(f"\n  ε_thresh (d'=1.0): {eps_thresh:.4f}")
    else:
        # Check if all d' > 1.0 or all d' < 1.0
        if results and results[-1][2] > 1.0:
            print(f"\n  ε_thresh (d'=1.0): > {epsilons[-1]:.3f} (all d' > 1.0)")
        else:
            print(f"\n  ε_thresh (d'=1.0): < {epsilons[0]:.3f} (all d' < 1.0)")

    # ── AutoAttack Evaluation ───────────────────────────
    if HAS_AA:
        print("\n" + "=" * 60)
        print("AUTOATTACK (standard, ε=0.031, n=1000)")
        print("=" * 60)
        aa_acc = eval_autoattack(model, device, eps_val=0.031, n_samples=1000)
        if aa_acc is not None:
            print(f"\n  AutoAttack robust accuracy: {aa_acc:.2f}%")

    # ── Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Checkpoint: rhan_v7_best.pth (epoch 13, Phase 1)")
    for eps_val, acc, d_prime in results:
        print(f"  ε={eps_val:.3f} → PGD-100: {acc:.2f}%  d'={d_prime:.4f}")
    if eps_thresh is not None:
        print(f"  ε_thresh (d'=1.0): {eps_thresh:.4f}")
    if HAS_AA and aa_acc is not None:
        print(f"  AutoAttack (ε=0.031): {aa_acc:.2f}%")
    print()


if __name__ == '__main__':
    main()
