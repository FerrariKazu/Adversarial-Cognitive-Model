#!/usr/bin/env python3
"""
Evaluate RHAN-STL-10 checkpoint: PGD-100 + AutoAttack + SDT
Compares against CIFAR-10 RHAN-TRADES-Curriculum baseline.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from model_rhan_stl10 import RHANSTL10
from dataset_stl10 import get_stl10_loaders, STL10_MIN, STL10_MAX, STL10_CLASSES

# ── AutoAttack ────────────────────────────────────────────
try:
    from autoattack import AutoAttack as AA
    HAS_AA = True
except ImportError:
    HAS_AA = False
    print("WARNING: autoattack not installed. Skipping AutoAttack.")

CHECKPOINT = "rhan_stl10_best.pth"


def load_model(device):
    model = RHANSTL10(head_type='cosine').to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', CHECKPOINT)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {CHECKPOINT}  ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def pgd100(model, images, labels, epsilon, clip_min, clip_max):
    """PGD-100 attack."""
    step_size = epsilon / 4
    x_adv = images.clone().detach() + 0.001 * torch.randn_like(images)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()
    for _ in range(100):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), labels)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - images, min=-epsilon, max=epsilon)
        x_adv = (images + delta).detach()
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for m in bn_modules:
        m.train()
    return x_adv


def compute_dprime(acc_frac, n_classes=10):
    """Compute d' from accuracy."""
    from scipy.stats import norm
    if acc_frac <= 1.0 / n_classes:
        return 0.0
    if acc_frac >= 0.999:
        return 5.0
    hit = acc_frac
    fa = (1 - acc_frac) / (n_classes - 1)
    return norm.ppf(hit) - norm.ppf(fa)


def find_eps_thresh(epsilons, d_primes, target=1.0):
    """Interpolate ε where d'=target."""
    for i in range(len(d_primes) - 1):
        if d_primes[i] >= target and d_primes[i + 1] < target:
            frac = (target - d_primes[i]) / (d_primes[i + 1] - d_primes[i])
            return epsilons[i] + frac * (epsilons[i + 1] - epsilons[i])
    if d_primes[-1] > target:
        return "> {:.3f}".format(epsilons[-1])
    return "< {:.3f}".format(epsilons[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("RHAN-STL-10 EVALUATION")
    print("=" * 70)

    model = load_model(device)
    clip_min_t = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
    clip_max_t = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)
    _, test_loader = get_stl10_loaders(batch_size=256)

    # ── 1. PGD-100 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PGD-100 EVALUATION")
    print("=" * 70)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    pgd_accs = []
    d_primes = []

    for eps in epsilons:
        if eps == 0.00:
            # Clean accuracy
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits = model(imgs)
                    correct += logits.argmax(1).eq(lbls).sum().item()
                    total += lbls.size(0)
            acc = 100. * correct / total
            t = 0
        else:
            correct = total = 0
            t0 = time.time()
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                x_adv = pgd100(model, imgs, lbls, eps, clip_min_t, clip_max_t)
                with torch.no_grad():
                    logits = model(x_adv)
                    correct += logits.argmax(1).eq(lbls).sum().item()
                total += lbls.size(0)
            acc = 100. * correct / total
            t = time.time() - t0

        d = compute_dprime(acc / 100.0)
        pgd_accs.append(acc)
        d_primes.append(d)
        print("  ε={:.3f}: {:6.2f}%  d'={:.4f}  ({:.0f}s)".format(eps, acc, d, t))

    eps_thresh = find_eps_thresh(epsilons, d_primes)
    print("\n  εthresh (d'=1.0): {}".format(eps_thresh))

    # ── 2. Per-class breakdown ─────────────────────────────
    print("\n" + "=" * 70)
    print("PER-CLASS BREAKDOWN (PGD-100)")
    print("=" * 70)

    class_correct_clean = {i: 0 for i in range(10)}
    class_correct_adv = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        # Clean
        with torch.no_grad():
            preds = model(imgs).argmax(1)
            for i in range(imgs.size(0)):
                l = lbls[i].item()
                class_total[l] += 1
                if preds[i] == l:
                    class_correct_clean[l] += 1

        # Adversarial (PGD-100, ε=0.05)
        x_adv = pgd100(model, imgs, lbls, 0.05, clip_min_t, clip_max_t)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
            for i in range(imgs.size(0)):
                l = lbls[i].item()
                if preds[i] == l:
                    class_correct_adv[l] += 1

    print("\n  {:<12} {:>8} {:>12} {:>8}".format("Class", "Clean", "Adv ε=0.05", "Drop"))
    print("  " + "-" * 44)
    for i, cls in enumerate(STL10_CLASSES):
        if class_total[i] > 0:
            c_clean = 100. * class_correct_clean[i] / class_total[i]
            c_adv = 100. * class_correct_adv[i] / class_total[i]
            drop = c_clean - c_adv
            tag = " ◄ KEY" if cls in ['car', 'truck'] else ""
            print("  {:<12} {:>7.1f}% {:>11.1f}% {:>7.1f}%{}".format(cls, c_clean, c_adv, drop, tag))

    # ── 3. Car vs Truck ────────────────────────────────────
    print("\n" + "=" * 70)
    print("CAR vs TRUCK (key question)")
    print("=" * 70)
    car_i = STL10_CLASSES.index('car')
    truck_i = STL10_CLASSES.index('truck')
    car_clean = 100. * class_correct_clean[car_i] / max(class_total[car_i], 1)
    car_adv = 100. * class_correct_adv[car_i] / max(class_total[car_i], 1)
    truck_clean = 100. * class_correct_clean[truck_i] / max(class_total[truck_i], 1)
    truck_adv = 100. * class_correct_adv[truck_i] / max(class_total[truck_i], 1)
    print("  Car    — Clean: {:.1f}%  Adv ε=0.05: {:.1f}%".format(car_clean, car_adv))
    print("  Truck  — Clean: {:.1f}%  Adv ε=0.05: {:.1f}%".format(truck_clean, truck_adv))
    gap_closed = "YES ✓" if car_adv > 0 and truck_adv > 0 else ("PARTIAL" if max(car_adv, truck_adv) > 0 else "NO ✗")
    print("  Gap closed vs CIFAR-10 (0% both): {}".format(gap_closed))

    # ── 4. AutoAttack ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("AUTOATTACK (standard, ε=0.031, n=1000)")
    print("=" * 70)

    aa_acc = None
    if HAS_AA:
        # Collect 1000 samples
        all_imgs, all_lbls = [], []
        for imgs, lbls in test_loader:
            all_imgs.append(imgs)
            all_lbls.append(lbls)
            if sum(x.size(0) for x in all_imgs) >= 1000:
                break
        x = torch.cat(all_imgs, dim=0)[:1000].to(device)
        y = torch.cat(all_lbls, dim=0)[:1000].to(device)

        with torch.no_grad():
            clean_correct = model(x).argmax(1).eq(y).sum().item()
        print("  Clean acc: {:.2f}%".format(100. * clean_correct / y.size(0)))

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)

        adversary = AA(Wrapper(model).to(device).eval(), norm='Linf',
                       eps=0.031 / 255.0, version='standard', device=device)
        print("  Running AutoAttack...")
        t0 = time.time()
        adv_imgs = adversary.run_standard_evaluation(x, y, bs=256)
        print("  AutoAttack done in {:.0f}s".format(time.time() - t0))

        with torch.no_grad():
            aa_correct = model(adv_imgs).argmax(1).eq(y).sum().item()
        aa_acc = 100. * aa_correct / y.size(0)
        print("  AutoAttack robust acc: {:.2f}%".format(aa_acc))

        # Per-class AA
        print("\n  Per-class AutoAttack:")
        with torch.no_grad():
            aa_preds = model(adv_imgs).argmax(1)
            for i, cls in enumerate(STL10_CLASSES):
                mask = y == i
                if mask.sum() > 0:
                    cls_acc = 100. * aa_preds[mask].eq(i).sum().item() / mask.sum().item()
                    tag = " ◄" if cls in ['car', 'truck'] else ""
                    print("    {:<12} {:>6.1f}%{}".format(cls, cls_acc, tag))
    else:
        print("  SKIPPED (install autoattack: pip install autoattack)")

    # ── 5. SDT Summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("SDT SUMMARY")
    print("=" * 70)
    print("  εthresh (d'=1.0):       {}".format(eps_thresh))
    print("  PGD-10 ε=0.10 acc:     {:.2f}%".format(pgd_accs[3]))
    print("  PGD-10 ε=0.05 acc:     {:.2f}%".format(pgd_accs[2]))
    print("  Clean accuracy:         {:.2f}%".format(pgd_accs[0]))
    if aa_acc is not None:
        print("  AutoAttack ε=0.031:     {:.2f}%".format(aa_acc))

    # ── 6. Comparison ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: STL-10 RHAN vs CIFAR-10 Curriculum vs Human")
    print("=" * 70)

    # CIFAR-10 Curriculum baseline
    cifar = {
        'clean': 78.12, '0.01': 75.00, '0.05': 65.23,
        '0.10': 52.93, '0.20': 29.49, '0.30': 10.16,
        'aa': 21.88, 'ethresh': '0.1850'
    }
    human = {
        'clean': 74.15, '0.01': None, '0.05': 69.17,
        '0.10': 59.17, '0.20': 62.22, '0.30': 58.61,
        'aa': None, 'ethresh': '>0.30'
    }

    stl_aa_str = "{:.2f}%".format(aa_acc) if aa_acc is not None else "N/A"

    print("\n  {:<25} {:>14} {:>14} {:>10}".format("Metric", "STL-10 RHAN", "CIFAR-10 Curr", "Human"))
    print("  " + "-" * 67)
    print("  {:<25} {:>13.2f}% {:>13.2f}% {:>9.1f}%".format("Clean", pgd_accs[0], cifar['clean'], human['clean']))
    print("  {:<25} {:>13.2f}% {:>13.2f}% {:>10}".format("PGD-10 ε=0.01", pgd_accs[1], cifar['0.01'], human['0.01'] or "N/A"))
    print("  {:<25} {:>13.2f}% {:>13.2f}% {:>9.1f}%".format("PGD-10 ε=0.05", pgd_accs[2], cifar['0.05'], human['0.05']))
    print("  {:<25} {:>13.2f}% {:>13.2f}% {:>9.1f}%".format("PGD-10 ε=0.10", pgd_accs[3], cifar['0.10'], human['0.10']))
    print("  {:<25} {:>13.2f}% {:>13.2f}% {:>9.1f}%".format("PGD-10 ε=0.20", pgd_accs[4], cifar['0.20'], human['0.20']))
    print("  {:<25} {:>13.2f}% {:>13.2f}% {:>9.1f}%".format("PGD-10 ε=0.30", pgd_accs[5], cifar['0.30'], human['0.30']))
    print("  {:<25} {:>14} {:>13.2f}% {:>10}".format("AutoAttack ε=0.031", stl_aa_str, cifar['aa'], human['aa'] or "N/A"))
    print("  {:<25} {:>14} {:>14} {:>10}".format("εthresh (d'=1.0)", str(eps_thresh), cifar['ethresh'], human['ethresh']))

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
