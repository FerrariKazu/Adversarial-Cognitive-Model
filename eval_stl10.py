#!/usr/bin/env python3
"""
Comprehensive evaluation of rhan_stl10_best.pth
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))

from model_rhan_stl10 import RHANSTL10
from dataset_stl10 import STL10_CLASSES, STL10_MEAN, STL10_STD, get_stl10_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Model ──────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_stl10_best.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = RHANSTL10(head_type='linear').to(device)
model.load_state_dict(ckpt)
model.eval()
print(f"Loaded: {ckpt_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Data ────────────────────────────────────────────────────────────────────
_, test_loader = get_stl10_loaders(batch_size=64)
clip_min = torch.tensor([-m/s for m, s in zip(STL10_MEAN, STL10_STD)]).view(1, 3, 1, 1).to(device)
clip_max = torch.tensor([(1-m)/s for m, s in zip(STL10_MEAN, STL10_STD)]).view(1, 3, 1, 1).to(device)
EPSILONS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]

# ── Collect all test data ──────────────────────────────────────────────────
print("Loading test data...")
all_imgs, all_lbls = [], []
for x, y in test_loader:
    all_imgs.append(x); all_lbls.append(y)
test_imgs = torch.cat(all_imgs, dim=0).to(device)
test_lbls = torch.cat(all_lbls, dim=0).to(device)
print(f"Full test set: {test_imgs.size(0)} images")

# AA subset (first 1000)
x_aa = test_imgs[:1000]
y_aa = test_lbls[:1000]

# ── PGD-100 ────────────────────────────────────────────────────────────────

def pgd_attack(model, x, y, eps, steps=100):
    if eps == 0:
        return x.clone()
    alpha = eps / 4
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x + torch.clamp(x_adv - x, -eps, eps), clip_min, clip_max).detach()
    return x_adv

BS = 64

def batch_forward(model, imgs, bs=BS):
    all_preds = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), bs):
            x_b = imgs[i:i+bs].to(device, non_blocking=True)
            all_preds.append(model(x_b).argmax(1).cpu())
    return torch.cat(all_preds)

def run_pgd(model, imgs, lbls, eps, steps=100, bs=BS):
    model.eval()
    all_preds = []
    for i in range(0, imgs.size(0), bs):
        x_b = imgs[i:i+bs].to(device, non_blocking=True)
        y_b = lbls[i:i+bs].to(device, non_blocking=True)
        x_adv = pgd_attack(model, x_b, y_b, eps, steps=steps)
        with torch.no_grad():
            all_preds.append(model(x_adv).argmax(1).cpu())
        del x_b, y_b, x_adv
        torch.cuda.empty_cache()
    return torch.cat(all_preds)

print("\n" + "=" * 70)
print("PGD-100 ACCURACY — STL-10 RHAN")
print("=" * 70)
print(f"{'ε':<10} {'Accuracy':<12} {'Drop':<12}")
print("-" * 34)

# Precompute clean predictions for reuse
clean_preds = batch_forward(model, test_imgs)

pgd_results = {}
for eps in EPSILONS:
    t0 = time.time()
    if eps == 0:
        preds = clean_preds
    else:
        preds = run_pgd(model, test_imgs, test_lbls, eps, steps=100)
    acc = 100.0 * preds.eq(test_lbls.cpu()).sum().item() / test_lbls.size(0)
    pgd_results[eps] = acc
    drop = pgd_results[0.00] - acc if eps > 0 else 0
    print(f"ε={eps:<8.2f} {acc:>6.2f}%      {drop:>6.2f}%  ({time.time()-t0:.0f}s)")

# ── AutoAttack ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("AutoAttack (ε=0.031, n=1000)")
print("=" * 70)

from autoattack import AutoAttack

class W(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)

wrapper = W(model)
t0 = time.time()
adversary = AutoAttack(wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=False)
x_adv_aa = adversary.run_standard_evaluation(x_aa, y_aa, bs=64)
aa_time = time.time() - t0

with torch.no_grad():
    preds_aa = wrapper(x_adv_aa).argmax(1)
    aa_correct = preds_aa.eq(y_aa).sum().item()
aa_acc = 100.0 * aa_correct / x_aa.size(0)
print(f"\nAutoAttack accuracy (ε=0.031): {aa_acc:.2f}% ({aa_correct}/{x_aa.size(0)})")
print(f"Time: {aa_time:.1f}s")

# ── Per-class ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PER-CLASS — Clean & AutoAttack (ε=0.031)")
print("=" * 70)

# Clean on same AA subset for fair comparison
clean_preds_aa = batch_forward(model, x_aa)

print(f"\n{'Class':<12} {'Clean Acc':<10} {'AA Acc':<10}")
print("-" * 34)
aa_class_counts = {}
for c in range(10):
    c_mask = y_aa == c
    n_c = c_mask.sum().item()
    clean_acc_c = 100.0 * clean_preds_aa[c_mask].eq(y_aa[c_mask]).sum().item() / max(n_c, 1)
    aa_acc_c = 100.0 * preds_aa[c_mask].eq(y_aa[c_mask]).sum().item() / max(n_c, 1)
    aa_class_counts[c] = (aa_acc_c, preds_aa[c_mask].eq(y_aa[c_mask]).sum().item(), n_c)
    print(f"{STL10_CLASSES[c]:>12s}: {clean_acc_c:>6.2f}%   {aa_acc_c:>6.2f}%")

car_acc_aa, car_c, car_n = aa_class_counts[2]
truck_acc_aa, truck_c, truck_n = aa_class_counts[9]
print(f"\n  KEY: Car vs Truck @ 96×96 under AutoAttack:")
print(f"    Car:   {car_acc_aa:.1f}% ({car_c}/{car_n})")
print(f"    Truck: {truck_acc_aa:.1f}% ({truck_c}/{truck_n})")
print(f"    Gap:   {car_acc_aa - truck_acc_aa:+.1f}pp")

# ── SDT ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SIGNAL DETECTION THEORY")
print("=" * 70)

def zscore(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(math.erfinv(2 * p - 1) * math.sqrt(2))

def confusion_from_preds(targets, preds, nc=10):
    cm = np.zeros((nc, nc), dtype=np.float64)
    for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
        cm[t, p] += 1
    return cm

def dprime_from_cm(cm, nc=10):
    dprimes = []
    for c in range(nc):
        hits = cm[c, c]
        misses = cm[c].sum() - hits
        fas = cm[:, c].sum() - hits
        crs = cm.sum() - hits - misses - fas
        tpr = hits / max(hits + misses, 1)
        fpr = fas / max(fas + crs, 1)
        dprimes.append(zscore(tpr) - zscore(fpr))
    return np.array(dprimes)

print(f"\n{'ε':<8} {'d_prime_avg':<12} {'d_prime_car':<12} {'d_prime_truck':<12}")
print("-" * 48)
dprime_table = {}
for eps in EPSILONS:
    if eps == 0:
        preds_here = clean_preds
    else:
        preds_here = run_pgd(model, test_imgs, test_lbls, eps, steps=100)
    cm = confusion_from_preds(test_lbls, preds_here)
    dp = dprime_from_cm(cm)
    dprime_table[eps] = dp
    print(f"ε={eps:<5.2f} {dp.mean():<12.2f} {dp[2]:<12.2f} {dp[9]:<12.2f}")

# εthresh interpolation for d'=1.0
eps_arr = np.array(EPSILONS)
dp_means = np.array([dprime_table[e].mean() for e in EPSILONS])
if dp_means[0] > 1.0 and dp_means[-1] < 1.0:
    idx = np.where(dp_means < 1.0)[0][0]
    e_l, e_h = eps_arr[idx-1], eps_arr[idx]
    d_l, d_h = dp_means[idx-1], dp_means[idx]
    ethresh = e_l + (1.0 - d_l) * (e_h - e_l) / (d_h - d_l)
    print(f"\nεthresh (d'=1.0): ≈ {ethresh:.4f} ({ethresh*255:.1f}/255)")
else:
    ethresh = None
    print(f"\nεthresh (d'=1.0): not reached in range")

# εthresh for Acc=50%
acc_arr = np.array([pgd_results[e] for e in EPSILONS])
if acc_arr[0] > 50 and acc_arr[-1] < 50:
    idx = np.where(acc_arr < 50)[0][0]
    e_l, e_h = eps_arr[idx-1], eps_arr[idx]
    a_l, a_h = acc_arr[idx-1], acc_arr[idx]
    ethresh_acc = e_l + (50 - a_l) * (e_h - e_l) / (a_h - a_l)
    print(f"εthresh (Acc=50%): ≈ {ethresh_acc:.4f} ({ethresh_acc*255:.1f}/255)")
else:
    ethresh_acc = None
    print(f"εthresh (Acc=50%): not reached in range")

# ── Comparison with CIFAR-10 ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: STL-10 RHAN vs CIFAR-10 RHAN-TRADES vs Human")
print("=" * 70)

# Try to evaluate CIFAR-10 model
cifar_ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_trades_curriculum_best.pth')
cifar10_results = {}
# Human benchmark from Elsayed et al. 2018 (CIFAR-10)
human = {0.00: 94.60, 0.01: 93.12, 0.05: 89.50, 0.10: 85.21, 0.20: 77.61, 0.30: 69.09}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)
cifar_clip_min = torch.tensor([-m/s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]).view(1, 3, 1, 1).to(device)
cifar_clip_max = torch.tensor([(1-m)/s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]).view(1, 3, 1, 1).to(device)

if os.path.exists(cifar_ckpt):
    print(f"\nEvaluating CIFAR-10 RHAN-TRADES-Curriculum...")
    from model_rhan import RHAN
    from dataset import get_dataloaders
    c10_model = RHAN(num_classes=10, head_type='linear').to(device)
    c10_model.load_state_dict(torch.load(cifar_ckpt, map_location=device, weights_only=False))
    c10_model.eval()
    
    _, c10_loader = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')
    c10_imgs, c10_lbls = [], []
    for x, y in c10_loader:
        c10_imgs.append(x); c10_lbls.append(y)
    c10_imgs = torch.cat(c10_imgs, dim=0).to(device)
    c10_lbls = torch.cat(c10_lbls, dim=0).to(device)
    print(f"CIFAR-10 test set: {c10_imgs.size(0)} images")
    
    def pgd_cifar(m, x, y, eps, steps=100):
        if eps == 0: return x.clone()
        a = eps / 4
        xa = x.clone() + torch.empty_like(x).uniform_(-eps, eps)
        xa = torch.clamp(xa, cifar_clip_min, cifar_clip_max).detach()
        for _ in range(steps):
            xa.requires_grad_(True)
            loss = nn.CrossEntropyLoss()(m(xa), y)
            g = torch.autograd.grad(loss, xa)[0]
            xa = xa.detach() + a * g.sign()
            xa = torch.clamp(x + torch.clamp(xa - x, -eps, eps), cifar_clip_min, cifar_clip_max).detach()
        return xa
    
    for eps in EPSILONS:
        t0 = time.time()
        if eps == 0:
            with torch.no_grad():
                preds_c = c10_model(c10_imgs).argmax(1).cpu()
            acc = 100.0 * preds_c.eq(c10_lbls.cpu()).sum().item() / c10_lbls.size(0)
        else:
            all_p = []
            bs = 256
            for i in range(0, c10_imgs.size(0), bs):
                xb, yb = c10_imgs[i:i+bs].to(device), c10_lbls[i:i+bs].to(device)
                xadv = pgd_cifar(c10_model, xb, yb, eps, steps=100)
                with torch.no_grad():
                    all_p.append(c10_model(xadv).argmax(1).cpu())
            preds_c = torch.cat(all_p)
            acc = 100.0 * preds_c.eq(c10_lbls.cpu()).sum().item() / c10_lbls.size(0)
        cifar10_results[eps] = acc
        print(f"  CIFAR-10 PGD-100 ε={eps:.2f}: {acc:.2f}% ({time.time()-t0:.0f}s)")
else:
    print(f"\nNo CIFAR-10 checkpoint at {cifar_ckpt}, using placeholder values")
    cifar10_results = {0.00: 94.38, 0.01: 89.23, 0.05: 62.67, 0.10: 40.57, 0.20: 23.13, 0.30: 15.10}

# Table
print(f"\n{'ε':<8} {'STL10-RHAN':<14} {'CIFAR10-Curr':<14} {'Human':<10}")
print("-" * 48)
for eps in EPSILONS:
    s = pgd_results.get(eps, 0)
    c = cifar10_results.get(eps, 0)
    h = human.get(eps, 0)
    print(f"{eps:<8.2f} {s:>6.2f}%        {c:>6.2f}%        {h:>6.2f}%")

# ── Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Model:                RHAN-STL10 (96×96)")
print(f"Parameters:           {sum(p.numel() for p in model.parameters()):,}")
print(f"Clean accuracy:       {pgd_results[0.00]:.2f}%")
print(f"AutoAttack (ε=0.031): {aa_acc:.2f}%")
print(f"Car AA:               {car_acc_aa:.1f}%")
print(f"Truck AA:             {truck_acc_aa:.1f}%")
print(f"Car-Truck gap (AA):   {car_acc_aa - truck_acc_aa:+.1f}pp")
print(f"d'-avg (clean):       {dprime_table[0.00].mean():.2f}")
if ethresh:
    print(f"εthresh (d'=1.0):      {ethresh:.4f} ({ethresh*255:.1f}/255)")
if ethresh_acc:
    print(f"εthresh (Acc=50%):    {ethresh_acc:.4f} ({ethresh_acc*255:.1f}/255)")
print("Done.")
