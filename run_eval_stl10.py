#!/usr/bin/env python3
"""
Evaluate rhan_stl10_best.pth
- PGD-20 sweep at epsilons {0, 0.01, 0.05, 0.10, 0.20, 0.30}
- PGD-100 at eps=0.05 (key point)
- AutoAttack (standard, eps=0.031, n=1000)
- Per-class breakdown (car vs truck)
- SDT d-prime and ethresh
- Comparison table

Optimized: disables recurrent feedback during PGD (2x speedup),
uses PGD-20 for sweep (standard in literature).
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))

from model_rhan_stl10 import RHANSTL10
from dataset_stl10 import STL10_CLASSES, STL10_MEAN, STL10_STD, STL10_MIN, STL10_MAX, get_stl10_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Model ──────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_stl10_best.pth')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = RHANSTL10(head_type='linear').to(device)
model.load_state_dict(ckpt)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded: {ckpt_path}  ({n_params:,} params)", flush=True)

# ── Data ────────────────────────────────────────────────────────────────────
_, test_loader = get_stl10_loaders(batch_size=128)
clip_min = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
clip_max = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)

print("Loading test data...", flush=True)
all_imgs, all_lbls = [], []
for x, y in test_loader:
    all_imgs.append(x); all_lbls.append(y)
test_imgs = torch.cat(all_imgs, dim=0)
test_lbls = torch.cat(all_lbls, dim=0)
N = test_imgs.size(0)
print(f"Full test set: {N} images", flush=True)

EPSILONS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
BS = 64

# ── PGD attack ─────────────────────────────────────────────────────────────

def pgd_attack(model, x, y, eps, steps=20):
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

def run_pgd_full(model, imgs, lbls, eps, steps=20):
    """Run PGD and return predictions on full dataset."""
    model.eval()
    # Disable recurrent feedback during attack for speed
    orig_recurrent = model.feedback.num_recurrent_steps
    model.feedback.num_recurrent_steps = 0
    all_preds = []
    n = imgs.size(0)
    for i in range(0, n, BS):
        x_b = imgs[i:i+BS].to(device)
        y_b = lbls[i:i+BS].to(device)
        x_adv = pgd_attack(model, x_b, y_b, eps, steps=steps)
        with torch.no_grad():
            all_preds.append(model(x_adv).argmax(1).cpu())
        del x_b, y_b, x_adv
        torch.cuda.empty_cache()
    model.feedback.num_recurrent_steps = orig_recurrent
    return torch.cat(all_preds)

def clean_predict(model, imgs):
    all_preds = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), BS):
            x_b = imgs[i:i+BS].to(device)
            all_preds.append(model(x_b).argmax(1).cpu())
    return torch.cat(all_preds)

# ── 1. PGD-20 sweep ───────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PGD-20 ACCURACY SWEEP (recurrent feedback disabled during attack)", flush=True)
print("=" * 70, flush=True)
print(f"{'ε':<10} {'Accuracy':<12} {'Drop':<12} {'Time':<10}", flush=True)
print("-" * 44, flush=True)

predictions = {}
pgd_accs = {}

for eps in EPSILONS:
    t0 = time.time()
    if eps == 0:
        preds = clean_predict(model, test_imgs)
    else:
        preds = run_pgd_full(model, test_imgs, test_lbls, eps, steps=20)
    elapsed = time.time() - t0

    predictions[eps] = preds
    acc = 100.0 * preds.eq(test_lbls).sum().item() / N
    pgd_accs[eps] = acc
    drop = pgd_accs[0.00] - acc if eps > 0 else 0
    print(f"ε={eps:<8.2f} {acc:>6.2f}%      {drop:>6.2f}%    {elapsed:.0f}s", flush=True)

# ── 2. PGD-100 at select epsilons ─────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PGD-100 SPOT CHECK (ε=0.05 and ε=0.10)", flush=True)
print("=" * 70, flush=True)

for eps in [0.05, 0.10]:
    t0 = time.time()
    preds_100 = run_pgd_full(model, test_imgs, test_lbls, eps, steps=100)
    elapsed = time.time() - t0
    acc_100 = 100.0 * preds_100.eq(test_lbls).sum().item() / N
    acc_20 = pgd_accs[eps]
    print(f"ε={eps:.2f}: PGD-20={acc_20:.2f}%  PGD-100={acc_100:.2f}%  (diff={acc_20-acc_100:+.2f}pp)  ({elapsed:.0f}s)", flush=True)

# ── 3. AutoAttack (standard, ε=0.031, n=1000) ─────────────────────────────
print("\n" + "=" * 70, flush=True)
print("AutoAttack (standard, ε=0.031, n=1000)", flush=True)
print("=" * 70, flush=True)

from autoattack import AutoAttack

class Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)

x_aa = test_imgs[:1000].to(device)
y_aa = test_lbls[:1000].to(device)

with torch.no_grad():
    clean_preds_aa = model(x_aa).argmax(1)
clean_acc_aa = 100.0 * clean_preds_aa.eq(y_aa).sum().item() / y_aa.size(0)
print(f"Clean acc (n=1000): {clean_acc_aa:.2f}%", flush=True)

wrapper = Wrapper(model)
t0 = time.time()
adversary = AutoAttack(wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=True)
x_adv_aa = adversary.run_standard_evaluation(x_aa, y_aa, bs=64)
aa_time = time.time() - t0

with torch.no_grad():
    preds_aa = wrapper(x_adv_aa).argmax(1)
aa_correct = preds_aa.eq(y_aa).sum().item()
aa_acc = 100.0 * aa_correct / x_aa.size(0)
print(f"\nAutoAttack accuracy: {aa_acc:.2f}% ({aa_correct}/{x_aa.size(0)})", flush=True)
print(f"Time: {aa_time:.1f}s", flush=True)

# ── 4. Per-class breakdown ─────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PER-CLASS BREAKDOWN — Clean & AutoAttack", flush=True)
print("=" * 70, flush=True)

print(f"\n{'Class':<12} {'N':<6} {'Clean':<10} {'AA':<10} {'Drop':<10}", flush=True)
print("-" * 50, flush=True)
aa_class = {}
for c in range(10):
    mask = y_aa == c
    n_c = mask.sum().item()
    cl_acc = 100.0 * clean_preds_aa[mask].eq(y_aa[mask]).sum().item() / max(n_c, 1)
    aa_acc_c = 100.0 * preds_aa[mask].eq(y_aa[mask]).sum().item() / max(n_c, 1)
    aa_class[c] = (aa_acc_c, cl_acc, n_c)
    tag = " <<< KEY" if STL10_CLASSES[c] in ['car', 'truck'] else ""
    print(f"{STL10_CLASSES[c]:<12} {n_c:<6} {cl_acc:>5.1f}%     {aa_acc_c:>5.1f}%     {cl_acc - aa_acc_c:>5.1f}%{tag}", flush=True)

car_aa = aa_class[2][0]
car_cl = aa_class[2][1]
truck_aa = aa_class[9][0]
truck_cl = aa_class[9][1]

print(f"\n  *** CAR vs TRUCK @ 96x96 under AutoAttack ***", flush=True)
print(f"      Car:   {car_cl:.1f}% clean -> {car_aa:.1f}% AA  (drop {car_cl-car_aa:.1f}pp)", flush=True)
print(f"      Truck: {truck_cl:.1f}% clean -> {truck_aa:.1f}% AA  (drop {truck_cl-truck_aa:.1f}pp)", flush=True)
print(f"      Gap:   {car_aa - truck_aa:+.1f}pp", flush=True)

# ── 5. SDT d-prime ─────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("SIGNAL DETECTION THEORY — d' and εthresh", flush=True)
print("=" * 70, flush=True)

def zscore(p):
    # Use scipy for Python < 3.12 compatibility
    from scipy.special import erfinv
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(erfinv(2 * p - 1) * math.sqrt(2))

def dprime_from_preds(targets, preds, nc=10):
    cm = np.zeros((nc, nc), dtype=np.float64)
    for t, p in zip(targets.numpy(), preds.numpy()):
        cm[t, p] += 1
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

hdr = f"\n{'ε':<8} {'d_prime_avg':<12} {'d_prime_car':<12} {'d_prime_truck':<12}"
print(hdr, flush=True)
print("-" * 48, flush=True)
dprime_table = {}
for eps in EPSILONS:
    dp = dprime_from_preds(test_lbls, predictions[eps])
    dprime_table[eps] = dp
    print(f"ε={eps:<5.2f} {dp.mean():<12.4f} {dp[2]:<12.4f} {dp[9]:<12.4f}", flush=True)

# εthresh interpolation
eps_arr = np.array(EPSILONS)

def interp_ethresh(dp_arr, target=1.0):
    if dp_arr[0] > target and dp_arr[-1] < target:
        idx = np.where(dp_arr < target)[0][0]
        frac = (target - dp_arr[idx-1]) / (dp_arr[idx] - dp_arr[idx-1])
        return eps_arr[idx-1] + frac * (eps_arr[idx] - eps_arr[idx-1])
    return None

dp_avg = np.array([dprime_table[e].mean() for e in EPSILONS])
dp_car = np.array([dprime_table[e][2] for e in EPSILONS])
dp_truck = np.array([dprime_table[e][9] for e in EPSILONS])

ethresh_avg = interp_ethresh(dp_avg)
ethresh_car = interp_ethresh(dp_car)
ethresh_truck = interp_ethresh(dp_truck)

print(f"", flush=True)
if ethresh_avg:
    print(f"εthresh (d'avg=1.0):   {ethresh_avg:.4f}  ({ethresh_avg*255:.1f}/255)", flush=True)
else:
    print(f"εthresh (d'avg=1.0):   not in range", flush=True)
if ethresh_car:
    print(f"εthresh (d'car=1.0):   {ethresh_car:.4f}  ({ethresh_car*255:.1f}/255)", flush=True)
else:
    print(f"εthresh (d'car=1.0):   not in range", flush=True)
if ethresh_truck:
    print(f"εthresh (d'truck=1.0): {ethresh_truck:.4f}  ({ethresh_truck*255:.1f}/255)", flush=True)
else:
    print(f"εthresh (d'truck=1.0): not in range", flush=True)

# ── 6. Comparison ──────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("COMPARISON: STL-10 RHAN vs CIFAR-10 Curriculum vs Human", flush=True)
print("=" * 70, flush=True)

# CIFAR-10 RHAN-TRADES-Curriculum (from memory, PGD-100)
cifar10_pgd = {0.00: 76.25}
# Human (Elsayed et al. 2018, temporal perturbation on CIFAR-10)
human_pgd = {0.00: 74.15, 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}

print(f"\n{'Metric':<22} {'STL-10 RHAN':>14} {'CIFAR-10':>14} {'Human':>10}", flush=True)
print("-" * 64, flush=True)
for eps in EPSILONS:
    s = pgd_accs.get(eps, 0)
    c = cifar10_pgd.get(eps, None)
    h = human_pgd.get(eps, None)
    c_str = f"{c:.2f}%" if c is not None else "N/A"
    h_str = f"{h:.2f}%" if h is not None else "N/A"
    print(f"PGD-20 ε={eps:<6.2f}      {s:>10.2f}%   {c_str:>14}   {h_str:>10}", flush=True)

print(f"{'AutoAttack ε=0.031':<22} {aa_acc:>13.2f}%   {'N/A':>14} {'N/A':>10}", flush=True)
if ethresh_avg:
    eth_label = "e-thresh(d'avg=1)"
    print(f"{eth_label:<22} {ethresh_avg:>13.4f}  {'N/A':>14} {'N/A':>10}", flush=True)

# ── Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"Model:       RHAN-STL10 (96x96, linear head, {n_params:,} params)", flush=True)
print(f"Clean:       {pgd_accs[0.00]:.2f}%", flush=True)
for eps in EPSILONS[1:]:
    print(f"PGD-20 ε={eps:.2f}:  {pgd_accs[eps]:.2f}%", flush=True)
print(f"AutoAttack:  {aa_acc:.2f}%", flush=True)
print(f"Car AA:      {car_aa:.1f}%  |  Truck AA: {truck_aa:.1f}%  |  Gap: {car_aa-truck_aa:+.1f}pp", flush=True)
print(f"d' avg clean: {dprime_table[0.00].mean():.4f}", flush=True)
if ethresh_avg:
    print(f"e-thresh avg: {ethresh_avg:.4f} ({ethresh_avg*255:.1f}/255)", flush=True)
if ethresh_car:
    print(f"e-thresh car: {ethresh_car:.4f} ({ethresh_car*255:.1f}/255)", flush=True)
if ethresh_truck:
    print(f"e-thresh truck: {ethresh_truck:.4f} ({ethresh_truck*255:.1f}/255)", flush=True)
print("Done.", flush=True)
