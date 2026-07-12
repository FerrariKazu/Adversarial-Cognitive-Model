#!/usr/bin/env python3
"""
Comparison Evaluation Pipeline
=============================
Evaluates the Standard Large Model and the Epoch 45 Checkpoint across:
1. Clean Test Accuracy.
2. White-box vs. Gray-box AutoAttack (Linf, eps=0.031).
3. PGD-20 sweep across EPSILONS (0.00, 0.01, 0.05, 0.10, 0.20, 0.30) to compute d-prime and epsilon_thresh.
4. Per-class breakdown with a focus on car vs. truck.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

# Add repo paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase1_training')))

from model_rhan_stl10_large import RHANLargeSTL10
from dataset_stl10 import STL10_CLASSES, STL10_MIN, STL10_MAX, get_stl10_loaders
from autoattack import AutoAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom autograd function for Gradient Bypassing (Gray-Box) AutoAttack
class GrayBoxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, model, orig_steps):
        ctx.save_for_backward(x)
        ctx.model = model
        ctx.orig_steps = orig_steps
        
        # Inference/Prediction with full recurrent feedback steps
        model.feedback.num_recurrent_steps = orig_steps
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
        return out.detach()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        model = ctx.model
        orig_steps = ctx.orig_steps
        
        # Calculate gradients using feedforward-only pass (num_recurrent_steps = 0)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            model.feedback.num_recurrent_steps = 0
            out = model(x_in)
            if isinstance(out, tuple):
                out = out[0]
            grad = torch.autograd.grad(out, x_in, grad_outputs=grad_output, retain_graph=False)[0]
            
        model.feedback.num_recurrent_steps = orig_steps
        return grad, None, None

class GrayBoxWrapper(nn.Module):
    def __init__(self, m, orig_steps=2):
        super().__init__()
        self.m = m
        self.orig_steps = orig_steps
    def forward(self, x):
        with autocast('cuda'):
            out = GrayBoxFunction.apply(x, self.m, self.orig_steps)
        return out.float()

class WhiteBoxWrapper(nn.Module):
    def __init__(self, m, orig_steps=2):
        super().__init__()
        self.m = m
        self.orig_steps = orig_steps
    def forward(self, x):
        with autocast('cuda'):
            self.m.feedback.num_recurrent_steps = self.orig_steps
            out = self.m(x)
            if isinstance(out, tuple):
                out = out[0]
        return out.float()

def pgd_attack(model, x, y, eps, steps=20, clip_min=0.0, clip_max=1.0):
    if eps == 0:
        return x.clone()
    alpha = eps / 4
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with autocast('cuda'):
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x + torch.clamp(x_adv - x, -eps, eps), clip_min, clip_max).detach()
    return x_adv

def run_pgd_sweep(model, test_imgs, test_lbls, epsilons, clip_min, clip_max):
    model.eval()
    orig_recurrent = model.feedback.num_recurrent_steps
    results = {}
    
    # We disable recurrent feedback during attack generation for gray-box PGD
    for eps in epsilons:
        print(f"  Running PGD-20 sweep at epsilon={eps:.3f}...", flush=True)
        all_preds = []
        n = test_imgs.size(0)
        bs = 64
        for i in range(0, n, bs):
            x_b = test_imgs[i:i+bs].to(device)
            y_b = test_lbls[i:i+bs].to(device)
            
            model.feedback.num_recurrent_steps = 0
            x_adv = pgd_attack(model, x_b, y_b, eps, steps=20, clip_min=clip_min, clip_max=clip_max)
            
            model.feedback.num_recurrent_steps = orig_recurrent
            with torch.no_grad():
                with autocast('cuda'):
                    pred = model(x_adv)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    all_preds.append(pred.argmax(1).cpu())
            del x_adv
        results[eps] = torch.cat(all_preds)
    return results

def zscore(p):
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

def interp_ethresh(dp_arr, eps_arr, target=1.0):
    if dp_arr[0] > target and dp_arr[-1] < target:
        idx = np.where(dp_arr < target)[0][0]
        frac = (target - dp_arr[idx-1]) / (dp_arr[idx] - dp_arr[idx-1])
        return eps_arr[idx-1] + frac * (eps_arr[idx] - eps_arr[idx-1])
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {args.name}")
    print(f"=======================================================")

    # Load model
    model = RHANLargeSTL10().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load data
    _, test_loader = get_stl10_loaders(batch_size=args.batch_size)
    clip_min = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
    clip_max = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)

    all_imgs, all_lbls = [], []
    for x, y in test_loader:
        all_imgs.append(x)
        all_lbls.append(y)
    test_imgs = torch.cat(all_imgs, dim=0)[:args.samples]
    test_lbls = torch.cat(all_lbls, dim=0)[:args.samples]
    N = test_imgs.size(0)
    print(f"Loaded {N} samples for evaluation.")

    # 1. Clean Evaluation
    with torch.no_grad():
        with autocast('cuda'):
            clean_preds = []
            for i in range(0, N, args.batch_size):
                x_b = test_imgs[i:i+args.batch_size].to(device)
                out = model(x_b)
                if isinstance(out, tuple):
                    out = out[0]
                clean_preds.append(out.argmax(1).cpu())
            clean_preds = torch.cat(clean_preds)
    clean_acc = 100.0 * clean_preds.eq(test_lbls).sum().item() / N
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    # 2. White-box AutoAttack
    print("\nRunning WHITE-BOX AutoAttack (eps=0.031)...", flush=True)
    wb_wrapper = WhiteBoxWrapper(model)
    wb_adversary = AutoAttack(wb_wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=False)
    x_adv_wb = wb_adversary.run_standard_evaluation(test_imgs.to(device), test_lbls.to(device), bs=args.batch_size)
    with torch.no_grad():
        with autocast('cuda'):
            wb_preds = wb_wrapper(x_adv_wb).argmax(1).cpu()
    wb_acc = 100.0 * wb_preds.eq(test_lbls).sum().item() / N
    print(f"White-Box AutoAttack Accuracy: {wb_acc:.2f}%")

    # 3. Gray-box AutoAttack
    print("\nRunning GRAY-BOX AutoAttack (eps=0.031)...", flush=True)
    gb_wrapper = GrayBoxWrapper(model)
    gb_adversary = AutoAttack(gb_wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=False)
    x_adv_gb = gb_adversary.run_standard_evaluation(test_imgs.to(device), test_lbls.to(device), bs=args.batch_size)
    with torch.no_grad():
        with autocast('cuda'):
            gb_preds = gb_wrapper(x_adv_gb).argmax(1).cpu()
    gb_acc = 100.0 * gb_preds.eq(test_lbls).sum().item() / N
    print(f"Gray-Box AutoAttack Accuracy: {gb_acc:.2f}%")

    # Bypass invariance gap
    bypass_gap = gb_acc - wb_acc
    print(f"Bypass Invariance Gap (Gray - White): {bypass_gap:+.2f}pp")

    # 4. Class breakdown for White-box AutoAttack
    print("\nPER-CLASS ACCURACY BREAKDOWN (White-Box AutoAttack)")
    print(f"{'Class':<12} {'N':<6} {'Clean':<10} {'AA (WB)':<10} {'Drop':<10}")
    print("-" * 50)
    class_stats = {}
    for c in range(10):
        mask = test_lbls == c
        n_c = mask.sum().item()
        cl_acc = 100.0 * clean_preds[mask].eq(test_lbls[mask]).sum().item() / max(n_c, 1)
        wb_acc_c = 100.0 * wb_preds[mask].eq(test_lbls[mask]).sum().item() / max(n_c, 1)
        gb_acc_c = 100.0 * gb_preds[mask].eq(test_lbls[mask]).sum().item() / max(n_c, 1)
        class_stats[STL10_CLASSES[c]] = {
            'clean': cl_acc,
            'aa_wb': wb_acc_c,
            'aa_gb': gb_acc_c,
            'n': n_c
        }
        tag = " <<< KEY" if STL10_CLASSES[c] in ['car', 'truck'] else ""
        print(f"{STL10_CLASSES[c]:<12} {n_c:<6} {cl_acc:>5.1f}%     {wb_acc_c:>5.1f}%     {cl_acc - wb_acc_c:>5.1f}%{tag}")

    # 5. PGD-20 Sweep and SDT calculations
    print("\nRunning PGD-20 sweeps for SDT calculation...", flush=True)
    EPSILONS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    sweep_results = run_pgd_sweep(model, test_imgs, test_lbls, EPSILONS, clip_min, clip_max)
    
    dprime_table = {}
    print(f"\n{'ε':<8} {'d_prime_avg':<12} {'d_prime_car':<12} {'d_prime_truck':<12}")
    print("-" * 48)
    for eps in EPSILONS:
        dp = dprime_from_preds(test_lbls, sweep_results[eps])
        dprime_table[eps] = dp
        print(f"ε={eps:<5.2f} {dp.mean():<12.4f} {dp[2]:<12.4f} {dp[9]:<12.4f}")

    # Interpolate epsilon_thresh (d' = 1.0)
    eps_arr = np.array(EPSILONS)
    dp_avg = np.array([dprime_table[e].mean() for e in EPSILONS])
    dp_car = np.array([dprime_table[e][2] for e in EPSILONS])
    dp_truck = np.array([dprime_table[e][9] for e in EPSILONS])

    ethresh_avg = interp_ethresh(dp_avg, eps_arr)
    ethresh_car = interp_ethresh(dp_car, eps_arr)
    ethresh_truck = interp_ethresh(dp_truck, eps_arr)

    # Save results as a structured numpy file for report/heatmap script
    out_dir = "competition/output"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.name}_stats.npz")
    print(f"Saving stats to absolute path: {os.path.abspath(out_file)}", flush=True)
    
    np.savez(out_file,
             clean_acc=clean_acc,
             wb_acc=wb_acc,
             gb_acc=gb_acc,
             bypass_gap=bypass_gap,
             class_stats=class_stats,
             epsilons=eps_arr,
             dp_avg=dp_avg,
             dp_car=dp_car,
             dp_truck=dp_truck,
             ethresh_avg=ethresh_avg if ethresh_avg else -1.0,
             ethresh_car=ethresh_car if ethresh_car else -1.0,
             ethresh_truck=ethresh_truck if ethresh_truck else -1.0)
    
    print(f"Files in output dir: {os.listdir(out_dir)}", flush=True)
    print("\nEvaluation complete. Stats saved.", flush=True)

if __name__ == "__main__":
    main()
