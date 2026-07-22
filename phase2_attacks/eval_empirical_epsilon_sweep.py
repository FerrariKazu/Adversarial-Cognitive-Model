#!/usr/bin/env python3
"""
Empirical Epsilon Sweep & SDT Sensitivity Evaluation for STL-10
===============================================================
Evaluates 4 key checkpoints across an 7-point epsilon grid (n=500 per point):
  1. Static TRADES Large Baseline (static_trades_large)
  2. RHAN-v10 Large ep45 (reproducibility anchor for 0.0553)
  3. RHAN-v10 Final Epoch 60 (rhan_v10_final)
  4. RHAN-v11 Best Epoch 60 (rhan_v11_best)

Epsilon Grid: [0.000, 0.031, 0.062, 0.094, 0.150, 0.200, 0.300]

Outputs full empirical d'(epsilon) table, per-class Hit/FA rates, clean accuracy,
and derived linear d'=1.0 crossing thresholds.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scipy.stats import norm
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model architectures
from phase1_training.model_rhan_stl10_large import RHANLargeSTL10
from phase1_training.model_rhan_v11 import RHANv11

MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
STD = torch.tensor([0.2603, 0.2565, 0.2713]).view(1, 3, 1, 1)


def acc_to_dprime_pooled(acc_pct, K=10):
    p_hit = np.clip(acc_pct / 100.0, 1e-4, 1.0 - 1e-4)
    p_fa = np.clip((1.0 - p_hit) / (K - 1), 1e-4, 1.0 - 1e-4)
    return float(norm.ppf(p_hit) - norm.ppf(p_fa))


def compute_per_class_sdt(y_true, y_pred, num_classes=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    dprime_per_class = []
    hit_rates = []
    fa_rates = []
    
    for c in range(num_classes):
        pos_mask = (y_true == c)
        neg_mask = (y_true != c)
        
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)
        
        if n_pos == 0 or n_neg == 0:
            dprime_per_class.append(0.0)
            hit_rates.append(0.0)
            fa_rates.append(0.0)
            continue
            
        hits = np.sum((y_pred == c) & pos_mask)
        fas = np.sum((y_pred == c) & neg_mask)
        
        # Apply 1/(2N) logit correction for boundary rates (0% or 100%)
        hit_rate = (hits + 0.5) / (n_pos + 1.0)
        fa_rate = (fas + 0.5) / (n_neg + 1.0)
        
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        d_val = float(z_hit - z_fa)
        
        dprime_per_class.append(d_val)
        hit_rates.append(float(hit_rate))
        fa_rates.append(float(fa_rate))
        
    macro_dprime = float(np.mean(dprime_per_class))
    return macro_dprime, hit_rates, fa_rates, dprime_per_class


def run_pgd_attack(model, images, labels, eps, steps=50, alpha=None, model_type="static"):
    if eps == 0.0:
        return images.clone()
        
    if alpha is None:
        alpha = (2.5 * eps) / steps
        
    device = images.device
    mean = MEAN.to(device)
    std = STD.to(device)
    
    stl_min = ((0.0 - mean) / std)
    stl_max = ((1.0 - mean) / std)
    
    # Initialize random perturbation
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    # Convert eps to normalized space approx
    eps_norm = eps / std.mean().item()
    alpha_norm = alpha / std.mean().item()
    
    x_adv = images.clone() + delta
    x_adv = torch.max(torch.min(x_adv, stl_max), stl_min).detach()
    
    for _ in range(steps):
        x_adv.requires_grad_()
        if model_type == "rhan_v11":
            logits, _, _, _ = model(x_adv, steps=4)
        else:
            logits = model(x_adv)
            
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        x_adv = x_adv.detach() + alpha_norm * grad.sign()
        
        # Project perturbation into l_infinity ball in normalized domain
        diff = x_adv - images
        diff = torch.clamp(diff, -eps_norm, eps_norm)
        x_adv = torch.max(torch.min(images + diff, stl_max), stl_min).detach()
        
    return x_adv


def eval_model_at_epsilon(model, dataloader, eps, device, model_type="static", pgd_steps=50):
    model.eval()
    all_targets = []
    all_preds = []
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        if eps > 0.0:
            x_eval = run_pgd_attack(model, images, targets, eps=eps, steps=pgd_steps, model_type=model_type)
        else:
            x_eval = images
            
        with torch.no_grad():
            if model_type == "rhan_v11":
                logits, _, _, _ = model(x_eval, steps=4)
            else:
                logits = model(x_eval)
                
            preds = logits.argmax(dim=-1)
            
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
    acc = float(np.mean(np.array(all_targets) == np.array(all_preds)) * 100.0)
    macro_dprime, hit_rates, fa_rates, dprime_per_class = compute_per_class_sdt(all_targets, all_preds)
    pooled_dprime = acc_to_dprime_pooled(acc)
    
    return {
        "accuracy": round(acc, 2),
        "macro_dprime": round(macro_dprime, 4),
        "pooled_dprime": round(pooled_dprime, 4),
        "hit_rates": [round(h, 4) for h in hit_rates],
        "fa_rates": [round(f, 4) for f in fa_rates],
        "dprime_per_class": [round(d, 4) for d in dprime_per_class]
    }


def find_dprime_crossing(epsilons, dprimes, target_d=1.0):
    eps_arr = np.array(epsilons)
    d_arr = np.array(dprimes)
    
    # Sort by descending d'
    sort_idx = np.argsort(-d_arr)
    eps_sorted = eps_arr[sort_idx]
    d_sorted = d_arr[sort_idx]
    
    if np.max(d_sorted) < target_d:
        return 0.0
    if np.min(d_sorted) > target_d:
        return float(np.max(eps_sorted))
        
    try:
        f_interp = interp1d(d_sorted, eps_sorted, kind='linear')
        return float(f_interp(target_d))
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Empirical Epsilon Sweep & SDT Threshold Evaluation for STL-10")
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to STL-10 dataset')
    parser.add_argument('--n-samples', type=int, default=500, help='Sample size per epsilon point (default: 500)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--pgd-steps', type=int, default=50, help='PGD attack steps (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset subsetting')
    parser.add_argument('--output-json', type=str, default='report/empirical_sweep_results_stl10.json', help='Output JSON path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("======================================================================")
    print("  Empirical Epsilon Sweep & SDT Evaluation Session")
    print(f"  Device: {device} | Samples: n={args.n_samples} | PGD Steps: {args.pgd_steps}")
    print("======================================================================")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare STL-10 Test Dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2565, 0.2713])
    ])

    full_testset = datasets.STL10(root=args.data_dir, split='test', download=True, transform=transform_test)
    indices = torch.randperm(len(full_testset))[:args.n_samples].tolist()
    subset_dataset = Subset(full_testset, indices)
    test_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Checkpoint paths
    ckpt_map = {
        "static_trades_large": ("checkpoints/rhan_stl10_tdv_trades_actual.pth", "static"),
        "rhan_stl10_large_ep45": ("checkpoints/rhan_stl10_large_pseudolabel_best.pth", "static"),
        "rhan_v10_final": ("checkpoints/rhan_stl10_v10_best.pth", "rhan_v10"),
        "rhan_v11_best": ("checkpoints/rhan_stl10_v11_best.pth", "rhan_v11"),
        "rhan_v11_rolling": ("checkpoints/rhan_stl10_v11_rolling.pth", "rhan_v11"),
    }

    eps_grid = [0.000, 0.0313, 0.0625, 0.0940, 0.1500, 0.2000, 0.3000]
    sweep_results = {}

    for model_name, (ckpt_path, model_type) in ckpt_map.items():
        print(f"\n[-->] Evaluating Checkpoint: '{model_name}' ({ckpt_path})...")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️ Warning: Checkpoint file '{ckpt_path}' not found! Skipping.", flush=True)
            continue

        # Load Architecture
        if model_type in ["static"]:
            model = RHANLargeSTL10().to(device)
        elif model_type in ["rhan_v10"]:
            model = RHANLargeSTL10().to(device)
        else: # rhan_v11
            model = RHANv11().to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        is_rhan = (model_type in ["rhan_v10", "rhan_v11"])
        ckpt_metrics = {"epsilons": eps_grid, "accuracy": [], "macro_dprime": [], "pooled_dprime": [], "per_eps_details": {}}

        t0 = time.time()
        for eps in eps_grid:
            print(f"  Evaluating eps={eps:.4f}...", end="", flush=True)
            t_eps = time.time()
            res = eval_model_at_epsilon(model, test_loader, eps=eps, device=device, model_type=model_type, pgd_steps=args.pgd_steps)
            dt = time.time() - t_eps

            ckpt_metrics["accuracy"].append(res["accuracy"])
            ckpt_metrics["macro_dprime"].append(res["macro_dprime"])
            ckpt_metrics["pooled_dprime"].append(res["pooled_dprime"])
            ckpt_metrics["per_eps_details"][str(eps)] = res

            print(f" Acc: {res['accuracy']:>5.2f}% | Macro d': {res['macro_dprime']:>6.4f} | Pooled d': {res['pooled_dprime']:>6.4f} ({dt:.1f}s)", flush=True)

        elapsed = time.time() - t0
        # Compute exact linear crossing threshold at d'=1.0
        thresh_macro = find_dprime_crossing(eps_grid, ckpt_metrics["macro_dprime"], target_d=1.0)
        thresh_pooled = find_dprime_crossing(eps_grid, ckpt_metrics["pooled_dprime"], target_d=1.0)

        ckpt_metrics["thresh_dprime_1_macro"] = round(thresh_macro, 4)
        ckpt_metrics["thresh_dprime_1_pooled"] = round(thresh_pooled, 4)
        ckpt_metrics["total_eval_time_sec"] = round(elapsed, 1)

        print(f"  ✓ Model '{model_name}' Sweep Complete in {elapsed/60.0:.1f} min.")
        print(f"    --> Threshold (Macro d'=1.0): eps = {thresh_macro:.4f}")
        print(f"    --> Threshold (Pooled d'=1.0): eps = {thresh_pooled:.4f}")

        sweep_results[model_name] = ckpt_metrics

    # Save empirical JSON output
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(sweep_results, f, indent=2)

    # Print Full Comparative Table
    print("\n==========================================================================================")
    print("  FULL EMPIRICAL SDT SENSITIVITY TABLE (STL-10, n=500)")
    print("==========================================================================================")
    header = f"{'Epsilon':<8} | " + " | ".join([f"{name[:18]:<18}" for name in sweep_results.keys()])
    print(header)
    print("-" * len(header))

    for idx, eps in enumerate(eps_grid):
        row = f"{eps:<8.4f} | "
        for model_name in sweep_results.keys():
            d_val = sweep_results[model_name]["macro_dprime"][idx]
            acc_val = sweep_results[model_name]["accuracy"][idx]
            row += f"{d_val:>6.3f} ({acc_val:>4.1f}%)    | "
        print(row)

    print("-" * len(header))
    row_thresh = f"{'eps_thresh':<8} | "
    for model_name in sweep_results.keys():
        t_val = sweep_results[model_name]["thresh_dprime_1_macro"]
        row_thresh += f"{t_val:>18.4f} | "
    print(row_thresh)
    print("==========================================================================================")
    print(f"Empirical results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
