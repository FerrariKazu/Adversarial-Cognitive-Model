#!/usr/bin/env python3
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_pretrained import RHANUnifiedSTL10
from train_rhan_stl10_tdv import get_stl10_dataloaders

def pgd_attack_classification(model, x, y, eps=0.031, steps=100, stl_min=None, stl_max=None):
    model.eval()
    if eps == 0.0:
        return x
    
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)
    
    alpha = max(eps / 10, 0.001)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, stl_min, stl_max).detach()
        
    return x_adv

def compute_sdt_for_class(preds, labels, target_class):
    # Treat target_class as signal
    signal_present = (labels == target_class)
    signal_absent = ~signal_present
    said_yes = (preds == target_class)
    said_no = ~said_yes
    
    hits = int(np.sum(signal_present & said_yes))
    misses = int(np.sum(signal_present & said_no))
    false_alarms = int(np.sum(signal_absent & said_yes))
    correct_rejections = int(np.sum(signal_absent & said_no))
    
    # Laplace smoothing
    hr = (hits + 0.5) / (hits + misses + 1)
    far = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    
    dprime = norm.ppf(hr) - norm.ppf(far)
    return float(dprime), hits, misses, false_alarms, correct_rejections, hr, far

def find_threshold_precise(epsilons, values, target_thresh):
    epsilons = np.array(epsilons)
    values = np.array(values)
    
    # Sort
    sort_idx = np.argsort(epsilons)
    epsilons = epsilons[sort_idx]
    values = values[sort_idx]
    
    for i in range(len(values) - 1):
        v1, v2 = values[i], values[i+1]
        e1, e2 = epsilons[i], epsilons[i+1]
        
        if (v1 >= target_thresh and v2 <= target_thresh) or (v1 <= target_thresh and v2 >= target_thresh):
            # linear interpolation
            if abs(v2 - v1) > 1e-6:
                frac = (target_thresh - v1) / (v2 - v1)
                return e1 + frac * (e2 - e1)
            else:
                return e1
    
    if len(values) > 0 and values[0] < target_thresh:
        return epsilons[0]
    return None

def main():
    parser = argparse.ArgumentParser(description='RHAN STL-10 PGD-100 + SDT Evaluation')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/rhan_stl10_tdv_trades_clean_consistency.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--samples', type=int, default=1000, help='Number of test samples to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = RHANUnifiedSTL10().to(device)
    script_dir = os.path.dirname(__file__)
    ckpt_path = os.path.abspath(os.path.join(script_dir, args.checkpoint))
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get dataloaders
    _, testloader, stl_min, stl_max = get_stl10_dataloaders(batch_size=args.batch_size)
    stl_min = stl_min.to(device)
    stl_max = stl_max.to(device)

    # Get subset of test samples
    all_imgs = []
    all_lbls = []
    for imgs, lbls in testloader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
        if sum(x.size(0) for x in all_imgs) >= args.samples:
            break
            
    x_test = torch.cat(all_imgs, dim=0)[:args.samples].to(device)
    y_test = torch.cat(all_lbls, dim=0)[:args.samples].to(device)

    # Define a fine-grained set of epsilons to interpolate thresholds accurately
    epsilons = [0.0, 0.005, 0.01, 0.015, 0.03, 0.05, 0.10, 0.20, 0.30]
    
    print(f"\nRunning PGD-100 evaluation on {x_test.size(0)} test samples...", flush=True)
    print("=" * 110, flush=True)
    print(f"{'Epsilon':<8} | {'Accuracy':<8} | {'Overall d_avg':<13} | {'Car Acc':<8} | {'Car d':<8} | {'Truck Acc':<9} | {'Truck d':<8}", flush=True)
    print("-" * 110, flush=True)

    # Stats arrays for thresholds interpolation
    eps_list = []
    overall_accs = []
    overall_dprimes = []
    car_accs = []
    car_dprimes = []
    truck_accs = []
    truck_dprimes = []

    for eps in epsilons:
        t0 = time.time()
        adv_imgs_list = []
        for i in range(0, x_test.size(0), args.batch_size):
            x_batch = x_test[i:i+args.batch_size]
            y_batch = y_test[i:i+args.batch_size]
            x_adv = pgd_attack_classification(model, x_batch, y_batch, eps=eps, steps=100, stl_min=stl_min, stl_max=stl_max)
            adv_imgs_list.append(x_adv)
        
        x_adv_all = torch.cat(adv_imgs_list, dim=0)
        
        with torch.no_grad():
            with autocast('cuda'):
                logits = model(x_adv_all)
            preds = logits.argmax(dim=1).cpu().numpy()
            
        labels_np = y_test.cpu().numpy()
        
        # Overall acc
        acc = 100.0 * np.sum(preds == labels_np) / len(labels_np)
        
        # Compute SDT d-primes for all classes
        dprimes_all = []
        for c in range(10):
            dp, _, _, _, _, _, _ = compute_sdt_for_class(preds, labels_np, c)
            dprimes_all.append(dp)
        d_avg = np.mean(dprimes_all)

        # Car (c=2) metrics
        car_mask = labels_np == 2
        car_acc = 100.0 * np.sum(preds[car_mask] == 2) / max(np.sum(car_mask), 1)
        car_d, _, _, _, _, _, _ = compute_sdt_for_class(preds, labels_np, 2)

        # Truck (c=9) metrics
        truck_mask = labels_np == 9
        truck_acc = 100.0 * np.sum(preds[truck_mask] == 9) / max(np.sum(truck_mask), 1)
        truck_d, _, _, _, _, _, _ = compute_sdt_for_class(preds, labels_np, 9)

        # Store stats
        eps_list.append(eps)
        overall_accs.append(acc)
        overall_dprimes.append(d_avg)
        car_accs.append(car_acc)
        car_dprimes.append(car_d)
        truck_accs.append(truck_acc)
        truck_dprimes.append(truck_d)

        print(f"{eps:<8.3f} | {acc:<7.2f}% | {d_avg:<13.4f} | {car_acc:<7.1f}% | {car_d:<8.4f} | {truck_acc:<8.1f}% | {truck_d:<8.4f}", flush=True)

    print("=" * 110, flush=True)

    # Compute thresholds
    clean_acc = overall_accs[0]
    acc_50pct_thresh = find_threshold_precise(eps_list, overall_accs, clean_acc / 2.0)
    dprime_1_thresh = find_threshold_precise(eps_list, overall_dprimes, 1.0)
    
    car_d_thresh = find_threshold_precise(eps_list, car_dprimes, 1.0)
    truck_d_thresh = find_threshold_precise(eps_list, truck_dprimes, 1.0)

    print("\nRobustness Threshold Summary:", flush=True)
    print(f"  Clean Accuracy (relative basis): {clean_acc:.2f}%", flush=True)
    print(f"  Accuracy 50% drop threshold (relative εthresh): " + (f"{acc_50pct_thresh:.4f}" if acc_50pct_thresh is not None else f"> {epsilons[-1]}"), flush=True)
    print(f"  SDT d'=1.0 overall threshold: " + (f"{dprime_1_thresh:.4f}" if dprime_1_thresh is not None else f"> {epsilons[-1]}"), flush=True)
    print(f"  Car Class SDT d'=1.0 threshold: " + (f"{car_d_thresh:.4f}" if car_d_thresh is not None else f"> {epsilons[-1]}"), flush=True)
    print(f"  Truck Class SDT d'=1.0 threshold: " + (f"{truck_d_thresh:.4f}" if truck_d_thresh is not None else f"> {epsilons[-1]}"), flush=True)

if __name__ == '__main__':
    main()
