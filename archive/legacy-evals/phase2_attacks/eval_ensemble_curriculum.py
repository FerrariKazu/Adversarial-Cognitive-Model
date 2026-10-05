#!/usr/bin/env python3
"""
AutoAttack and PGD-100 SDT evaluation script for the Ensemble of Phase B and Phase C
curriculum checkpoints of RHAN-v5 TRADES.
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local directories are in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders

# Helper to automatically install autoattack if not present
try:
    from autoattack import AutoAttack
except ImportError:
    print("autoattack package not found. Installing it automatically...", flush=True)
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "autoattack"], check=True)
        from autoattack import AutoAttack
    except Exception as e:
        print(f"Failed to install/import AutoAttack: {e}")
        print("Please manually run: pip install autoattack")
        sys.exit(1)

class EnsembleAAWrapper(nn.Module):
    def __init__(self, model_b, model_c, return_log=True):
        super().__init__()
        self.model_b = model_b
        self.model_c = model_c
        self.return_log = return_log
        
    def forward(self, x):
        out_b = self.model_b(x)
        out_c = self.model_c(x)
        logits_b = out_b[0] if isinstance(out_b, tuple) else out_b
        logits_c = out_c[0] if isinstance(out_c, tuple) else out_c
        
        prob_b = F.softmax(logits_b, dim=1)
        prob_c = F.softmax(logits_c, dim=1)
        
        avg_prob = (prob_b + prob_c) / 2.0
        if self.return_log:
            # We return log(P) to ensure numerical stability and correct gradient scaling
            # in AutoAttack's loss functions (like CE loss, which performs log_softmax).
            # Note that log_softmax(log(P)) ~ log(P) for normalized P.
            return torch.log(avg_prob + 1e-12)
        return avg_prob

def run_pgd_100_eval(wrapper, loader, epsilons, device, max_samples=500):
    """Run full PGD-100 evaluation on the ensemble wrapper for a list of epsilons."""
    from phase2_attacks.pgd import pgd_attack
    
    # Save the original wrapper return_log state and set to True for gradient-based attack
    original_return_log = wrapper.return_log
    wrapper.return_log = True
    
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)
    
    results = {}
    for eps in epsilons:
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, lbls in loader:
            if total >= max_samples:
                break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(
                    wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True
                )
            else:
                adv_images = images
            with torch.no_grad():
                outputs = wrapper(adv_images)
                _, preds = outputs.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)
        results[eps] = 100.0 * correct / max(total, 1)
        print(f"  PGD-100 ε={eps:.2f} Acc: {results[eps]:.2f}% ({correct}/{total})", flush=True)
        
    wrapper.return_log = original_return_log
    return results

def calculate_sdt_metrics(pgd_results, epsilons):
    """Compute SDT d-prime and interpolate eps_thresh (d'=1.0)."""
    dprimes = []
    for eps in epsilons:
        acc_pct = pgd_results[eps]
        acc = acc_pct / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(dprimes) - 1):
        d1, d2 = dprimes[i], dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(dprimes) > 0 and dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    
    return dprimes, eps_thresh

def main():
    parser = argparse.ArgumentParser(description="AutoAttack ensemble Phase B and Phase C evaluation")
    parser.add_argument('--phase_b_path', type=str, default='checkpoints/rhan_trades_phase_b_final.pth',
                        help='Path to Phase B final checkpoint')
    parser.add_argument('--phase_c_path', type=str, default='checkpoints/rhan_trades_phase_c_final.pth',
                        help='Path to Phase C final checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to evaluate under AutoAttack')
    parser.add_argument('--epsilon', type=float, default=0.031,
                        help='Epsilon attack budget for AutoAttack')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for AutoAttack')
    parser.add_argument('--fast', action='store_true',
                        help='Run fast AutoAttack (APGD-CE only)')
    args = parser.parse_args()
    
    # Memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU memory capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
    print(f"Loading dataset...", flush=True)
    _, testloader = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    
    # Verify file paths
    if not os.path.exists(args.phase_b_path):
        print(f"Error: Phase B checkpoint not found at {args.phase_b_path}")
        sys.exit(1)
    if not os.path.exists(args.phase_c_path):
        print(f"Error: Phase C checkpoint not found at {args.phase_c_path}")
        sys.exit(1)
        
    print(f"Loading Phase B checkpoint from {args.phase_b_path}...", flush=True)
    model_b = RHANv5(head_type='cosine')
    ckpt_b = torch.load(args.phase_b_path, map_location='cpu', weights_only=False)
    state_dict_b = ckpt_b['model_state_dict'] if (isinstance(ckpt_b, dict) and 'model_state_dict' in ckpt_b) else ckpt_b
    model_b.load_state_dict(state_dict_b)
    model_b = model_b.to(device).eval()
    for p in model_b.parameters():
        p.requires_grad = False
        
    print(f"Loading Phase C checkpoint from {args.phase_c_path}...", flush=True)
    model_c = RHANv5(head_type='cosine')
    ckpt_c = torch.load(args.phase_c_path, map_location='cpu', weights_only=False)
    state_dict_c = ckpt_c['model_state_dict'] if (isinstance(ckpt_c, dict) and 'model_state_dict' in ckpt_c) else ckpt_c
    model_c.load_state_dict(state_dict_c)
    model_c = model_c.to(device).eval()
    for p in model_c.parameters():
        p.requires_grad = False
        
    # Create Ensemble wrapper
    # We set return_log=True for AutoAttack gradient stability
    wrapper = EnsembleAAWrapper(model_b, model_c, return_log=True)
    
    # -------------------------------------------------------------
    # 1. Clean accuracy on full test set (10,000 images)
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATING CLEAN ACCURACY ON FULL TEST SET (10,000 images)")
    print("="*80, flush=True)
    
    clean_correct = 0
    clean_total = 0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct_clean = {i: 0 for i in range(10)}
    class_total_clean = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for imgs, lbls in testloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            # Use log-probs or raw probs, argmax is identical
            out = wrapper(imgs)
            preds = out.argmax(dim=1)
            
            clean_correct += (preds == lbls).sum().item()
            clean_total += lbls.size(0)
            
            for i in range(lbls.size(0)):
                lbl = lbls[i].item()
                class_total_clean[lbl] += 1
                if preds[i] == lbl:
                    class_correct_clean[lbl] += 1
                    
    full_clean_acc = 100.0 * clean_correct / clean_total
    print(f"Overall Clean Accuracy (Full Test Set): {full_clean_acc:.2f}% ({clean_correct}/{clean_total})")
    
    # -------------------------------------------------------------
    # 2. Run AutoAttack on 1000 images
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(f"RUNNING AUTOATTACK ON {args.num_samples} IMAGES (eps={args.epsilon:.4f})")
    print("="*80, flush=True)
    
    # Collect exact data samples for AutoAttack
    imgs_list, lbls_list = [], []
    for imgs, lbls in testloader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        if sum(x.size(0) for x in imgs_list) >= args.num_samples:
            break
    x_test = torch.cat(imgs_list, dim=0)[:args.num_samples].to(device)
    y_test = torch.cat(lbls_list, dim=0)[:args.num_samples].to(device)
    
    # Evaluate clean accuracy on this subset first to verify
    with torch.no_grad():
        subset_clean_logits = wrapper(x_test)
        subset_clean_preds = subset_clean_logits.argmax(dim=1)
        subset_clean_correct = (subset_clean_preds == y_test).sum().item()
        subset_clean_acc = 100.0 * subset_clean_correct / args.num_samples
    print(f"Subset Clean Accuracy (1000 images): {subset_clean_acc:.2f}% ({subset_clean_correct}/{args.num_samples})")
    
    # Run AutoAttack
    t0 = time.time()
    adversary = AutoAttack(wrapper, norm='Linf', eps=args.epsilon, version='standard', device=device, verbose=True)
    if args.fast:
        print("Running in FAST mode (APGD-CE only)...", flush=True)
        adversary.attacks_to_run = ['apgd-ce']
    else:
        print("Running in STANDARD mode (APGD-CE + APGD-T + FAB-T + Square)...", flush=True)
        
    try:
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM occurred with batch size {args.batch_size}, retrying with smaller batch size...", flush=True)
            torch.cuda.empty_cache()
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=max(args.batch_size // 2, 8))
        else:
            raise e
            
    elapsed = time.time() - t0
    print(f"\nFinished AutoAttack in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    
    # Calculate robust stats
    with torch.no_grad():
        robust_logits = wrapper(x_adv)
        robust_preds = robust_logits.argmax(dim=1)
        robust_correct = (robust_preds == y_test).sum().item()
        overall_robust_acc = 100.0 * robust_correct / args.num_samples
        
    print(f"Ensemble AutoAttack Robust Accuracy: {overall_robust_acc:.2f}% ({robust_correct}/{args.num_samples})")
    
    # Per-class stats on the 1000 images subset
    class_correct_subset_clean = {i: 0 for i in range(10)}
    class_correct_robust = {i: 0 for i in range(10)}
    class_total_subset = {i: 0 for i in range(10)}
    
    for i in range(args.num_samples):
        lbl = y_test[i].item()
        class_total_subset[lbl] += 1
        if subset_clean_preds[i] == lbl:
            class_correct_subset_clean[lbl] += 1
        if robust_preds[i] == lbl:
            class_correct_robust[lbl] += 1
            
    print("\n" + "-"*65)
    print(f"{'Class':>12} | {'Clean Acc (10k)':>15} | {'Clean Acc (1k)':>15} | {'Robust Acc (1k)':>15} | {'Count':>6}")
    print("-" * 65)
    for i in range(10):
        tot_1k = class_total_subset[i]
        cln_10k_pct = 100.0 * class_correct_clean[i] / max(class_total_clean[i], 1)
        cln_1k_pct = 100.0 * class_correct_subset_clean[i] / max(tot_1k, 1)
        rob_1k_pct = 100.0 * class_correct_robust[i] / max(tot_1k, 1)
        print(f"{class_names[i]:>12} | {cln_10k_pct:>13.1f}% | {cln_1k_pct:>13.1f}% | {rob_1k_pct:>13.1f}% | {tot_1k:>6}")
    print("-" * 65 + "\n")
    
    # -------------------------------------------------------------
    # 3. Decision & Epsilon Threshold calculation if AutoAttack > 30%
    # -------------------------------------------------------------
    print("="*80)
    print("DECISION RULE SUMMARY")
    print("="*80)
    print(f"Ensemble Standard AutoAttack (1000 images): {overall_robust_acc:.2f}%")
    
    if overall_robust_acc > 30.0:
        print("\n>>> SUCCESS: Ensemble AutoAttack robustness is > 30%!")
        print("    -> ACTION: Report BOTH the individual checkpoints AND the ensemble in the paper.")
        
        print("\nRunning PGD-100 Signal Detection Theory (SDT) evaluation to compute ensemble ε_thresh...")
        epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
        pgd_res = run_pgd_100_eval(wrapper, testloader, epsilons, device, max_samples=500)
        dprimes, eps_thresh = calculate_sdt_metrics(pgd_res, epsilons)
        
        print("\n" + "="*80)
        print("Ensemble Robustness Threshold Summary (ε_thresh at d'=1.0):")
        print("="*80)
        print(f"  -> ε_thresh (d'=1.0): {eps_thresh:.4f}" if eps_thresh is not None else "  -> ε_thresh (d'=1.0): >0.30")
        for i, eps in enumerate(epsilons):
            print(f"     ε={eps:.2f} -> PGD-100 Acc: {pgd_res[eps]:.2f}% | d': {dprimes[i]:.4f}")
        print("="*80 + "\n")
    else:
        print("\n>>> NOTE: Ensemble AutoAttack robustness is <= 30.0%.")
        print("    -> Only individual checkpoints should be reported in the main tables unless requested.")
        
        print("\nComputing ensemble ε_thresh anyway for completeness...")
        epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
        pgd_res = run_pgd_100_eval(wrapper, testloader, epsilons, device, max_samples=500)
        dprimes, eps_thresh = calculate_sdt_metrics(pgd_res, epsilons)
        
        print("\n" + "="*80)
        print("Ensemble Robustness Threshold Summary (ε_thresh at d'=1.0):")
        print("="*80)
        print(f"  -> ε_thresh (d'=1.0): {eps_thresh:.4f}" if eps_thresh is not None else "  -> ε_thresh (d'=1.0): >0.30")
        for i, eps in enumerate(epsilons):
            print(f"     ε={eps:.2f} -> PGD-100 Acc: {pgd_res[eps]:.2f}% | d': {dprimes[i]:.4f}")
        print("="*80 + "\n")

    # Clean up memory
    del model_b, model_c, wrapper, x_test, y_test, x_adv
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
