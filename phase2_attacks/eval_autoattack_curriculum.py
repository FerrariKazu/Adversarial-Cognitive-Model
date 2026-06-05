#!/usr/bin/env python3
"""
AutoAttack comparison script for RHAN-v5 TRADES Phase B vs Phase C Final checkpoints.
Evaluates standard AutoAttack on 1000 test images (Linf, eps=0.031) and prints
a detailed per-class breakdown and deployment decision comparison.
"""

import os
import sys
import time
import argparse
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

class AAWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, tuple) else out

def run_evaluation(checkpoint_path, testloader, num_samples, eps, device, batch_size, fast_mode=False):
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    print(f"Loading weights...", flush=True)
    
    model = RHANv5(head_type='cosine')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Check if checkpoint is dict with state_dict or raw state dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
        
    wrapper = AAWrapper(model)
    
    # Collect data samples
    imgs_list, lbls_list = [], []
    for imgs, lbls in testloader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        if sum(x.size(0) for x in imgs_list) >= num_samples:
            break
    x_test = torch.cat(imgs_list, dim=0)[:num_samples].to(device)
    y_test = torch.cat(lbls_list, dim=0)[:num_samples].to(device)
    
    # Clean accuracy
    print(f"Evaluating clean accuracy on {num_samples} samples...", flush=True)
    with torch.no_grad():
        clean_logits = wrapper(x_test)
        clean_preds = clean_logits.argmax(1)
        clean_correct = (clean_preds == y_test).sum().item()
        clean_acc = 100.0 * clean_correct / num_samples
    print(f"Clean Accuracy: {clean_acc:.2f}% ({clean_correct}/{num_samples})")
    
    # AutoAttack configuration
    t0 = time.time()
    adversary = AutoAttack(wrapper, norm='Linf', eps=eps, version='standard', device=device, verbose=True)
    
    if fast_mode:
        print(f"Running FAST AutoAttack (APGD-CE only) at eps={eps:.4f}...", flush=True)
        adversary.attacks_to_run = ['apgd-ce']
    else:
        print(f"Running STANDARD AutoAttack (APGD-CE + APGD-T + FAB-T + Square) at eps={eps:.4f}...", flush=True)
        
    # Run attack in chunks to avoid OOM
    try:
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM occurred with batch size {batch_size}, retrying with smaller batch size...", flush=True)
            torch.cuda.empty_cache()
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=max(batch_size // 2, 8))
        else:
            raise e
                
    elapsed = time.time() - t0
    print(f"Finished AutoAttack in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    
    # Robust accuracy
    with torch.no_grad():
        robust_logits = wrapper(x_adv)
        robust_preds = robust_logits.argmax(1)
        robust_correct = (robust_preds == y_test).sum().item()
        robust_acc = 100.0 * robust_correct / num_samples
        
    print(f"Robust Accuracy: {robust_acc:.2f}% ({robust_correct}/{num_samples})")
    
    # Per-class stats
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct_clean = {i: 0 for i in range(10)}
    class_correct_robust = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    for i in range(num_samples):
        lbl = y_test[i].item()
        class_total[lbl] += 1
        if clean_preds[i] == lbl:
            class_correct_clean[lbl] += 1
        if robust_preds[i] == lbl:
            class_correct_robust[lbl] += 1
            
    per_class_results = {}
    print("\nPer-class Breakdown:")
    print(f"{'Class':>12} | {'Clean Acc':>10} | {'Robust Acc':>10} | {'Count':>6}")
    print("-" * 47)
    for i in range(10):
        tot = class_total[i]
        cln_a = 100.0 * class_correct_clean[i] / max(tot, 1)
        rob_a = 100.0 * class_correct_robust[i] / max(tot, 1)
        per_class_results[class_names[i]] = {
            'clean': cln_a,
            'robust': rob_a,
            'count': tot
        }
        print(f"{class_names[i]:>12} | {cln_a:>9.1f}% | {rob_a:>9.1f}% | {tot:>6}")
        
    # Free GPU memory
    del model, wrapper, x_test, y_test, x_adv
    torch.cuda.empty_cache()
    
    return clean_acc, robust_acc, per_class_results

def main():
    parser = argparse.ArgumentParser(description="AutoAttack curriculum checkpoint comparison")
    parser.add_argument('--phase_b_path', type=str, default='checkpoints/rhan_trades_phase_b_final.pth',
                        help='Path to Phase B final checkpoint')
    parser.add_argument('--phase_c_path', type=str, default='checkpoints/rhan_trades_phase_c_final.pth',
                        help='Path to Phase C final checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to evaluate')
    parser.add_argument('--epsilon', type=float, default=0.031,
                        help='Epsilon attack budget')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for AutoAttack')
    parser.add_argument('--fast', action='store_true',
                        help='Run fast mode (APGD-CE only)')
    parser.add_argument('--phase', type=str, choices=['b', 'c', 'both'], default='both',
                        help='Which phase checkpoint to evaluate')
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
    # Get loader
    _, testloader = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    
    results = {}
    
    # Determine checkpoints to run
    run_b = args.phase in ['b', 'both']
    run_c = args.phase in ['c', 'both']
    
    # Verify file paths
    if run_b and not os.path.exists(args.phase_b_path):
        print(f"Warning: Phase B checkpoint not found at {args.phase_b_path}. Skipping Phase B.")
        run_b = False
    if run_c and not os.path.exists(args.phase_c_path):
        print(f"Warning: Phase C checkpoint not found at {args.phase_c_path}. Skipping Phase C.")
        run_c = False
        
    if not run_b and not run_c:
        print("Error: No valid checkpoints to evaluate. Exiting.")
        sys.exit(1)
        
    if run_b:
        print("\n" + "="*80)
        print("RUNNING AUTOATTACK ON PHASE B FINAL")
        print("="*80)
        cln_b, rob_b, cls_b = run_evaluation(
            args.phase_b_path, testloader, args.num_samples, args.epsilon, device, args.batch_size, args.fast
        )
        results['Phase B'] = {'clean': cln_b, 'robust': rob_b, 'classes': cls_b}
        
    if run_c:
        print("\n" + "="*80)
        print("RUNNING AUTOATTACK ON PHASE C FINAL")
        print("="*80)
        cln_c, rob_c, cls_c = run_evaluation(
            args.phase_c_path, testloader, args.num_samples, args.epsilon, device, args.batch_size, args.fast
        )
        results['Phase C'] = {'clean': cln_c, 'robust': rob_c, 'classes': cls_c}
        
    # Print comparison report
    if 'Phase B' in results and 'Phase C' in results:
        print("\n" + "="*80)
        print("PHASE B VS PHASE C AUTOATTACK COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Metric':<25} | {'Phase B Final':>15} | {'Phase C Final':>15} | {'Difference':>12}")
        print("-" * 75)
        print(f"{'Clean Accuracy':<25} | {results['Phase B']['clean']:>14.2f}% | {results['Phase C']['clean']:>14.2f}% | {results['Phase B']['clean'] - results['Phase C']['clean']:>11.2f}%")
        print(f"{'Robust Accuracy (ε=0.031)':<25} | {results['Phase B']['robust']:>14.2f}% | {results['Phase C']['robust']:>14.2f}% | {results['Phase B']['robust'] - results['Phase C']['robust']:>11.2f}%")
        print("-" * 75)
        
        # Per-class comparison
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print("\nPer-class Robust Accuracy Comparison:")
        print(f"{'Class':>12} | {'Phase B Robust':>15} | {'Phase C Robust':>15} | {'Difference':>12}")
        print("-" * 62)
        for name in class_names:
            rb = results['Phase B']['classes'][name]['robust']
            rc = results['Phase C']['classes'][name]['robust']
            print(f"{name:>12} | {rb:>14.1f}% | {rc:>14.1f}% | {rb - rc:>11.1f}%")
            
        print("\n" + "="*80)
        print("DEPLOYMENT RECOMMENDATION AND CRITERIA DECISION")
        print("="*80)
        rob_diff = results['Phase B']['robust'] - results['Phase C']['robust']
        if rob_diff > 0:
            print(f"Phase B AutoAttack robustness ({results['Phase B']['robust']:.2f}%) exceeds Phase C ({results['Phase C']['robust']:.2f}%).")
            print(">>> RECOMMENDATION: PHASE B is the deployment checkpoint.")
            print("    Rationale: Phase B provides a better overall clean/robust trade-off boundary,")
            print("               avoiding the representational drift of excessive ε=0.150 regularization.")
            print("               Phase C remains the research headline (highest εthresh=0.150 curriculum target).")
        else:
            print(f"Phase B AutoAttack robustness ({results['Phase B']['robust']:.2f}%) is less than or equal to Phase C ({results['Phase C']['robust']:.2f}%).")
            print(">>> RECOMMENDATION: PHASE C is the deployment checkpoint (and research headline).")
            print("    Rationale: Phase C maintains higher or equal robust sensitivity under standard attack,")
            print("               proving that the curriculum effectively scaled the boundary margins.")
        print("="*80 + "\n")
    elif 'Phase B' in results:
        print("\nNote: Only Phase B Final was evaluated. Compare with Phase C results manually to make the decision recommendation.")
    elif 'Phase C' in results:
        print("\nNote: Only Phase C Final was evaluated. Compare with Phase B results manually to make the decision recommendation.")

if __name__ == '__main__':
    main()
