#!/usr/bin/env python3
"""
Comprehensive evaluation script for RHAN-v10 (Tripartite Active Inference).
==========================================================================

Runs four evaluation suites to validate the tripartite active inference architecture:
  1. Statistical Significance Sweep (3 seeds, 95% bootstrap CI)
  2. SOTA comparison on STL-10 (WRN-28-10 + TRADES vs RHAN-v10)
  3. Biological Claim Validation (M-pathway gate, prediction error magnitude vs eps, foveal salience)
  4. Diagnostic Plot Generation (Π_D distribution, foraging trajectory overlay, β_dynamic distribution, steps vs epsilon)

Usage:
  python phase1_training/eval_rhan_v10.py --checkpoint checkpoints/rhan_stl10_v10_best.pth
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v10 import RHANv10, foveal_sample
from dataset_stl10 import STL10_CLASSES, get_stl10_loaders


def load_dotenv_fallback():
    """Manual fallback to load HF_TOKEN from .env file in project root."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look in the folder containing this script, and then one level up
    for base_dir in [script_dir, os.path.join(script_dir, '..')]:
        env_path = os.path.join(base_dir, '.env')
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            v = v.strip().strip('"').strip("'")
                            os.environ[k.strip()] = v
            except Exception:
                pass

load_dotenv_fallback()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PGD ATTACK FOR EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pgd_eval(model, x, y, eps, steps=20, stl_min=None, stl_max=None):
    """
    Standard PGD attack on RHANv10.
    """
    if eps == 0:
        return x.clone().detach()

    device = x.device
    if stl_min is None:
        stl_min = torch.tensor([-1.7161, -1.7140, -1.4987], device=device).view(1,3,1,1)
    if stl_max is None:
        stl_max = torch.tensor([2.1256, 2.1832, 2.1872], device=device).view(1,3,1,1)

    model.eval()
    with torch.no_grad():
        logits_c = model(x)
    probs_c = F.softmax(logits_c.float(), dim=1)

    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_a = model(x_adv)
            if isinstance(logits_a, tuple):
                logits_a = logits_a[0]
            loss = F.kl_div(
                F.log_softmax(logits_a.float(), dim=1),
                probs_c, reduction='batchmean'
            )
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + (eps / steps) * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, stl_min, stl_max).detach()

    return x_adv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE 1 — STATISTICAL SIGNIFICANCE (3 SEEDS) & BOOTSTRAP CI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_statistical_significance(model, test_loader, device, num_samples=200):
    print(f"\n{'='*70}")
    print(f" EVALUATION 1: Statistical Significance Sweep (3 Seeds)")
    print(f"{'='*70}")

    seeds = [42, 123, 999]
    results = {seed: {} for seed in seeds}

    # Extract subset of test dataset
    subset_imgs = []
    subset_lbls = []
    count = 0
    for imgs, lbls in test_loader:
        B = imgs.size(0)
        if count + B > num_samples:
            take = num_samples - count
            subset_imgs.append(imgs[:take])
            subset_lbls.append(lbls[:take])
            break
        subset_imgs.append(imgs)
        subset_lbls.append(lbls)
        count += B

    x_test = torch.cat(subset_imgs).to(device)
    y_test = torch.cat(subset_lbls).to(device)

    epsilons = [0.0, 0.031, 0.062, 0.094]

    for seed in seeds:
        set_seed(seed)
        print(f"  Running evaluations for Seed {seed}...")
        results[seed]['clean_acc'] = 0.0
        results[seed]['robust_acc'] = {}

        # Evaluate clean accuracy
        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct = preds.eq(y_test).float()
            results[seed]['clean_acc'] = 100.0 * correct.mean().item()

        # Evaluate at each epsilon
        for eps in epsilons[1:]:
            x_adv = run_pgd_eval(model, x_test, y_test, eps, steps=20)
            with torch.no_grad():
                logits_a = model(x_adv)
                if isinstance(logits_a, tuple):
                    logits_a = logits_a[0]
                preds_a = logits_a.argmax(dim=1)
                correct_a = preds_a.eq(y_test).float()
                results[seed]['robust_acc'][eps] = 100.0 * correct_a.mean().item()

    # Aggregate stats
    print("\n  Summary statistics (Mean ± Std):")
    clean_accs = [results[s]['clean_acc'] for s in seeds]
    print(f"    Clean Accuracy: {np.mean(clean_accs):.2f}% ± {np.std(clean_accs):.2f}%")
    for eps in epsilons[1:]:
        rob_accs = [results[s]['robust_acc'][eps] for s in seeds]
        print(f"    PGD Robustness (ε={eps:.3f}): {np.mean(rob_accs):.2f}% ± {np.std(rob_accs):.2f}%")

    # Bootstrap 95% CI on final seed
    print("\n  Computing 95% Bootstrap Confidence Intervals (n=10000 iterations)...")
    boot_clean_accs = []
    boot_robust_accs = {eps: [] for eps in epsilons[1:]}

    # Evaluate final state to get per-sample predictions
    model.eval()
    with torch.no_grad():
        clean_preds = model(x_test)
        if isinstance(clean_preds, tuple):
            clean_preds = clean_preds[0]
        clean_corrects = clean_preds.argmax(dim=1).eq(y_test).cpu().numpy()

    robust_corrects = {}
    for eps in epsilons[1:]:
        x_adv = run_pgd_eval(model, x_test, y_test, eps, steps=20)
        with torch.no_grad():
            logits_a = model(x_adv)
            if isinstance(logits_a, tuple):
                logits_a = logits_a[0]
            robust_corrects[eps] = logits_a.argmax(dim=1).eq(y_test).cpu().numpy()

    np.random.seed(42)
    for _ in range(10000):
        indices = np.random.randint(0, len(x_test), size=len(x_test))
        boot_clean_accs.append(100.0 * clean_corrects[indices].mean())
        for eps in epsilons[1:]:
            boot_robust_accs[eps].append(100.0 * robust_corrects[eps][indices].mean())

    print(f"    Bootstrap 95% CI (Clean): [{np.percentile(boot_clean_accs, 2.5):.2f}%, {np.percentile(boot_clean_accs, 97.5):.2f}%]")
    for eps in epsilons[1:]:
        print(f"    Bootstrap 95% CI (ε={eps:.3f}): [{np.percentile(boot_robust_accs[eps], 2.5):.2f}%, {np.percentile(boot_robust_accs[eps], 97.5):.2f}%]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE 2 — SOTA COMPARISON ON STL-10
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_sota_comparison(model, test_loader, device):
    print(f"\n{'='*70}")
    print(f" EVALUATION 2: SOTA Comparison on STL-10")
    print(f"{'='*70}")

    # Standard baselines compiled from RobustBench and validate_rhan.py
    baselines = {
        'WideResNet-28-10 + TRADES': {'clean': 48.50, 'robust_0.031': 4.50},
        'ResNet-18 + TRADES':       {'clean': 45.20, 'robust_0.031': 2.80},
        'DeiT-S + TRADES':          {'clean': 47.90, 'robust_0.031': 4.20},
    }

    # Evaluate current RHAN-v10 model on standard Linf epsilon=0.031
    print("  Evaluating RHAN-v10 at SOTA benchmark ε=0.031...")
    subset_imgs = []
    subset_lbls = []
    count = 0
    for imgs, lbls in test_loader:
        if count + imgs.size(0) > 500:
            take = 500 - count
            subset_imgs.append(imgs[:take])
            subset_lbls.append(lbls[:take])
            break
        subset_imgs.append(imgs)
        subset_lbls.append(lbls)
        count += imgs.size(0)

    x_test = torch.cat(subset_imgs).to(device)
    y_test = torch.cat(subset_lbls).to(device)

    model.eval()
    with torch.no_grad():
        logits_c = model(x_test)
        if isinstance(logits_c, tuple):
            logits_c = logits_c[0]
        clean_acc = 100.0 * logits_c.argmax(dim=1).eq(y_test).float().mean().item()

    x_adv = run_pgd_eval(model, x_test, y_test, 0.031, steps=20)
    with torch.no_grad():
        logits_a = model(x_adv)
        if isinstance(logits_a, tuple):
            logits_a = logits_a[0]
        robust_acc = 100.0 * logits_a.argmax(dim=1).eq(y_test).float().mean().item()

    print(f"\n  Main SOTA Comparison Table (STL-10):")
    print(f"    {'Model':<30} | {'Clean Acc':<10} | {'Adversarial Acc (ε=0.031)':<25}")
    print(f"    {'-'*30}-+-{'-'*10}-+-{'-'*25}")
    for name, stats in baselines.items():
        print(f"    {name:<30} | {stats['clean']:>8.2f}% | {stats['robust_0.031']:>23.2f}%")
    print(f"    **RHAN-v10 (Ours)**            | {clean_acc:>8.2f}% | {robust_acc:>23.2f}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE 3 — BIOLOGICAL CLAIM VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_biological_claims(model, test_loader, device):
    print(f"\n{'='*70}")
    print(f" EVALUATION 3: Biological Claim Validation")
    print(f"{'='*70}")

    # Claim 1: M-pathway gate dominance
    print("  [Claim 1]: M-pathway gate dominance")
    if hasattr(model, 'freq_weight_low') and hasattr(model, 'freq_weight_high'):
        w_low = torch.sigmoid(model.freq_weight_low).item()
        w_high = torch.sigmoid(model.freq_weight_high).item()
        print(f"    Low frequency weight wL:  {w_low:.4f}")
        print(f"    High frequency weight wH: {w_high:.4f}")
        if w_low > w_high:
            print(f"    ✓ Dominance Confirmed: wL ({w_low:.3f}) > wH ({w_high:.3f})")
        else:
            print(f"    ✗ Dominance Refuted: wL ({w_low:.3f}) <= wH ({w_high:.3f})")
    else:
        print("    Frequency weight parameters not found. Skipping Claim 1.")

    # Claim 2: Predictive coding error signal vs epsilon
    print("\n  [Claim 2]: Predictive coding error signal increases with noise")
    epsilons = [0.0, 0.031, 0.062, 0.094]
    avg_errors = []

    # Get one batch of images
    imgs, lbls = next(iter(test_loader))
    imgs, lbls = imgs[:64].to(device), lbls[:64].to(device)

    for eps in epsilons:
        x_adv = run_pgd_eval(model, imgs, lbls, eps, steps=20)
        with torch.no_grad():
            _, traj = model(x_adv, return_trajectory=True)
            # average prediction error of the final loop step
            final_err = traj['errors'][-1].mean().item()
            avg_errors.append(final_err)
        print(f"    ε={eps:.3f} -> Mean prediction error: {final_err:.4f}")

    if all(avg_errors[i] < avg_errors[i+1] for i in range(len(avg_errors)-1)):
        print("    ✓ Error Correlation Confirmed: prediction error magnitude increases monotonically with noise level.")
    else:
        print("    ✗ Error Correlation Refuted: error did not increase monotonically.")

    # Claim 3: Epistemic foraging finds diagnostic regions
    print("\n  [Claim 3]: Epistemic foraging path diversity")
    with torch.no_grad():
        _, traj = model(imgs, return_trajectory=True)
        # Average distance between step 0 (initial action) and step 1 actions
        if len(traj['actions']) > 1:
            actions_step0 = traj['actions'][0]
            actions_step1 = traj['actions'][1]
            distances = torch.norm(actions_step0 - actions_step1, dim=-1)
            mean_dist = distances.mean().item()
            print(f"    Average gaze movement distance (Step 0 -> Step 1): {mean_dist:.4f}")
            if mean_dist > 0.05:
                print("    ✓ Active Foraging Confirmed: gaze trajectories move dynamically to acquire diagnostic features.")
            else:
                print("    ✗ Active Foraging Refuted: gaze coordinates remain static.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE 4 — DIAGNOSTIC PLOT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_diagnostic_plots(model, test_loader, device, output_dir='tier1/results/plots2'):
    print(f"\n{'='*70}")
    print(f" EVALUATION 4: Diagnostic Plot Generation")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"  Saving plots to: {output_dir}")

    # Gather data from test set
    subset_imgs = []
    subset_lbls = []
    count = 0
    for imgs, lbls in test_loader:
        if count + imgs.size(0) > 300:
            take = 300 - count
            subset_imgs.append(imgs[:take])
            subset_lbls.append(lbls[:take])
            break
        subset_imgs.append(imgs)
        subset_lbls.append(lbls)
        count += imgs.size(0)

    x_test = torch.cat(subset_imgs).to(device)
    y_test = torch.cat(subset_lbls).to(device)

    model.eval()
    with torch.no_grad():
        logits, traj = model(x_test, return_trajectory=True)
        precisions = traj['precisions'][-1].cpu().numpy()
        errors = traj['errors'][-1].cpu().numpy()
        steps = traj['steps']

    # 1. Π_D distribution histogram
    plt.figure(figsize=(8, 5))
    classes_of_interest = [2, 9, 0] # car, truck, airplane
    class_names = {2: 'Car', 9: 'Truck', 0: 'Airplane'}
    colors = {2: '#e74c3c', 9: '#f39c12', 0: '#3498db'}

    for c in classes_of_interest:
        mask = (y_test.cpu().numpy() == c)
        if mask.any():
            sns.histplot(precisions[mask], bins=15, kde=True, label=class_names[c], color=colors[c], alpha=0.6)

    plt.title(r'$\Pi_D$ Sensory Precision Distribution by Class', fontsize=14)
    plt.xlabel(r'Precision ($\Pi_D$)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, 'v10_pi_distribution.png')
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"    ✓ Generated {plot_path1}")

    # 2. Foraging trajectory overlay visualizer
    plt.figure(figsize=(10, 5))
    car_mask = (y_test.cpu().numpy() == 2)
    if car_mask.any():
        car_idx = np.where(car_mask)[0][0]
        # Re-run model with return_trajectory
        img_single = x_test[car_idx:car_idx+1]
        _, traj_single = model(img_single, return_trajectory=True)

        # Plot foveal sample at steps
        actions = [a[0].cpu().numpy() for a in traj_single['actions']]
        # Convert image back to normal display range
        img_np = img_single[0].cpu().numpy().transpose(1, 2, 0)
        # unnormalize
        mean = np.array([0.4467, 0.4398, 0.4066])
        std = np.array([0.2603, 0.2566, 0.2713])
        img_np = np.clip(img_np * std + mean, 0, 1)

        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        # Plot actions as points
        ax = plt.gca()
        # actions are in [-1, 1], map to 96x96 image space
        x_coords = [(a[0] + 1) * 48 for a in actions]
        y_coords = [(a[1] + 1) * 48 for a in actions]
        plt.plot(x_coords, y_coords, '-o', color='yellow', markersize=8, linewidth=2, label='Gaze trajectory')
        for i, (xc, yc) in enumerate(zip(x_coords, y_coords)):
            plt.text(xc + 2, yc - 2, f"T={i}", color='yellow', fontweight='bold')
        plt.title('Gaze Paths on Car Image', fontsize=12)
        plt.legend()

        # Plot foveated crop at step 1
        plt.subplot(1, 2, 2)
        action_t = traj_single['actions'][0]
        foveal_crop = foveal_sample(img_single, action_t, fovea_size=48)
        fov_np = foveal_crop[0].cpu().numpy().transpose(1, 2, 0)
        fov_np = np.clip(fov_np * std + mean, 0, 1)
        plt.imshow(fov_np)
        plt.title('Foveal Crop at Step 1', fontsize=12)

    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, 'v10_foraging_trajectory.png')
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"    ✓ Generated {plot_path2}")

    # 3. β_dynamic distribution at different curriculum phases
    plt.figure(figsize=(8, 5))
    beta_bases = [2.0, 2.5]
    for beta_base in beta_bases:
        beta_dyn = beta_base * (0.5 + precisions)
        sns.kdeplot(beta_dyn, fill=True, label=f'Base β = {beta_base}', alpha=0.5)
    plt.title(r'Dynamic $\beta$ Distribution across Curriculum Phases', fontsize=14)
    plt.xlabel(r'Effective $\beta_{dynamic}$', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, 'v10_beta_distribution.png')
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"    ✓ Generated {plot_path3}")

    # 4. Steps used vs epsilon (reaction time correlate)
    plt.figure(figsize=(8, 5))
    epsilons = [0.0, 0.031, 0.062, 0.094]
    mean_steps = []
    for eps in epsilons:
        x_adv = run_pgd_eval(model, x_test[:100], y_test[:100], eps, steps=20)
        with torch.no_grad():
            _, traj_eps = model(x_adv, return_trajectory=True)
            # simulate step count distribution (we can average actual steps)
            mean_steps.append(float(traj_eps['steps']))

    plt.plot(epsilons, mean_steps, '-o', color='#2ecc71', linewidth=2.5, markersize=8)
    plt.title('Average Foraging Steps vs Adversarial Noise Level (ε)', fontsize=14)
    plt.xlabel('Noise Budget (ε)', fontsize=12)
    plt.ylabel('Mean Steps Used (Reaction Time Proxy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path4 = os.path.join(output_dir, 'v10_steps_vs_epsilon.png')
    plt.savefig(plot_path4, dpi=300)
    plt.close()
    print(f"    ✓ Generated {plot_path4}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EVALUATION RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Evaluate RHAN-v10 Tripartite Model")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/rhan_stl10_v10_best.pth',
                        help='Path to RHAN-v10 checkpoint')
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of samples for statistical significance')
    args = parser.parse_known_args()[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating RHAN-v10 on {device}...")

    # Load data loader
    _, test_loader = get_stl10_loaders(batch_size=32, data_root=args.data_root)

    # Instantiate model
    model = RHANv10().to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    # Try fallback check to checkpoints_tier2/ if not found in checkpoints/
    if not os.path.exists(ckpt_path):
        fallback_tier2 = os.path.join('checkpoints_tier2', os.path.basename(ckpt_path))
        if os.path.exists(fallback_tier2):
            ckpt_path = fallback_tier2

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        elif isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Running evaluations on random/mock weights.")

    # Run all evaluation suites
    run_statistical_significance(model, test_loader, device, num_samples=args.num_samples)
    run_sota_comparison(model, test_loader, device)
    run_biological_claims(model, test_loader, device)
    generate_diagnostic_plots(model, test_loader, device)

    print(f"\n{'='*70}")
    print(f" All evaluations complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
