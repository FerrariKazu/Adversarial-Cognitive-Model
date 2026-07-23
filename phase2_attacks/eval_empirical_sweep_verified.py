#!/usr/bin/env python3
"""
Verified Empirical Epsilon Sweep & SDT Sensitivity Evaluation for STL-10
========================================================================
Extended from phase2_attacks/eval_domain_clipping_validation.py to ensure
100% verified domain clipping and exact per-channel epsilon normalization.

Evaluates 4 key checkpoints across an 7-point epsilon grid (n=500 per point):
  1. static_trades_large   (checkpoints/rhan_stl10_tdv_trades_actual.pth)
  2. rhan_stl10_large_ep45 (checkpoints/rhan_stl10_large_pseudolabel_rolling.pth) -> Sanity target: eps_thresh ≈ 0.0553
  3. rhan_v10_final        (checkpoints/rhan_stl10_v10_best.pth)
  4. rhan_v11_best         (checkpoints/rhan_stl10_v11_best.pth)

Epsilon Grid: [0.0000, 0.0313, 0.0625, 0.0940, 0.1500, 0.2000, 0.3000]
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase1_training')))

# Import verified model architectures
from phase1_training.model_rhan_stl10 import RHANSTL10
from phase1_training.model_rhan_stl10_large import RHANLargeSTL10
from phase1_training.model_rhan_stl10_pretrained import RHANUnifiedSTL10
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
    N = len(y_true)

    hit_rates = []
    fa_rates = []
    dprime_per_class = []

    for c in range(num_classes):
        is_c = (y_true == c)
        not_c = (y_true != c)
        n_c = np.sum(is_c)
        n_not_c = np.sum(not_c)

        if n_c == 0 or n_not_c == 0:
            hit_rates.append(0.0)
            fa_rates.append(0.0)
            dprime_per_class.append(0.0)
            continue

        hits = np.sum((y_pred == c) & is_c)
        fas = np.sum((y_pred == c) & not_c)

        # 1/(2N) logit correction for boundary performance
        hit_rate = (hits + 0.5) / (n_c + 1.0)
        fa_rate = (fas + 0.5) / (n_not_c + 1.0)

        hit_rate = np.clip(hit_rate, 1e-4, 1.0 - 1e-4)
        fa_rate = np.clip(fa_rate, 1e-4, 1.0 - 1e-4)

        dp = float(norm.ppf(hit_rate) - norm.ppf(fa_rate))
        hit_rates.append(float(hits / n_c))
        fa_rates.append(float(fas / n_not_c))
        dprime_per_class.append(dp)

    macro_dprime = float(np.mean(dprime_per_class))
    return hit_rates, fa_rates, dprime_per_class, macro_dprime


class PixelToNormalizedWrapper(nn.Module):
    """
    Wraps model so input is in pixel space [0, 1].
    Normalizes input internally using exact per-channel MEAN and STD tensors.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_pixel):
        device = x_pixel.device
        mean = MEAN.to(device)
        std = STD.to(device)
        x_norm = (x_pixel - mean) / std
        logits = self.model(x_norm)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits


def run_pgd_pixel_space(model_wrapper, x_pixel, y, eps_pixel, steps=50, alpha_pixel=None):
    """
    PGD-L∞ attack operating directly in pixel space [0, 1].
    Enforces exact L_infinity perturbation bounds [-eps_pixel, +eps_pixel] in pixel space
    and strict domain clamping to [0.0, 1.0].
    """
    if eps_pixel == 0.0:
        with torch.no_grad():
            return model_wrapper(x_pixel), x_pixel.clone()

    if alpha_pixel is None:
        alpha_pixel = eps_pixel / 4.0

    device = x_pixel.device
    x_orig = x_pixel.clone().detach()

    # Small random initialization within [-eps, +eps]
    delta = (torch.rand_like(x_pixel) * 2.0 - 1.0) * (0.001 * eps_pixel)
    x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model_wrapper(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # L_infinity step in pixel space [0, 1]
        x_adv = x_adv.detach() + alpha_pixel * grad.sign()

        # Project back to L_infinity ball [-eps_pixel, +eps_pixel] in pixel space
        delta = torch.clamp(x_adv - x_orig, -eps_pixel, eps_pixel)

        # Domain clamp strictly to valid pixel bounds [0.0, 1.0]
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    with torch.no_grad():
        logits_adv = model_wrapper(x_adv)

    return logits_adv, x_adv


def find_dprime_crossing(eps_list, dprime_list, target_d=1.0):
    eps_arr = np.array(eps_list)
    d_arr = np.array(dprime_list)

    if np.max(d_arr) < target_d:
        return 0.0
    if np.min(d_arr) > target_d:
        return float(np.max(eps_arr))

    # Interpolate along sorted d' values
    sort_idx = np.argsort(-d_arr)
    eps_sorted = eps_arr[sort_idx]
    d_sorted = d_arr[sort_idx]

    try:
        f_interp = interp1d(d_sorted, eps_sorted, kind='linear')
        return float(f_interp(target_d))
    except Exception:
        return 0.0


def download_checkpoint_if_missing(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found locally at {ckpt_path}. Attempting download from HuggingFace...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            import shutil

            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                try:
                    from kaggle_secrets import UserSecretsClient
                    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
                except Exception:
                    pass
            if not hf_token:
                try:
                    from google.colab import userdata
                    hf_token = userdata.get('HF_TOKEN')
                except Exception:
                    pass

            filename = os.path.basename(ckpt_path)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            downloaded = hf_hub_download(
                repo_id='FerrariKazu/rhan-checkpoints',
                filename=filename,
                repo_type='dataset',
                token=hf_token
            )
            shutil.copy2(downloaded, ckpt_path)
            print(f"  ✓ Successfully downloaded checkpoint to {ckpt_path}", flush=True)
        except Exception as e:
            print(f"  ❌ Hugging Face download failed for {ckpt_path}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Verified Empirical Epsilon Sweep & SDT Threshold Evaluation for STL-10")
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to STL-10 dataset')
    parser.add_argument('--n-samples', type=int, default=500, help='Sample size per epsilon point (minimum 500)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--pgd-steps', type=int, default=50, help='PGD attack steps (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset subsetting')
    parser.add_argument('--output-json', type=str, default='report/empirical_sweep_results_stl10.json', help='Output JSON path')
    parser.add_argument('--skip-models', type=str, default='', help='Comma-separated model names to skip')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("======================================================================")
    print("  VERIFIED Empirical Epsilon Sweep & SDT Evaluation Session")
    print(f"  Device: {device} | Samples: n={args.n_samples} | PGD Steps: {args.pgd_steps}")
    print("======================================================================")

    # Print per-channel normalized epsilon mapping verification table
    std_r, std_g, std_b = 0.2603, 0.2565, 0.2713
    eps_grid = [0.0000, 0.0313, 0.0625, 0.0940, 0.1500, 0.2000, 0.3000]

    print("\n--- PER-CHANNEL NORMALIZED EPSILON VERIFICATION ---")
    print("Pixel Eps | Raw Fraction | Norm Eps R (s=0.2603) | Norm Eps G (s=0.2565) | Norm Eps B (s=0.2713)")
    print("---------------------------------------------------------------------------------------------------")
    for eps in eps_grid:
        eps_r = eps / std_r
        eps_g = eps / std_g
        eps_b = eps / std_b
        print(f" {eps:7.4f} |   {eps*255:5.1f}/255   |       {eps_r:8.4f}       |       {eps_g:8.4f}       |       {eps_b:8.4f}")
    print("---------------------------------------------------------------------------------------------------\n")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare STL-10 Test Dataset in Pixel Space [0, 1]
    transform_pixel = transforms.Compose([
        transforms.ToTensor(), # Loads images in [0, 1] range directly
    ])

    full_testset = datasets.STL10(root=args.data_dir, split='test', download=True, transform=transform_pixel)
    indices = torch.randperm(len(full_testset))[:args.n_samples].tolist()
    subset_dataset = Subset(full_testset, indices)
    test_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Checkpoint paths
    ckpt_map = {
        "static_trades_large": ("checkpoints/rhan_stl10_tdv_trades_actual.pth", "unified_base"),
        "rhan_stl10_large_ep45": ("checkpoints/rhan_stl10_large_pseudolabel_rolling.pth", "static_large"),
        "rhan_v10_final": ("checkpoints/rhan_stl10_v10_best.pth", "static_large"),
        "rhan_v11_best": ("checkpoints/rhan_stl10_v11_best.pth", "rhan_v11"),
    }

    skip_list = [s.strip() for s in args.skip_models.split(',') if s.strip()]
    sweep_results = {}

    for model_name, (ckpt_path, model_type) in ckpt_map.items():
        if model_name in skip_list:
            print(f"\n[-->] Skipping Checkpoint: '{model_name}' (explicitly requested in --skip-models)", flush=True)
            continue

        print(f"\n[-->] Evaluating Checkpoint: '{model_name}' ({ckpt_path})...", flush=True)
        download_checkpoint_if_missing(ckpt_path)

        if not os.path.exists(ckpt_path):
            print(f"  ❌ ERROR: Checkpoint file '{ckpt_path}' not found! Skipping.", flush=True)
            continue

        # Load Architecture
        if model_type == "unified_base":
            base_model = RHANUnifiedSTL10().to(device)
        elif model_type == "static_large":
            base_model = RHANLargeSTL10().to(device)
        else:  # rhan_v11
            base_model = RHANv11().to(device)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Handle different checkpoint nesting formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if state_dict and all(k.startswith('model.') for k in list(state_dict.keys())[:5]):
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}

        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        matched = len(state_dict) - len(unexpected)
        print(f"  Loaded {matched}/{len(state_dict)} keys ({len(missing)} missing, {len(unexpected)} unexpected)", flush=True)

        if matched == 0:
            print(f"  ❌ ERROR: Zero keys matched for '{model_name}'. Skipping.", flush=True)
            continue

        base_model.eval()
        model_wrapper = PixelToNormalizedWrapper(base_model).to(device)
        model_wrapper.eval()

        ckpt_metrics = {
            "epsilons": eps_grid,
            "accuracy": [],
            "macro_dprime": [],
            "pooled_dprime": [],
            "per_eps_details": {}
        }

        t0 = time.time()
        for eps in eps_grid:
            eps_r = eps / std_r
            eps_g = eps / std_g
            eps_b = eps / std_b
            print(f"  Evaluating eps_pixel={eps:.4f} (norm_eps=[R:{eps_r:.4f}, G:{eps_g:.4f}, B:{eps_b:.4f}])...", end="", flush=True)
            t_eps = time.time()

            all_targets = []
            all_preds = []

            for images_pixel, targets in test_loader:
                images_pixel, targets = images_pixel.to(device), targets.to(device)

                logits_adv, _ = run_pgd_pixel_space(
                    model_wrapper, images_pixel, targets, eps_pixel=eps, steps=args.pgd_steps
                )
                preds = logits_adv.argmax(dim=-1)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)

            correct = np.sum(all_targets == all_preds)
            acc = float(100.0 * correct / len(all_targets))

            hit_rates, fa_rates, dprime_per_class, macro_dp = compute_per_class_sdt(all_targets, all_preds)
            pooled_dp = acc_to_dprime_pooled(acc)

            dt = time.time() - t_eps

            ckpt_metrics["accuracy"].append(acc)
            ckpt_metrics["macro_dprime"].append(macro_dp)
            ckpt_metrics["pooled_dprime"].append(pooled_dp)
            ckpt_metrics["per_eps_details"][str(eps)] = {
                "accuracy": acc,
                "macro_dprime": macro_dp,
                "pooled_dprime": pooled_dp,
                "hit_rates": hit_rates,
                "fa_rates": fa_rates,
                "dprime_per_class": dprime_per_class
            }

            print(f" Acc: {acc:>5.2f}% | Macro d': {macro_dp:>6.4f} | Pooled d': {pooled_dp:>6.4f} ({dt:.1f}s)", flush=True)

        elapsed = time.time() - t0
        thresh_macro = find_dprime_crossing(eps_grid, ckpt_metrics["macro_dprime"], target_d=1.0)
        thresh_pooled = find_dprime_crossing(eps_grid, ckpt_metrics["pooled_dprime"], target_d=1.0)

        ckpt_metrics["thresh_dprime_1_macro"] = round(thresh_macro, 4)
        ckpt_metrics["thresh_dprime_1_pooled"] = round(thresh_pooled, 4)
        ckpt_metrics["total_eval_time_sec"] = round(elapsed, 1)

        print(f"  ✓ Model '{model_name}' Sweep Complete in {elapsed/60.0:.1f} min.", flush=True)
        print(f"    --> Threshold (Macro d'=1.0): eps = {thresh_macro:.4f}", flush=True)
        print(f"    --> Threshold (Pooled d'=1.0): eps = {thresh_pooled:.4f}", flush=True)

        sweep_results[model_name] = ckpt_metrics

    # Write output to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(sweep_results, f, indent=2)

    print("\n==========================================================================================")
    print("  FULL EMPIRICAL SDT SENSITIVITY TABLE (STL-10, VERIFIED DOMAIN CLAMPING, n=500)")
    print("==========================================================================================")

    model_names = list(sweep_results.keys())
    header = f"{'Epsilon':<9} | " + " | ".join([f"{name:<20}" for name in model_names])
    print(header)
    print("-" * len(header))

    for i, eps in enumerate(eps_grid):
        row = f"{eps:<9.4f} | "
        for name in model_names:
            acc = sweep_results[name]["accuracy"][i]
            dp = sweep_results[name]["macro_dprime"][i]
            row += f"{dp:>6.3f} ({acc:>4.1f}%)    | "
        print(row)

    print("-" * len(header))
    thresh_row = f"{'eps_thresh':<9} | "
    for name in model_names:
        tr = sweep_results[name]["thresh_dprime_1_macro"]
        thresh_row += f"{tr:>18.4f} | "
    print(thresh_row)
    print("==========================================================================================")
    print(f"Empirical results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
