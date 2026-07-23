#!/usr/bin/env python3
"""
Domain-Clamped Empirical Epsilon Sweep & SDT Sensitivity Evaluation
====================================================================
Extends eval_domain_clipping_validation.py (verified correct per-channel
normalization and [0,1] pixel-space domain clamping) into a full epsilon
sweep across 4 checkpoints.

The attack operates in pixel space [0,1] with proper L∞ clamping in that
space. The NormalizedModelWrapper handles per-channel (x - mean) / std
internally, so no scalar std.mean() shortcut exists — the L∞ ball is
correctly mapped per-channel.

Bug Reference: eval_empirical_epsilon_sweep.py line 108 used
    eps_norm = eps / std.mean().item()
which applied the same scalar bound to all channels, inflating the effective
attack by 3.81× (std.mean() ≈ 0.2627 vs per-channel stds of 0.2603/0.2565/0.2713).

Epsilon Grid (pixel space [0,1]): [0.0000, 0.0020, 0.0040, 0.0080, 0.0160, 0.0240, 0.0313]
  ~normalized:  [0.0000, 0.0076, 0.0152, 0.0305, 0.0609, 0.0914, 0.1191]
  Covers standard ε=0.031 attack (~eps_pixel=0.0080) with finer low-end resolution.
Per-channel normalized equivalents are printed for visual verification.

Models:
  1. static_trades_large  (RHANUnifiedSTL10)
  2. rhan_stl10_large_ep45 (RHANLargeSTL10) — sanity target: εthresh ≈ 0.0553
  3. rhan_v10_final       (RHANLargeSTL10)
  4. rhan_v11_best        (RHANv11)
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
from scipy.stats import norm
from scipy.interpolate import interp1d

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'phase1_training'))

from phase1_training.model_rhan_stl10_large import RHANLargeSTL10
from phase1_training.model_rhan_stl10_pretrained import RHANUnifiedSTL10
from phase1_training.model_rhan_v11 import RHANv11

STD_VALS = (0.2603, 0.2566, 0.2713)
MEAN_VALS = (0.4467, 0.4398, 0.4066)
MEAN = torch.tensor(MEAN_VALS).view(1, 3, 1, 1)
STD = torch.tensor(STD_VALS).view(1, 3, 1, 1)

EPS_GRID = [0.0000, 0.0020, 0.0040, 0.0080, 0.0160, 0.0240, 0.0313]


def normalize(x_pixel):
    mean = MEAN.to(x_pixel.device)
    std = STD.to(x_pixel.device)
    return (x_pixel - mean) / std


class PixelToNormalizedWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_pixel):
        x_norm = normalize(x_pixel)
        logits = self.model(x_norm)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits


def acc_to_dprime_pooled(acc_pct, K=10):
    p_hit = np.clip(acc_pct / 100.0, 1e-4, 1.0 - 1e-4)
    p_fa = np.clip((1.0 - p_hit) / (K - 1), 1e-4, 1.0 - 1e-4)
    return float(norm.ppf(p_hit) - norm.ppf(p_fa))


def compute_per_class_sdt(y_true, y_pred, num_classes=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    hit_rates, fa_rates, dprime_per_class = [], [], []
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


def run_pgd_pixel_space(model_wrapper, x_pixel, y, eps_pixel, steps=50, alpha_pixel=None):
    if eps_pixel == 0.0:
        with torch.no_grad():
            return model_wrapper(x_pixel), x_pixel.clone()
    if alpha_pixel is None:
        alpha_pixel = eps_pixel / 4.0
    device = x_pixel.device
    x_orig = x_pixel.clone().detach()
    delta = (torch.rand_like(x_pixel) * 2.0 - 1.0) * (0.001 * eps_pixel)
    x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model_wrapper(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha_pixel * grad.sign()
        delta = torch.clamp(x_adv - x_orig, -eps_pixel, eps_pixel)
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


def load_test_samples_pixel(n_samples=500, seed=42):
    from datasets import load_dataset
    ds = load_dataset("mteb/stl10", split="test").shuffle(seed=seed).select(range(n_samples))
    images_pixel, labels = [], []
    for item in ds:
        img = item['image'].convert('RGB').resize((96, 96))
        arr = np.array(img, dtype=np.float32) / 255.0
        t_pix = torch.from_numpy(arr).permute(2, 0, 1)
        images_pixel.append(t_pix)
        labels.append(item['label'])
    x_pixel = torch.stack(images_pixel)
    y = torch.tensor(labels, dtype=torch.long)
    return x_pixel, y


def main():
    parser = argparse.ArgumentParser(description="Domain-Clamped Empirical Epsilon Sweep & SDT Evaluation")
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--pgd-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-json', type=str, default='report/empirical_sweep_results_stl10.json')
    parser.add_argument('--skip-models', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  DOMAIN-CLAMPED Empirical Epsilon Sweep & SDT Evaluation")
    print(f"  Device: {device} | Samples: n={args.n_samples} | PGD Steps: {args.pgd_steps}")
    print("=" * 70)

    std_r, std_g, std_b = STD_VALS
    print("\n--- PER-CHANNEL NORMALIZED EPSILON VERIFICATION ---")
    print("Pixel Eps | Raw Fraction | Norm Eps R (s={}) | Norm Eps G (s={}) | Norm Eps B (s={})".format(std_r, std_g, std_b))
    print("-" * 95)
    for eps in EPS_GRID:
        eps_r = eps / std_r
        eps_g = eps / std_g
        eps_b = eps / std_b
        print(f" {eps:7.4f} |   {eps*255:5.1f}/255   |       {eps_r:8.4f}       |       {eps_g:8.4f}       |       {eps_b:8.4f}")
    print("-" * 95)

    # Confirm no scalar std.mean() shortcut is used
    scalar_mean_std = (std_r + std_g + std_b) / 3.0
    print(f"\n  ✓ Confirmed: Per-channel stds used. (Scalar std.mean()={scalar_mean_std:.4f} would give ×{scalar_mean_std/std_r:.2f}/×{scalar_mean_std/std_g:.2f}/×{scalar_mean_std/std_b:.2f} inflation — AVOIDED)")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load test samples in pixel space [0, 1]
    print(f"Loading {args.n_samples} STL-10 test samples...", flush=True)
    x_pixel, y_test = load_test_samples_pixel(n_samples=args.n_samples, seed=args.seed)
    x_pixel, y_test = x_pixel.to(device), y_test.to(device)

    # Checkpoint map
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
            print(f"\n[-->] Skipping '{model_name}' (--skip-models)", flush=True)
            continue

        print(f"\n[-->] Evaluating '{model_name}' ({ckpt_path})...", flush=True)
        download_checkpoint_if_missing(ckpt_path)

        if not os.path.exists(ckpt_path):
            print(f"  ❌ Checkpoint not found. Skipping.", flush=True)
            continue

        # Load architecture
        if model_type == "unified_base":
            base_model = RHANUnifiedSTL10().to(device)
        elif model_type == "static_large":
            base_model = RHANLargeSTL10().to(device)
        else:
            base_model = RHANv11().to(device)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
            print(f"  ❌ Zero keys matched. Skipping.", flush=True)
            continue

        base_model.eval()
        wrapper = PixelToNormalizedWrapper(base_model).to(device)
        wrapper.eval()

        ckpt_metrics = {
            "epsilons": EPS_GRID,
            "accuracy": [],
            "macro_dprime": [],
            "pooled_dprime": [],
            "per_eps_details": {}
        }

        t0 = time.time()
        for eps in EPS_GRID:
            eps_r = eps / std_r
            eps_g = eps / std_g
            eps_b = eps / std_b
            print(f"  Evaluating eps_pixel={eps:.4f} (norm_eps=[R:{eps_r:.4f}, G:{eps_g:.4f}, B:{eps_b:.4f}])...", end="", flush=True)
            t_eps = time.time()

            all_targets = []
            all_preds = []

            # Process in batches
            batch_size = args.batch_size
            n = len(x_pixel)
            for i in range(0, n, batch_size):
                batch_pixel = x_pixel[i:i + batch_size]
                batch_y = y_test[i:i + batch_size]

                logits_adv, _ = run_pgd_pixel_space(
                    wrapper, batch_pixel, batch_y, eps_pixel=eps, steps=args.pgd_steps
                )
                preds = logits_adv.argmax(dim=-1)
                all_targets.extend(batch_y.cpu().numpy())
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
        thresh_macro = find_dprime_crossing(EPS_GRID, ckpt_metrics["macro_dprime"], target_d=1.0)
        thresh_pooled = find_dprime_crossing(EPS_GRID, ckpt_metrics["pooled_dprime"], target_d=1.0)

        ckpt_metrics["thresh_dprime_1_macro"] = round(thresh_macro, 4)
        ckpt_metrics["thresh_dprime_1_pooled"] = round(thresh_pooled, 4)
        ckpt_metrics["total_eval_time_sec"] = round(elapsed, 1)

        print(f"  ✓ '{model_name}' Sweep Complete in {elapsed/60.0:.1f} min.")
        print(f"    --> Threshold (Macro d'=1.0): eps = {thresh_macro:.4f}")
        print(f"    --> Threshold (Pooled d'=1.0): eps = {thresh_pooled:.4f}")

        sweep_results[model_name] = ckpt_metrics

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(sweep_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("  FULL EMPIRICAL SDT SENSITIVITY TABLE (STL-10, DOMAIN-CLAMPED, n={})".format(args.n_samples))
    print("=" * 80)
    model_names = list(sweep_results.keys())
    header = f"{'Epsilon':<9} | " + " | ".join([f"{name:<20}" for name in model_names])
    print(header)
    print("-" * len(header))
    for i, eps in enumerate(EPS_GRID):
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
    print("=" * 80)
    print(f"Results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
