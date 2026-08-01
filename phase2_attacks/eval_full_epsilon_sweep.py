#!/usr/bin/env python3
"""
Full Epsilon Sweep Evaluation
==============================
Extends eval_domain_clipping_validation.py (verified-correct per-channel clamping)
into a full sweep over epsilon values for four checkpoints.

Per-channel epsilon conversion (NO std.mean() shortcut):
    eps_norm_per_channel = eps_pixel / std   (element-wise, shape 1x3x1x1)

STL-10 std = [0.2603, 0.2566, 0.2713]

Usage (Colab/Kaggle):
    # Pixel-space (DEFAULT): listed eps is a [0,1] pixel bound, converted
    #   per-channel via /std  -> listed 0.0313 attacks at norm ~0.12 (Run A v1).
    python3 phase2_attacks/eval_full_epsilon_sweep.py \
        --n-samples 500 --pgd-steps 50 --output-dir ./sweep_results

    # NORM-space (--eps-norm-space): listed eps applied DIRECTLY to normalized
    #   inputs — the Finding-17 baseline table convention (quick_eval_hf.py /
    #   eval_rhan_v11.py: eps=0.031/0.062/0.094 in norm space). Use for matched sweep:
    python3 phase2_attacks/eval_full_epsilon_sweep.py \
        --n-samples 500 --pgd-steps 50 --eps-norm-space \
        --eps-list 0.0 0.031 0.062 0.094 --output-dir ./sweep_results

Checkpoints expected under ./checkpoints/ (or pass --ckpt-specs):
    static_trades_large        -> rhan_stl10_large_pseudolabel_best.pth  (arch=large)
    rhan_stl10_large_ep45      -> rhan_stl10_large_ep45.pth              (arch=large)
    rhan_v10_final             -> rhan_v10_final.pth                     (arch=v10)
    rhan_stl10_v11_best        -> rhan_stl10_v11_best.pth                (arch=v11)
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reduce fragmentation on large-batch CUDA allocs
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'phase1_training'))

# ── STL-10 normalization constants ─────────────────────────────────────────────
MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
STD  = torch.tensor([0.2603, 0.2566, 0.2713]).view(1, 3, 1, 1)

DEFAULT_EPS_LIST = [0.0, 0.0313, 0.0625, 0.094, 0.15, 0.2, 0.3]

DEFAULT_CHECKPOINTS = [
    ("static_trades_large",   "checkpoints/rhan_stl10_large_pseudolabel_best.pth", "large"),
    ("rhan_stl10_large_ep45", "checkpoints/rhan_stl10_large_ep45.pth",             "large"),
    ("rhan_v10_final",        "checkpoints/rhan_v10_final.pth",                    "v10"),
    ("rhan_stl10_v11_best",   "checkpoints/rhan_stl10_v11_best.pth",               "v11"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_domain_bounds(device):
    stl_min = (torch.zeros(1, 3, 1, 1, device=device) - MEAN.to(device)) / STD.to(device)
    stl_max = (torch.ones(1, 3, 1, 1, device=device) - MEAN.to(device)) / STD.to(device)
    return stl_min, stl_max


def resolve_eps_norm(eps, device, norm_space=False):
    """Map a listed epsilon to a (1,3,1,1) per-channel bound in normalized space.

    norm_space=False (default): listed eps is a PIXEL-space [0,1] bound; convert
        per-channel via eps / STD (Run A's first sweep: listed 0.0313 -> norm ~0.12).
    norm_space=True: listed eps IS the NORM-space bound, applied directly to the
        normalized inputs — the exact Finding-17 baseline convention
        (quick_eval_hf.py / eval_rhan_v11.py use eps=0.031/0.062/0.094 in norm space).
    """
    if norm_space:
        return torch.full((1, 3, 1, 1), eps, device=device)
    return eps / STD.to(device)


def log_eps(eps_listed, eps_norm, prefix="", norm_space=False):
    """Print the listed epsilon and its per-channel norm-space equivalent.

    norm_space=True means the listed epsilon IS the norm-space bound, so the
    label reflects that instead of misnaming it a pixel value.
    """
    key = "eps_norm" if norm_space else "eps_pixel"
    r = eps_norm[0, 0, 0, 0].item()
    g = eps_norm[0, 1, 0, 0].item()
    b = eps_norm[0, 2, 0, 0].item()
    print(f"  [EPS] {prefix}{key}={eps_listed:.4f} -> "
          f"eps_norm=[R:{r:.4f}, G:{g:.4f}, B:{b:.4f}]", flush=True)


# ── Data loader ────────────────────────────────────────────────────────────────

def load_test_samples(n_samples, seed=42):
    """Returns tensors kept on CPU; caller moves batches to GPU as needed."""
    print(f"--> Loading {n_samples} test samples (seed={seed})...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("mteb/stl10", split="test").shuffle(seed=seed).select(range(n_samples))
    images_norm, labels = [], []
    for item in ds:
        img = item['image'].convert('RGB').resize((96, 96))
        arr = np.array(img, dtype=np.float32) / 255.0
        t_pix = torch.from_numpy(arr).permute(2, 0, 1)
        t_norm = (t_pix - MEAN.squeeze(0)) / STD.squeeze(0)
        images_norm.append(t_norm)
        labels.append(item['label'])
    # Keep on CPU — batches are sent to GPU inside the attack loop
    return torch.stack(images_norm), torch.tensor(labels, dtype=torch.long)


# ── PGD (verified-correct — same clamping as eval_domain_clipping_validation.py) ──

def _pgd_batch(model, xb, probs_c, eps_norm, alpha, stl_min, stl_max, steps):
    """PGD on a single GPU batch."""
    x_adv = xb.clone().detach() + 0.001 * torch.randn_like(xb)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            logits_a = model(x_adv)
            if isinstance(logits_a, tuple):
                logits_a = logits_a[0]
            loss = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                            probs_c, reduction='batchmean')
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - xb, -eps_norm, eps_norm)
        x_adv = torch.clamp(xb + delta, stl_min, stl_max).detach()
    return x_adv


def run_pgd_batched(model, x_norm_cpu, y_cpu, eps, steps, device, batch_size=50,
                    norm_space=False):
    """
    Batched PGD — data lives on CPU, each mini-batch is moved to GPU individually
    so we never materialise 500 full-resolution images + gradients at once.
    """
    stl_min, stl_max = get_domain_bounds(device)
    eps_norm = resolve_eps_norm(eps, device, norm_space=norm_space)   # (1,3,1,1)
    alpha = eps_norm / 4.0

    model.eval()
    n = x_norm_cpu.size(0)
    adv_chunks = []

    for start in range(0, n, batch_size):
        xb = x_norm_cpu[start:start + batch_size].to(device)
        with torch.no_grad():
            logits_c = model(xb)
            if isinstance(logits_c, tuple):
                logits_c = logits_c[0]
        probs_c = F.softmax(logits_c.float(), dim=1)
        xb_adv = _pgd_batch(model, xb, probs_c, eps_norm, alpha, stl_min, stl_max, steps)
        adv_chunks.append(xb_adv.cpu())
        del xb, xb_adv, logits_c, probs_c
        torch.cuda.empty_cache()

    return torch.cat(adv_chunks, dim=0)  # on CPU


# ── d-prime (batched inference) ────────────────────────────────────────────────

def compute_dprime_batched(model, x_adv_cpu, y_true_cpu, device, batch_size=50):
    from scipy import stats as sps
    model.eval()
    all_probs = []
    n = x_adv_cpu.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x_adv_cpu[start:start + batch_size].to(device)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_probs.append(F.softmax(logits.float(), dim=1).cpu())
            del xb
    probs = torch.cat(all_probs, dim=0).numpy()
    y_np = y_true_cpu.numpy()
    dprimes = []
    for c in range(probs.shape[1]):
        hit = probs[y_np == c, c]
        fa  = probs[y_np != c, c]
        if len(hit) < 2 or len(fa) < 2:
            continue
        hr = np.clip(np.mean(hit > 0.5), 1e-6, 1 - 1e-6)
        fr = np.clip(np.mean(fa  > 0.5), 1e-6, 1 - 1e-6)
        dprimes.append(float(sps.norm.ppf(hr) - sps.norm.ppf(fr)))
    return float(np.mean(dprimes)) if dprimes else float('nan')


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(arch, ckpt_path, device, freeze_gaze=False):
    if arch == "v11":
        from phase1_training.model_rhan_v11 import RHANv11
        model = RHANv11().to(device)
    elif arch == "large":
        from phase1_training.model_rhan_stl10_large import RHANLargeSTL10
        model = RHANLargeSTL10().to(device)
    elif arch == "v10":
        from phase1_training.model_rhan_v10 import RHANv10
        model = RHANv10().to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        for k in ('model', 'model_state_dict', 'state_dict'):
            if isinstance(state, dict) and k in state:
                state = state[k]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(missing)
        print(f"  Loaded {n_loaded}/{len(state)} keys "
              f"({len(missing)} missing, {len(unexpected)} unexpected)", flush=True)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Training must complete before running eval."
        )

    model.eval()
    if freeze_gaze:
        if hasattr(model, 'freeze_gaze'):
            model.freeze_gaze = True
            print("  ISOLATION TEST: foveal gaze frozen to image center (0,0)", flush=True)
        else:
            print(f"  WARNING: --freeze-gaze requested but arch={arch} has no freeze_gaze attr",
                  flush=True)
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples',  type=int,   default=500)
    parser.add_argument('--pgd-steps',  type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=50,
                        help='Mini-batch size for PGD and inference (reduces VRAM use)')
    parser.add_argument('--output-dir', type=str,   default='./sweep_results')
    parser.add_argument('--eps-list',   type=float, nargs='+', default=DEFAULT_EPS_LIST)
    parser.add_argument('--eps-norm-space', action='store_true',
                        help='Treat listed epsilons as NORM-space bounds (applied directly to '
                             'normalized inputs), matching the Finding-17 baseline table '
                             '(quick_eval_hf/eval_rhan_v11 convention). Default: pixel-space [0,1] '
                             'converted per-channel via /std.')
    parser.add_argument('--freeze-gaze', action='store_true',
                        help='ISOLATION TEST (Run B): freeze foveal gaze to image center (0,0) '
                             'during evaluation (must match --freeze-gaze training).')
    parser.add_argument('--ckpt-specs', type=str, nargs='*', default=None,
                        help='label:ckpt_path:arch  (overrides built-in checkpoint list)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = DEFAULT_CHECKPOINTS
    if args.ckpt_specs:
        checkpoints = [tuple(s.split(':')) for s in args.ckpt_specs]

    print("=" * 60, flush=True)
    print("  Full Epsilon Sweep Evaluation", flush=True)
    print(f"  Device:     {device}", flush=True)
    print(f"  n_samples:  {args.n_samples}", flush=True)
    print(f"  batch_size: {args.batch_size}", flush=True)
    print(f"  PGD steps:  {args.pgd_steps}", flush=True)
    print(f"  Epsilons:   {args.eps_list}", flush=True)
    print("=" * 60, flush=True)

    # ── Print all per-channel conversions BEFORE any GPU work ─────────────────
    _mode = ("NORM-space (matched to Finding-17 baseline)" if args.eps_norm_space
             else "PIXEL-space [0,1] (per-channel /std conversion)")
    print(f"\n[CHANNEL-WISE EPSILON VERIFICATION — {_mode}]", flush=True)
    if args.eps_norm_space and args.eps_list == DEFAULT_EPS_LIST:
        print("  NOTE: --eps-norm-space with the DEFAULT eps list reinterpreted those"
              " pixel values (0.0313, 0.0625, ...) as norm-space bounds, which is ~4x WEAKER."
              " Pass --eps-list explicitly (e.g. 0.0 0.031 0.062 0.094) for the matched grid.",
              flush=True)
    for eps in args.eps_list:
        eps_norm = resolve_eps_norm(eps, device, norm_space=args.eps_norm_space)
        log_eps(eps, eps_norm, norm_space=args.eps_norm_space)
    print("", flush=True)

    # Data stays on CPU; batches are moved to GPU inside attack/inference loops
    x_norm_cpu, y_test_cpu = load_test_samples(args.n_samples)

    csv_path = os.path.join(args.output_dir, 'epsilon_sweep_results.csv')
    fieldnames = ['ckpt_label', 'eps_pixel',
                  'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
                  'acc_pct', 'macro_dprime']
    rows = []

    for label, ckpt_path, arch in checkpoints:
        print(f"\n{'='*60}", flush=True)
        print(f"  Checkpoint : {label}", flush=True)
        print(f"  Path       : {ckpt_path}", flush=True)
        print(f"{'='*60}", flush=True)

        model = load_model(arch, ckpt_path, device, freeze_gaze=args.freeze_gaze)

        for eps in args.eps_list:
            eps_norm = resolve_eps_norm(eps, device, norm_space=args.eps_norm_space)
            log_eps(eps, eps_norm, prefix=f"[{label}] ", norm_space=args.eps_norm_space)

            t0 = time.time()
            if eps == 0.0:
                x_adv_cpu = x_norm_cpu
            else:
                x_adv_cpu = run_pgd_batched(
                    model, x_norm_cpu, y_test_cpu,
                    eps=eps,
                    steps=args.pgd_steps,
                    device=device,
                    batch_size=args.batch_size,
                    norm_space=args.eps_norm_space,
                )

            # Batched accuracy
            correct = 0
            for start in range(0, x_adv_cpu.size(0), args.batch_size):
                xb = x_adv_cpu[start:start + args.batch_size].to(device)
                yb = y_test_cpu[start:start + args.batch_size].to(device)
                with torch.no_grad():
                    logits = model(xb)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    correct += logits.argmax(dim=1).eq(yb).sum().item()
                del xb, yb
            acc = 100.0 * correct / x_adv_cpu.size(0)

            dp = compute_dprime_batched(model, x_adv_cpu, y_test_cpu,
                                        device, args.batch_size)
            elapsed = time.time() - t0

            r = eps_norm[0, 0, 0, 0].item()
            g = eps_norm[0, 1, 0, 0].item()
            b = eps_norm[0, 2, 0, 0].item()

            print(f"  -> Acc: {acc:6.2f}%  |  d': {dp:.4f}  ({elapsed:.1f}s)", flush=True)

            rows.append({
                'ckpt_label':   label,
                'eps_pixel':    round(eps, 4),
                'eps_norm_R':   round(r, 4),
                'eps_norm_G':   round(g, 4),
                'eps_norm_B':   round(b, 4),
                'acc_pct':      round(acc, 2),
                'macro_dprime': round(dp, 4),
            })

        del model
        torch.cuda.empty_cache()

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Final table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  RESULTS TABLE", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Checkpoint':<30} {'eps_px':>7} {'Acc%':>7} {'d_prime':>8}", flush=True)
    print("-" * 60, flush=True)
    for row in rows:
        print(f"{row['ckpt_label']:<30} {row['eps_pixel']:>7.4f} "
              f"{row['acc_pct']:>7.2f} {row['macro_dprime']:>8.4f}", flush=True)

    print(f"\n  CSV: {csv_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
