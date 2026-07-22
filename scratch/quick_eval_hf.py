#!/usr/bin/env python3
"""
RHAN-v11 Diagnostic Evaluation — HuggingFace Checkpoints
=========================================================
Usage examples:
  # Default quick eval (200 samples, PGD-10):
  python3 scratch/quick_eval_hf.py

  # Your requested diagnostic (500 samples, PGD-50 + PGD-100 spot check):
  python3 scratch/quick_eval_hf.py --pgd-steps 50 --pgd-steps-spot 100 --n-samples 500

  # Only evaluate the best checkpoint, skip rolling:
  python3 scratch/quick_eval_hf.py --skip-rolling
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../phase1_training')))

from phase1_training.model_rhan_v11 import RHANv11
from huggingface_hub import hf_hub_download

# ─── STL-10 normalisation constants ───────────────────────────────────────────
MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
STD  = torch.tensor([0.2603, 0.2566, 0.2713]).view(1, 3, 1, 1)
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

# ─── HF helpers ───────────────────────────────────────────────────────────────
def _hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), '../.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
    return token


def load_hf_checkpoint(repo_id, filename, target_path):
    print(f"--> [{filename}] Checking local cache...", flush=True)
    os.makedirs(os.path.dirname(target_path) or '.', exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 50 * 1024 * 1024:
        size_mb = os.path.getsize(target_path) / 1e6
        print(f"    ✓ Already cached locally ({size_mb:.0f} MB): {target_path}", flush=True)
        return True
    print(f"    Downloading from {repo_id}...", flush=True)
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename,
                                 repo_type='dataset', token=_hf_token())
        import shutil
        shutil.copy2(cached, target_path)
        print(f"    ✓ Saved to {target_path}", flush=True)
        return True
    except Exception as e:
        print(f"    ✗ Failed: {e}", flush=True)
        return False


# ─── Dataset loading ───────────────────────────────────────────────────────────
def load_test_samples(n_samples=500):
    print(f"--> Streaming {n_samples} test samples from mteb/stl10...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("mteb/stl10", split="test").shuffle(seed=42).select(range(n_samples))
    images, labels = [], []
    for item in ds:
        img = item['image'].convert('RGB').resize((96, 96))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - MEAN.squeeze(0)) / STD.squeeze(0)
        images.append(t)
        labels.append(item['label'])
    x = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)
    # Print class distribution
    print(f"    ✓ {x.shape[0]} samples loaded.", flush=True)
    dist = {STL10_CLASSES[c]: (y == c).sum().item() for c in range(10)}
    print(f"    Class dist: { {k: v for k, v in dist.items()} }", flush=True)
    return x, y


# ─── PGD attack ───────────────────────────────────────────────────────────────
def run_pgd(model, x, y, eps, steps, alpha=None):
    """
    PGD-L∞ with KL-divergence loss (TRADES-style).
    alpha defaults to eps/4 (standard TRADES recommendation).
    """
    if eps == 0:
        return x.clone().detach()
    if alpha is None:
        alpha = eps / 4.0          # fixed step size, not eps/steps

    device = x.device
    stl_min = (-MEAN / STD).to(device)
    stl_max = ((1.0 - MEAN) / STD).to(device)

    model.eval()
    with torch.no_grad():
        logits_c = model(x)
    probs_c = F.softmax(logits_c.float(), dim=1)

    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)

    for step in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            logits_a = model(x_adv)
            loss = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                            probs_c, reduction='batchmean')
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x + torch.clamp(x_adv - x, -eps, eps),
                            stl_min, stl_max).detach()
    return x_adv


def pgd_accuracy(model, x_test, y_test, eps, steps, batch_size, label):
    N = x_test.size(0)
    n_batches = (N + batch_size - 1) // batch_size
    correct = 0
    t0 = time.time()
    print(f"  --> {label}: {steps} steps, ε={eps:.3f}, α={eps/4:.4f} ...", flush=True)
    for i in range(0, N, batch_size):
        xb = x_test[i:i+batch_size].to(x_test.device if x_test.is_cuda else 'cpu')
        yb = y_test[i:i+batch_size].to(xb.device)
        x_adv = run_pgd(model, xb, yb, eps=eps, steps=steps)
        with torch.no_grad():
            correct += model(x_adv).argmax(1).eq(yb).sum().item()
        batch_idx = i // batch_size + 1
        elapsed = time.time() - t0
        eta = elapsed / batch_idx * (n_batches - batch_idx)
        print(f"      batch {batch_idx:>3}/{n_batches}  |  "
              f"running acc={100*correct/(i+xb.size(0)):.1f}%  |  "
              f"ETA {eta:.0f}s", flush=True)
    acc = 100.0 * correct / N
    print(f"  ✓ {label}: {acc:.2f}% ({correct}/{N})  [{time.time()-t0:.0f}s]", flush=True)
    return acc


# ─── Main evaluation loop ─────────────────────────────────────────────────────
def evaluate(ckpt_name, ckpt_path, x_test, y_test, device, args):
    print(f"\n{'='*60}", flush=True)
    print(f"  Checkpoint: {ckpt_name}", flush=True)
    print(f"{'='*60}", flush=True)

    model = RHANv11().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ('model', 'model_state_dict', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"State dict  missing={len(missing)}  unexpected={len(unexpected)}", flush=True)
    model.eval()

    N = x_test.size(0)
    BS = args.batch_size

    # ── 1. Clean accuracy + diagnostics ───────────────────────────────────────
    correct_clean = 0
    recon_mses, alphas = [], []
    prec_accum = {c: [] for c in range(10)}   # per-class Π_D tensors

    with torch.no_grad():
        for i in range(0, N, BS):
            xb = x_test[i:i+BS].to(device)
            yb = y_test[i:i+BS].to(device)
            logits, traj = model(xb, return_trajectory=True)
            correct_clean += logits.argmax(1).eq(yb).sum().item()
            if traj.get('recon_errors'):
                recon_mses.append(traj['recon_errors'][-1].item())
            if traj.get('gate_alphas'):
                alphas.append(traj['gate_alphas'][-1].mean().item())
            if traj.get('precisions'):
                prec = traj['precisions'][-1]          # (B,)
                for c in range(10):
                    mask = (yb == c)
                    if mask.any():
                        prec_accum[c].append(prec[mask].cpu())

    clean_acc = 100.0 * correct_clean / N
    print(f"\n  Clean Accuracy     : {clean_acc:.2f}%  ({correct_clean}/{N})", flush=True)
    print(f"  Recon MSE          : {np.mean(recon_mses):.4f}" if recon_mses else "  Recon MSE          : n/a", flush=True)
    print(f"  Foveal Gate α      : {np.mean(alphas):.4f}" if alphas else "  Foveal Gate α      : n/a", flush=True)

    # ── 2. Full Π_D per-class table ───────────────────────────────────────────
    print(f"\n  Π_D per class (clean inference):", flush=True)
    print(f"  {'Class':<12}  {'Π_D':>6}  {'N':>5}", flush=True)
    print(f"  {'-'*30}", flush=True)
    for c in range(10):
        if prec_accum[c]:
            vals = torch.cat(prec_accum[c])
            mark = " ◄" if STL10_CLASSES[c] in ('car', 'truck') else ""
            print(f"  {STL10_CLASSES[c]:<12}  {vals.mean():.4f}  {len(vals):>5}{mark}", flush=True)
        else:
            print(f"  {STL10_CLASSES[c]:<12}  {'n/a':>6}  {'0':>5}", flush=True)

    # ── 3. PGD-N (primary steps) at both eps ──────────────────────────────────
    x_test_dev = x_test.to(device)
    for eps_val in [0.031, 0.062]:
        pgd_accuracy(model, x_test_dev, y_test.to(device),
                     eps=eps_val, steps=args.pgd_steps, batch_size=BS,
                     label=f"PGD-{args.pgd_steps} (ε={eps_val:.3f})")

    # ── 4. Spot-check at higher step count (eps=0.031 only) ───────────────────
    if args.pgd_steps_spot and args.pgd_steps_spot > args.pgd_steps:
        pgd_accuracy(model, x_test_dev, y_test.to(device),
                     eps=0.031, steps=args.pgd_steps_spot, batch_size=BS,
                     label=f"PGD-{args.pgd_steps_spot} spot-check (ε=0.031)")

    print(flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RHAN-v11 Fast HF Evaluation")
    p.add_argument('--n-samples',      type=int, default=200,
                   help='Test samples to stream from mteb/stl10 (default: 200, max: 8000)')
    p.add_argument('--pgd-steps',      type=int, default=10,
                   help='PGD iteration count for primary eval (default: 10)')
    p.add_argument('--pgd-steps-spot', type=int, default=0,
                   help='Optional spot-check PGD step count at eps=0.031 (0 = skip)')
    p.add_argument('--batch-size',     type=int, default=16,
                   help='Eval batch size (default: 16)')
    p.add_argument('--skip-rolling',   action='store_true',
                   help='Skip the rolling (latest epoch) checkpoint')
    p.add_argument('--skip-best',      action='store_true',
                   help='Skip the best checkpoint')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RHAN-v11 Diagnostic Eval | device={device} | "
          f"n={args.n_samples} | PGD-{args.pgd_steps}"
          + (f"+{args.pgd_steps_spot}" if args.pgd_steps_spot else ""), flush=True)

    rolling_path = "checkpoints/rhan_stl10_v11_rolling.pth"
    best_path    = "checkpoints/rhan_stl10_v11_best.pth"

    has_best    = (not args.skip_best and
                   load_hf_checkpoint("FerrariKazu/rhan-checkpoints",
                                      "rhan_stl10_v11_best.pth", best_path))
    has_rolling = (not args.skip_rolling and
                   load_hf_checkpoint("FerrariKazu/rhan-checkpoints-rolling",
                                      "rhan_stl10_v11_rolling.pth", rolling_path))

    if not has_best and not has_rolling:
        print("No checkpoints available. Exiting.")
        return

    x_test, y_test = load_test_samples(args.n_samples)

    if has_best:
        evaluate("rhan_stl10_v11_best.pth  [Best Peak Accuracy]",
                 best_path, x_test, y_test, device, args)

    if has_rolling:
        evaluate("rhan_stl10_v11_rolling.pth  [Latest Epoch]",
                 rolling_path, x_test, y_test, device, args)


if __name__ == '__main__':
    main()
