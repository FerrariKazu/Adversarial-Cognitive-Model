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

    # SEED-AVERAGED PROTOCOL (recommended; replaces single-seed numbers):
    #   n=300 per seed, 3 seeds, priority eps grid. Each seed draws a FRESH sample
    #   subset AND fresh PGD init. Crossover is real only if (RHAN - baseline) at
    #   eps=0.094 > 2 x sqrt(std_RHAN^2 + std_baseline^2). Per-checkpoint gaze:
    #   the optional 4th ckpt-spec field ('freeze') applies --freeze-gaze to that
    #   checkpoint only (Run B), so all four checkpoints run in ONE batch.
    python3 phase2_attacks/eval_full_epsilon_sweep.py \
        --n-samples 300 --seeds 41 42 43 --pgd-steps 50 --batch-size 32 \
        --eps-norm-space --eps-list 0.0 0.031 0.094 \
        --baseline-label trades_large_baseline \
        --ckpt-specs \
          trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
          null_ablation_v11:checkpoints/rhan_stl10_v11_rolling.pth:v11 \
          run_a_norecon:checkpoints/rhan_v11_isolation_norecon_best.pth:v11 \
          run_b_fixedgaze:checkpoints/rhan_v11_isolation_fixedgaze_best.pth:v11:freeze

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


def _get_hf_token():
    """Resolve HF_TOKEN from env, Colab userdata, or Kaggle secrets."""
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
        if token:
            return token
    except Exception:
        pass
    return None


def _hf_upload_csv(csv_path, repo_id, path_in_repo, hf_token):
    """Upload a local CSV to HF dataset repo (best-effort, non-fatal)."""
    if not os.path.exists(csv_path) or not hf_token:
        return
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=hf_token)
        create_repo(repo_id=repo_id, repo_type='dataset', private=False,
                    exist_ok=True, token=hf_token)
        api.upload_file(
            path_or_fileobj=csv_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type='dataset',
            token=hf_token,
        )
        print(f"  [hf-sync] uploaded {os.path.basename(csv_path)} -> {repo_id}/{path_in_repo}",
              flush=True)
    except Exception as e:
        print(f"  [hf-sync] WARNING: upload failed: {e}", flush=True)


def _hf_download_csv(csv_path, repo_id, path_in_repo, hf_token):
    """Download a CSV from HF if local copy is missing. Returns True if downloaded."""
    if os.path.exists(csv_path) or not hf_token:
        return False
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        # Ensure local directory exists before writing
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        # Download to HF cache (no local_dir avoids nested-dir bug with
        # filename='subdir/file.csv' + local_dir='subdir/').
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=path_in_repo,
            token=hf_token,
        )
        # Copy (not move) from cache to the expected local path.
        shutil.copy2(downloaded_path, csv_path)
        print(f"  [hf-sync] downloaded {path_in_repo} from {repo_id}", flush=True)
        return True
    except Exception as e:
        print(f"  [hf-sync] no CSV on HF ({e}) — starting fresh", flush=True)
        return False


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
    label reflects that instead of misnaming it a pixel value, and an explicit
    confirmation line is printed stating the bound was applied directly.
    """
    key = "eps_norm" if norm_space else "eps_pixel"
    r = eps_norm[0, 0, 0, 0].item()
    g = eps_norm[0, 1, 0, 0].item()
    b = eps_norm[0, 2, 0, 0].item()
    print(f"  [EPS] {prefix}{key}={eps_listed:.4f} -> "
          f"eps_norm=[R:{r:.4f}, G:{g:.4f}, B:{b:.4f}]", flush=True)
    if norm_space:
        print(f"  ✓ CONFIRM: norm-space eps={eps_listed:.3f} applied directly, "
              f"NOT converted from pixel (same bound on all channels)", flush=True)


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

    Returns (adv_cpu, delta_max): adv_cpu is the adversarial batch on CPU, and
    delta_max is the achieved max |delta| per channel in normalized units (used
    to confirm the clamp bound was applied directly at eps).
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

    adv_cpu = torch.cat(adv_chunks, dim=0)  # on CPU
    # Achieved perturbation bound check (normalized units, per channel).
    # Must be <= the applied clamp bound on every channel.
    delta_max = (adv_cpu - x_norm_cpu).abs().amax(dim=(0, 2, 3))
    return adv_cpu, delta_max


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
    if arch == "v12":
        from phase1_training.model_rhan_v12 import RHANv12
        model = RHANv12().to(device)
    elif arch == "v11":
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

def set_seed(seed):
    """Seed numpy + torch (CPU + CUDA) so each protocol seed pins BOTH the
    per-seed sample subset (load_test_samples seed) and the PGD init noise."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_std(vals):
    """(mean, unbiased sample std, ddof=1) over the given values."""
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def crossover_report(agg, eps_target, baseline_label, labels):
    """RHAN - baseline gap at eps_target, with the '> 2 x combined std' verdict.

    The crossover is only real if (RHAN mean - baseline mean) > 2 x
    sqrt(std_RHAN^2 + std_baseline^2) — a deliberately conservative criterion
    given the ~2-4 pp single-draw variance observed between identical runs.
    """
    out = []
    bkey = (baseline_label, eps_target)
    if bkey not in agg:
        return out
    if len(agg[bkey]['accs']) < 2:
        return [f"  eps={eps_target:>6}: baseline has <2 seeds ({len(agg[bkey]['accs'])}); "
                "crossover check skipped (need >=2 seeds per model)"]
    bm, bs = _mean_std(agg[bkey]['accs'])
    for lab in labels:
        if lab == baseline_label:
            continue
        key = (lab, eps_target)
        if key not in agg:
            continue
        if len(agg[key]['accs']) < 2:
            out.append(f"  eps={eps_target:>6}: {lab}: <2 seeds; crossover check skipped")
            continue
        rm, rs = _mean_std(agg[key]['accs'])
        diff = rm - bm
        combined = np.sqrt(rs ** 2 + bs ** 2)
        verdict = ("CROSSOVER REAL" if diff > 2.0 * combined
                   else ("positive but NOT significant" if diff > 0
                         else "at or below baseline"))
        out.append(
            f"  eps={eps_target:>6}: {lab:<22} {rm:6.2f}+-{rs:4.2f} vs "
            f"{baseline_label} {bm:6.2f}+-{bs:4.2f} | d={diff:+5.2f} pp | "
            f"2*sig_comb={2.0*combined:5.2f} | {verdict}"
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples',  type=int,   default=500)
    parser.add_argument('--seed',       type=int,   default=42,
                        help='Random seed for reproducible PGD init noise (default 42)')
    parser.add_argument('--seeds',      type=int,   nargs='+', default=None,
                        help='3-seed averaging protocol: each seed draws a FRESH sample '
                             'subset (load_test_samples seed) AND fresh PGD init noise. '
                             'Default: [--seed].')
    parser.add_argument('--baseline-label', type=str, default='trades_large_baseline',
                        help='Checkpoint label treated as baseline for the epsilon crossover '
                             'significance check (default trades_large_baseline).')
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
    parser.add_argument('--resume', action='store_true',
                        help='Skip (label, seed, eps) cells already present in the output dir '
                             'per-seed CSV: carry their rows forward into the fresh CSV and the '
                             'aggregation instead of re-running PGD for them (idempotent '
                             're-runs after a partial/timeout session).')
    parser.add_argument('--freeze-gaze', action='store_true',
                        help='GLOBAL freeze-gaze (all checkpoints). For per-checkpoint control '
                             'use the optional 4th field in --ckpt-specs: label:path:arch:freeze')
    parser.add_argument('--hf-sync', action='store_true',
                        help='After each seed, upload the per-seed CSV to a HuggingFace '
                             'dataset repo so --resume survives Colab/Kaggle restarts. '
                             'Requires HF_TOKEN in the environment.')
    parser.add_argument('--hf-dataset-repo', type=str, default=None,
                        help='HF dataset repo for eval sync (default: auto-detect from '
                             'HF_TOKEN username: <user>/rhan-eval-sweep)')
    parser.add_argument('--hf-eval-subdir', type=str, default='',
                        help='Subdirectory inside the HF repo to store the per-seed CSV '
                             '(e.g. sweep_stage3_d_ais_hpc_pgd100). Empty = root.')
    parser.add_argument('--ckpt-specs', type=str, nargs='*', default=None,
                        help='label:ckpt_path:arch[:freeze]  (overrides built-in list; the '
                             'optional 4th field freezes gaze for that checkpoint only)')
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds if args.seeds else [args.seed]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = [(l, p, a, False) for l, p, a in DEFAULT_CHECKPOINTS]
    if args.ckpt_specs:
        parsed = []
        for s in args.ckpt_specs:
            parts = s.split(':')
            if len(parts) < 3:
                raise SystemExit(f"Bad --ckpt-specs entry: {s!r} (want label:path:arch[:freeze])")
            label, ckpt_path, arch = parts[0], parts[1], parts[2]
            freeze = len(parts) > 3 and parts[3].strip().lower() in ('freeze', 'freeze-gaze', '1', 'true')
            parsed.append((label, ckpt_path, arch, freeze))
        checkpoints = parsed

    print("=" * 60, flush=True)
    print("  Full Epsilon Sweep Evaluation - SEED-AVERAGED PROTOCOL", flush=True)
    print(f"  Device:     {device}", flush=True)
    print(f"  n_samples:  {args.n_samples} per seed", flush=True)
    print(f"  seeds:      {seeds}  ({len(seeds)}-seed mean +- std)", flush=True)
    print(f"  batch_size: {args.batch_size}", flush=True)
    print(f"  PGD steps:  {args.pgd_steps}", flush=True)
    print(f"  Epsilons:   {args.eps_list}", flush=True)
    print(f"  Baseline:   {args.baseline_label}", flush=True)
    print("=" * 60, flush=True)

    # ── Print all per-channel conversions BEFORE any GPU work ─────────────────
    _mode = ("NORM-space (matched to Finding-17 baseline)" if args.eps_norm_space
             else "PIXEL-space [0,1] (per-channel /std conversion)")
    print(f"\n[CHANNEL-WISE EPSILON VERIFICATION — {_mode}]", flush=True)
    if args.eps_norm_space:
        print("  FINDING-17 MATCHED CONVENTION: eps is applied DIRECTLY to the normalized"
              " inputs — the exact grid/convention used for the Finding-17 baseline table"
              " (quick_eval_hf / eval_rhan_v11: eps=0.031/0.062/0.094 in norm space,", flush=True)
        print("  e.g. TRADES Large baseline 48.0/40.3/33.7). NO /std pixel conversion.",
              flush=True)
    if not args.eps_norm_space:
        print("  WARNING: PIXEL-SPACE MODE — listed eps is a [0,1] pixel bound divided by"
              " per-channel std, so these attacks are ~3.84x STRONGER than the Finding-17"
              " norm-space table (eps=0.031/0.062/0.094). For the MATCHED comparison re-run"
              " with:  --eps-norm-space --eps-list 0.0 0.031 0.062 0.094", flush=True)
    if args.eps_norm_space and args.eps_list == DEFAULT_EPS_LIST:
        print("  NOTE: --eps-norm-space with the DEFAULT eps list reinterpreted those"
              " pixel values (0.0313, 0.0625, ...) as norm-space bounds, which is ~4x WEAKER."
              " Pass --eps-list explicitly (e.g. 0.0 0.031 0.062 0.094) for the matched grid.",
              flush=True)
    for eps in args.eps_list:
        eps_norm = resolve_eps_norm(eps, device, norm_space=args.eps_norm_space)
        log_eps(eps, eps_norm, norm_space=args.eps_norm_space)
    print("", flush=True)

    # Load per-seed datasets ONCE (fresh subset per seed), CPU-side.
    datasets = {}
    for seed in seeds:
        set_seed(seed)
        datasets[seed] = load_test_samples(args.n_samples, seed=seed)

    csv_per_seed = os.path.join(args.output_dir, 'epsilon_sweep_per_seed.csv')
    csv_agg = os.path.join(args.output_dir, 'epsilon_sweep_results.csv')
    seed_fields = ['ckpt_label', 'seed', 'eps_pixel',
                   'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
                   'acc_pct', 'macro_dprime']

    # ── HF sync setup (--hf-sync) ───────────────────────────────────────────────
    hf_token = _get_hf_token() if args.hf_sync else None
    hf_repo = args.hf_dataset_repo
    hf_subdir = args.hf_eval_subdir
    if args.hf_sync and not hf_repo:
        # auto-detect from token
        try:
            from huggingface_hub import HfApi
            _api = HfApi(token=hf_token)
            _username = _api.whoami()['name']
            hf_repo = f"{_username}/rhan-eval-sweep"
        except Exception:
            hf_repo = 'FerrariKazu/rhan-eval-sweep'
    if args.hf_sync:
        hf_csv_path = os.path.join(hf_subdir, 'epsilon_sweep_per_seed.csv') if hf_subdir else 'epsilon_sweep_per_seed.csv'
        print(f"  [hf-sync] will upload per-seed CSV to {hf_repo}/{hf_csv_path}", flush=True)

    agg = {}   # (label, eps) -> {'accs': [...], 'dps': [...]} across seeds

    # --resume: a cell key is (label, seed, round(eps,4)). Rows already in the
    # output dir per-seed CSV are carried forward — skipped in the compute loop,
    # written into the fresh CSV, and folded into the aggregation so the
    # aggregated CSV + crossover verdicts still cover previously-computed cells
    # (an idempotent re-run after a partial/timeout session never re-pays PGD).
    done = {}
    # --resume + --hf-sync: if local CSV is missing (Colab restart), pull from HF
    if args.resume and not os.path.exists(csv_per_seed) and args.hf_sync:
        _hf_download_csv(csv_per_seed, hf_repo, hf_csv_path, hf_token)
    if args.resume and os.path.exists(csv_per_seed):
        with open(csv_per_seed, newline='') as f:
            for r in csv.DictReader(f):
                try:
                    key = (r['ckpt_label'], int(r['seed']),
                           round(float(r['eps_pixel']), 4))
                except (KeyError, ValueError):
                    continue
                done[key] = r
        for (label, _s, eps), r in done.items():
            agg.setdefault((label, eps), {'accs': [], 'dps': []})
            agg[(label, eps)]['accs'].append(float(r['acc_pct']))
            agg[(label, eps)]['dps'].append(float(r['macro_dprime']))
        if done:
            print(f"  [resume] {len(done)} previously-evaluated cell(s) found "
                  f"in {os.path.basename(csv_per_seed)} — skipping them and "
                  f"carrying their rows forward.", flush=True)

    # Per-seed CSV is written incrementally (flushed after every seed) so a
    # long run that hits a Kaggle session timeout still keeps partial data.
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_per_seed, 'w', newline='') as f_seed:
        writer_seed = csv.DictWriter(f_seed, fieldnames=seed_fields)
        writer_seed.writeheader()
        for r in done.values():
            writer_seed.writerow(r)
        f_seed.flush()

        for label, ckpt_path, arch, freeze_this in checkpoints:
            cell_keys = [(label, seed, round(eps, 4))
                         for seed in seeds for eps in args.eps_list]
            if args.resume and all(k in done for k in cell_keys):
                print(f"\n  [resume] {label}: all {len(cell_keys)} cell(s) "
                      f"already evaluated — skipping model load.", flush=True)
                continue
            print(f"\n{'='*60}", flush=True)
            print(f"  Checkpoint : {label}", flush=True)
            print(f"  Path       : {ckpt_path}", flush=True)
            print(f"  Freeze gaze: {freeze_this or args.freeze_gaze}", flush=True)
            print(f"{'='*60}", flush=True)

            model = load_model(arch, ckpt_path, device,
                               freeze_gaze=freeze_this or args.freeze_gaze)

            for seed in seeds:
                x_norm_cpu, y_test_cpu = datasets[seed]
                for eps in args.eps_list:
                    if args.resume and (label, seed, round(eps, 4)) in done:
                        print(f"  [resume] SKIP {label} seed={seed} eps={eps} "
                              f"(already evaluated)", flush=True)
                        continue
                    eps_norm = resolve_eps_norm(eps, device, norm_space=args.eps_norm_space)
                    log_eps(eps, eps_norm, prefix=f"[{label} seed={seed}] ",
                            norm_space=args.eps_norm_space)

                    t0 = time.time()
                    delta_max = None
                    if eps == 0.0:
                        x_adv_cpu = x_norm_cpu
                    else:
                        x_adv_cpu, delta_max = run_pgd_batched(
                            model, x_norm_cpu, y_test_cpu,
                            eps=eps,
                            steps=args.pgd_steps,
                            device=device,
                            batch_size=args.batch_size,
                            norm_space=args.eps_norm_space,
                        )
                        d0, d1, d2 = (delta_max[i].item() for i in range(3))
                        e0, e1, e2 = (eps_norm[0, i, 0, 0].item() for i in range(3))
                        ok = d0 <= e0 + 1e-4 and d1 <= e1 + 1e-4 and d2 <= e2 + 1e-4
                        mode = ("applied DIRECTLY in norm space" if args.eps_norm_space
                                else f"converted via /std (eps_norm per channel)")
                        print(f"  [BOUND CHECK] max |δ|_norm = [R:{d0:.4f}, G:{d1:.4f}, "
                              f"B:{d2:.4f}] <= per-channel bound [{e0:.4f}, {e1:.4f}, {e2:.4f}] "
                              f"({mode}) → {'OK' if ok else 'OVER-BUDGET!'}", flush=True)

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

                    writer_seed.writerow({
                        'ckpt_label':   label,
                        'seed':         seed,
                        'eps_pixel':    round(eps, 4),
                        'eps_norm_R':   round(r, 4),
                        'eps_norm_G':   round(g, 4),
                        'eps_norm_B':   round(b, 4),
                        'acc_pct':      round(acc, 2),
                        'macro_dprime': round(dp, 4),
                    })
                    agg.setdefault((label, eps), {'accs': [], 'dps': []})
                    agg[(label, eps)]['accs'].append(acc)
                    agg[(label, eps)]['dps'].append(dp)

                f_seed.flush()   # partial-run survival after each seed
                # --hf-sync: upload CSV to HF so --resume survives Colab restarts
                if args.hf_sync:
                    _hf_upload_csv(csv_per_seed, hf_repo, hf_csv_path, hf_token)

            del model
            torch.cuda.empty_cache()

    # ── Write aggregated CSV (mean ± std over seeds) ───────────────────────────
    agg_fields = ['ckpt_label', 'eps_pixel',
                  'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
                  'acc_mean', 'acc_std', 'macro_dprime_mean', 'macro_dprime_std', 'n_seeds']
    agg_rows = []
    for (label, eps), rec in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        am, astd = _mean_std(rec['accs'])
        dm, dstd = _mean_std(rec['dps'])
        eps_norm = resolve_eps_norm(eps, device, norm_space=args.eps_norm_space)
        r, g, b = (eps_norm[0, i, 0, 0].item() for i in range(3))
        agg_rows.append({
            'ckpt_label': label, 'eps_pixel': round(eps, 4),
            'eps_norm_R': round(r, 4), 'eps_norm_G': round(g, 4), 'eps_norm_B': round(b, 4),
            'acc_mean': round(am, 2), 'acc_std': round(astd, 2),
            'macro_dprime_mean': round(dm, 4), 'macro_dprime_std': round(dstd, 4),
            'n_seeds': len(rec['accs']),
        })
    with open(csv_agg, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)

    # ── Final table (mean ± std) ───────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("  RESULTS TABLE - mean +- std over seeds", flush=True)
    print("=" * 72, flush=True)
    print(f"{'Checkpoint':<30} {'eps':>6} {'Acc% (mean+-std)':>18} {'d-prime (mean+-std)':>22}", flush=True)
    print("-" * 72, flush=True)
    for row in agg_rows:
        print(f"{row['ckpt_label']:<30} {row['eps_pixel']:>6.3f} "
              f"{row['acc_mean']:>7.2f}+-{row['acc_std']:<5.2f} "
              f"{row['macro_dprime_mean']:>8.4f}+-{row['macro_dprime_std']:<6.4f}", flush=True)

    # ── Crossover significance ─────────────────────────────────────────────────
    print("\n  CROSSOVER SIGNIFICANCE - RHAN vs baseline (criterion: d > 2*sig_combined)", flush=True)
    labels = [l for l, _, _, _ in checkpoints]
    for eps in args.eps_list:
        if eps == 0.0:
            continue
        for line in crossover_report(agg, eps, args.baseline_label, labels):
            print(line, flush=True)

    print(f"\n  Per-seed CSV: {csv_per_seed}", flush=True)
    print(f"  Aggregated  : {csv_agg}", flush=True)
    print("=" * 72, flush=True)
    print("  NOTE: std is the unbiased sample std (ddof=1) over seeds. With 3 seeds it is", flush=True)
    print("  noisy; the '> 2*sig_combined' crossover criterion is deliberately conservative.", flush=True)


if __name__ == '__main__':
    main()
