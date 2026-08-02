#!/usr/bin/env python3
"""
Colab Notebook Pipeline: 3-Seed Averaging Protocol (one batch)
================================================================
Twin of cloud_setup/kaggle_v11_3seed_protocol.py — same protocol, Colab setup.
Replaces every partial single-draw sweep number from the isolation runs.

  Protocol (eval_full_epsilon_sweep.py, --seeds):
    - n = 300 samples per seed (fresh sample subset + fresh PGD init per seed)
    - 3 seeds: 41, 42, 43  ->  report mean ± std (sample std, ddof=1)
    - PGD-50, NORM-space eps grid (Finding-17 matched convention, NO /std conv)
    - Priority eps: 0.000, 0.031, 0.094  (0.062 optional via INCLUDE_062=1)
    - Crossover is real only if:
        (RHAN mean - baseline mean) at eps=0.094 > 2 x sqrt(std_RHAN^2 + std_baseline^2)

  Checkpoints (all on HF; eval-only, no training):
    1. trades_large_baseline : rhan_stl10_large_pseudolabel_best.pth  (arch=large)
    2. null_ablation_v11     : rhan_stl10_v11_rolling.pth             (arch=v11)
    3. run_a_norecon         : rhan_v11_isolation_norecon_best.pth    (arch=v11)
    4. run_b_fixedgaze       : rhan_v11_isolation_fixedgaze_best.pth  (arch=v11, :freeze)

  Determinism audit (confirmed 2026-08-02): eval_full_epsilon_sweep.py sets NONE
  of torch.use_deterministic_algorithms(True), cudnn.deterministic, or
  CUBLAS_WORKSPACE_CONFIG — residual GPU nondeterminism (F.grid_sample /
  affine_grid in foveal_sample, attention softmax reductions) is expected.
  Per instruction we do NOT chase bit-for-bit determinism; we quantify it
  with the 3-seed mean ± std protocol instead.

Usage: Copy cells into Colab (T4 or better, 12h+ session). Set HF_TOKEN in
Colab Secrets (key icon in sidebar). Estimated runtime: ~7-9 h on a T4.
"""

# %% [markdown]
# # RHAN-v11 3-Seed Averaging Protocol (baseline vs null-ablation vs Run A vs Run B)

# %% [markdown]
# ## Step 1: Install Dependencies

# %%
import os, sys, subprocess

def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    rc = process.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
run("pip install --quiet huggingface_hub datasets Pillow scipy")

# Optionally widen the grid to include eps=0.062 (adds ~40% runtime).
INCLUDE_062 = os.environ.get("INCLUDE_062", "0") == "1"
EPS_LIST = "0.0 0.031 0.062 0.094" if INCLUDE_062 else "0.0 0.031 0.094"
print(f"Epsilon grid: {EPS_LIST}")

# %% [markdown]
# ## Step 2: Clone and Sync Repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'

if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))

# %% [markdown]
# ## Step 3: Set HF_TOKEN and Environment

# %%
import torch
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN not found. Set it in Colab Secrets (key icon in sidebar).")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print(f"✓ HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Step 4: Resolve the four checkpoints (local-first, HF fallback)

# %%
import os as _os
from huggingface_hub import hf_hub_download as _hf_dl

def resolve(filename, repo):
    """Return a local path to filename, local-first with HF fallback."""
    path = f"checkpoints/{filename}"
    if _os.path.exists(path):
        print(f"  Present locally: {path}", flush=True)
        return path
    print(f"  Downloading {filename} from {repo}...", flush=True)
    return _hf_dl(repo_id=repo, repo_type="dataset",
                  filename=filename, local_dir="checkpoints")

_bsl  = resolve("rhan_stl10_large_pseudolabel_best.pth", "FerrariKazu/rhan-checkpoints")
_null = resolve("rhan_stl10_v11_rolling.pth",            "FerrariKazu/rhan-checkpoints-rolling")
_runA = resolve("rhan_v11_isolation_norecon_best.pth",    "FerrariKazu/rhan-checkpoints")
_runB = resolve("rhan_v11_isolation_fixedgaze_best.pth",  "FerrariKazu/rhan-checkpoints")

print("\nCheckpoint map:")
for name, p in [("baseline", _bsl), ("null_ablation_v11", _null),
                ("run_a_norecon", _runA), ("run_b_fixedgaze", _runB)]:
    print(f"  {name:<20} -> {p}")

# %% [markdown]
# ## Step 5: ONE-BATCH 3-SEED PROTOCOL EVALUATION (all four checkpoints)

# %%
print("\n" + "=" * 72)
print("  RUNNING 3-SEED PROTOCOL: n=300/seed, seeds 41/42/43, PGD-50, norm-space")
print("=" * 72)

run(
    f"python3 phase2_attacks/eval_full_epsilon_sweep.py "
    f"--n-samples 300 "
    f"--seeds 41 42 43 "
    f"--pgd-steps 50 "
    f"--batch-size 32 "
    f"--output-dir report/sweep_3seed_protocol "
    f"--eps-norm-space "
    f"--eps-list {EPS_LIST} "
    f"--baseline-label trades_large_baseline "
    f"--ckpt-specs "
    f"trades_large_baseline:{_bsl}:large "
    f"null_ablation_v11:{_null}:v11 "
    f"run_a_norecon:{_runA}:v11 "
    f"run_b_fixedgaze:{_runB}:v11:freeze"
)

# %% [markdown]
# ## Step 6: Results

# %%
import csv, os

agg = "report/sweep_3seed_protocol/epsilon_sweep_results.csv"
per_seed = "report/sweep_3seed_protocol/epsilon_sweep_per_seed.csv"
if os.path.exists(agg):
    with open(agg) as f:
        rows = list(csv.DictReader(f))
    print("Aggregated (mean ± std over seeds):")
    for r in rows:
        print(f"  {r['ckpt_label']:<22} eps={float(r['eps_pixel']):>6.3f}  "
              f"acc {r['acc_mean']}±{r['acc_std']}  "
              f"d' {r['macro_dprime_mean']}±{r['macro_dprime_std']}  (n={r['n_seeds']})")
    print(f"\nCSVs: {agg}\n      {per_seed}")
else:
    print("Aggregated CSV not found — check the eval log for errors.")

print("\n" + "=" * 72)
print("  DONE — 3-SEED PROTOCOL COMPLETE.")
print("  Crossover at eps=0.094 is REAL only if (RHAN - baseline) > 2·σ_combined.")
print("  Do NOT write findings until the mean±std table is reviewed.")
print("=" * 72)
