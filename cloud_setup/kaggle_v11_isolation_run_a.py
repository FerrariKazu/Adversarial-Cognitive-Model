#!/usr/bin/env python3
"""
Kaggle Notebook Pipeline: RHAN-v11 Isolation Runs A + B (single notebook)
=========================================================================
Runs BOTH isolation experiments back-to-back in one notebook so a single
Kaggle run completes both:

  RUN A — Zero Generative Prior (--w-recon 0)
  RUN B — Freeze Foveal Gaze to Center (--freeze-gaze, --w-recon 0.10)

Each run is followed by the MATCHED Finding-17 evaluation:
  - PGD-50, NORM-space eps=[0, 0.031, 0.062, 0.094], n=500
    (matched to Finding-17 baseline table — see --eps-norm-space in eval script)
  - PLUS the TRADES Large baseline (rhan_stl10_large_pseudolabel_best.pth)
    as a Finding-17 sanity check (expect ~48.0/40.3/33.7). For Run B the
    baseline is swept WITHOUT --freeze-gaze (it was not trained frozen).

Run A Configuration:
  - Base checkpoint: checkpoints/rhan_stl10_large_pseudolabel_best.pth
  - Curriculum: 60 epochs fresh start
  - Loss weights: --w-trades 0.55 --w-foraging 0 --w-precision 0 --w-halt 0 --w-recon 0
  - Output checkpoint: checkpoints/rhan_v11_isolation_norecon_best.pth

Run B Configuration:
  - Base checkpoint: checkpoints/rhan_stl10_large_pseudolabel_best.pth
  - Loss weights: --w-trades 0.55 --w-foraging 0 --w-precision 0 --w-halt 0 --w-recon 0.10
  - Flag: --freeze-gaze (fixation hardcoded to center (0,0))
  - Output checkpoint: checkpoints/rhan_v11_isolation_fixedgaze_best.pth

Both checkpoints sync to HuggingFace: FerrariKazu/rhan-checkpoints

If training is already done (both checkpoints are on HF), set SKIP_TRAINING=True
in Step 1 — the notebook then runs BOTH matched evals eval-only, and the eval
steps download the checkpoints from HuggingFace.

NOTE on --force-single-gpu (T4x2 crash fix):
  Kaggle T4x2 notebooks expose 2 GPUs, so train_rhan_v11.py auto-wraps the
  model in nn.DataParallel. On T4/Turing (sm_75), DataParallel + channels_last
  + fp16 autocast triggers 'CUDA error: misaligned address' during the very
  first warmup step. --force-single-gpu skips the DataParallel wrap (trains on
  GPU 0 only), fixing the crash and matching Run B's single-GPU setup for a
  clean comparison. If you want to attempt 2-GPU training anyway, drop the
  flag and set os.environ['CUDA_LAUNCH_BLOCKING']='1' to pinpoint the failing
  kernel.

Usage: Copy cells directly into Kaggle Notebook. Set HF_TOKEN in Kaggle Secrets.
"""

# %% [markdown]
# # RHAN-v11 Isolation Runs A + B on Kaggle (one notebook, both experiments)

# %% [markdown]
# ## Step 1: Dependencies and Environment Setup

# %%
import os, sys, subprocess, time

# Fetch HF_TOKEN from Kaggle Secrets or environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

if not hf_token:
    raise RuntimeError("HF_TOKEN not found. Set it in Kaggle Secrets as 'HF_TOKEN'.")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, text=True)
    return result.returncode

# ── Toggle: set True to SKIP both 60-epoch trainings (eval-only). ─────────────
# Both isolation checkpoints are already on FerrariKazu/rhan-checkpoints, so
# eval-only is ~1-2h instead of ~12h+. The eval steps auto-download them.
SKIP_TRAINING = False

run("pip install --quiet --upgrade pip setuptools wheel")
# Pin the same torch build Run A used on Colab (cu121). Kaggle's preinstalled
# torch (newer CUDA builds) has known Turing/sm_75 kernel regressions that can
# produce 'CUDA error: misaligned address' even on a single GPU.
run("pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
run("pip install --quiet opencv-python datasets huggingface_hub Pillow scipy")

# %% [markdown]
# ## Step 2: Clone and Sync Repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
REPO_URL = f'https://github.com/FerrariKazu/{REPO_NAME}.git'

os.chdir('/kaggle/working')

if not os.path.exists(f'/kaggle/working/{REPO_NAME}'):
    print('Cloning repository...')
    subprocess.run(f'git clone {REPO_URL}', shell=True, check=True)

os.chdir(f'/kaggle/working/{REPO_NAME}')
print('Syncing repository to latest commit...')
subprocess.run('git fetch origin main && git reset --hard origin/main', shell=True, check=True)

# Set PYTHONPATH
os.environ["PYTHONPATH"] = f"/kaggle/working/{REPO_NAME}:{os.environ.get('PYTHONPATH', '')}"
print(f"Working directory successfully set to: {os.getcwd()}")

# ── Shared checkpoint resolvers ───────────────────────────────────────────────
import os as _os
from huggingface_hub import hf_hub_download as _hf_dl

def resolve_ckpt(base):
    """Return (label, path) for a checkpoint, local-first with HF fallback."""
    best = f"checkpoints/{base}_best.pth"
    if _os.path.exists(best):
        print(f"  Checkpoint present locally: {best}", flush=True)
        return f"{base}_best", best
    print(f"  Downloading {base}_best.pth from FerrariKazu/rhan-checkpoints...", flush=True)
    try:
        path = _hf_dl(repo_id="FerrariKazu/rhan-checkpoints", repo_type="dataset",
                      filename=f"{base}_best.pth", local_dir="checkpoints")
        print(f"  Downloaded best: {path}", flush=True)
        return f"{base}_best", path
    except Exception as _e1:
        print(f"  Not in rhan-checkpoints ({_e1.__class__.__name__}). Trying rolling...",
              flush=True)
        try:
            path = _hf_dl(repo_id="FerrariKazu/rhan-checkpoints-rolling",
                          repo_type="dataset",
                          filename=f"{base}_rolling.pth", local_dir="checkpoints")
            print(f"  Downloaded rolling: {path}", flush=True)
            return f"{base}_rolling", path
        except Exception as _e2:
            raise RuntimeError(
                f"No checkpoint found for {base} on HF or locally.\n"
                f"  rhan-checkpoints error:         {_e1}\n"
                f"  rhan-checkpoints-rolling error: {_e2}\n"
                "Run the training step before evaluating."
            )

def resolve_baseline():
    """TRADES Large baseline — Finding-17 sanity check (~48.0/40.3/33.7)."""
    path = "checkpoints/rhan_stl10_large_pseudolabel_best.pth"
    if _os.path.exists(path):
        print(f"  TRADES Large baseline present locally: {path}", flush=True)
        return path
    print("  Downloading TRADES Large baseline (rhan_stl10_large_pseudolabel_best.pth)...",
          flush=True)
    path = _hf_dl(repo_id="FerrariKazu/rhan-checkpoints", repo_type="dataset",
                  filename="rhan_stl10_large_pseudolabel_best.pth",
                  local_dir="checkpoints")
    print(f"  Downloaded baseline: {path}", flush=True)
    return path

# %% [markdown]
# ## Step 3: Run A Training — Zero Generative Prior (--w-recon 0)

# %%
if not SKIP_TRAINING:
    print("\n" + "="*70)
    print("  LAUNCHING RUN A ON KAGGLE: Zero Generative Prior (--w-recon 0)")
    print("="*70)
    run(
        f"python3 phase1_training/train_rhan_v11.py "
        f"--target-ckpt checkpoints/rhan_stl10_large_pseudolabel_best.pth "
        f"--w-trades 0.55 "
        f"--w-foraging 0 "
        f"--w-precision 0 "
        f"--w-halt 0 "
        f"--w-recon 0 "
        f"--force-single-gpu "
        f"--ckpt-name rhan_v11_isolation_norecon"
    )
else:
    print("  SKIP_TRAINING=True → skipping Run A training "
          "(eval will pull the checkpoint from HuggingFace)", flush=True)

# %% [markdown]
# ## Step 4: Run A Matched Evaluation (+ TRADES Large baseline)

# %%
print("\n" + "="*70)
print("  RUNNING MATCHED EVALUATION ON KAGGLE: Run A (rhan_v11_isolation_norecon)")
print("="*70)

_lbl_a, _ckpt_a = resolve_ckpt("rhan_v11_isolation_norecon")
_bsl = resolve_baseline()
print(f"  Using checkpoint: {_ckpt_a} (label={_lbl_a})", flush=True)
print(f"  Using baseline : {_bsl}", flush=True)

run(
    f"python3 phase2_attacks/eval_full_epsilon_sweep.py "
    f"--n-samples 500 "
    f"--pgd-steps 50 "
    f"--batch-size 32 "
    f"--output-dir report/sweep_isolation_run_a "
    f"--eps-norm-space "
    f"--eps-list 0.0 0.031 0.062 0.094 "
    f"--seed 42 "
    f"--ckpt-specs {_lbl_a}:{_ckpt_a}:v11 "
    f"trades_large_baseline:{_bsl}:large"
)

# %% [markdown]
# ## Step 5: Run B Training — Freeze Foveal Gaze to Center (--freeze-gaze)

# %%
if not SKIP_TRAINING:
    print("\n" + "="*70)
    print("  LAUNCHING RUN B ON KAGGLE: Freeze Foveal Gaze to Center (--freeze-gaze)")
    print("="*70)
    run(
        f"python3 phase1_training/train_rhan_v11.py "
        f"--target-ckpt checkpoints/rhan_stl10_large_pseudolabel_best.pth "
        f"--w-trades 0.55 "
        f"--w-foraging 0 "
        f"--w-precision 0 "
        f"--w-halt 0 "
        f"--w-recon 0.10 "
        f"--freeze-gaze "
        f"--force-single-gpu "
        f"--ckpt-name rhan_v11_isolation_fixedgaze"
    )
else:
    print("  SKIP_TRAINING=True → skipping Run B training "
          "(eval will pull the checkpoint from HuggingFace)", flush=True)

# %% [markdown]
# ## Step 6: Run B Matched Evaluation (--freeze-gaze) + TRADES Large baseline (no freeze)

# %%
print("\n" + "="*70)
print("  RUNNING MATCHED EVALUATION ON KAGGLE: Run B (rhan_v11_isolation_fixedgaze)")
print("="*70)

_lbl_b, _ckpt_b = resolve_ckpt("rhan_v11_isolation_fixedgaze")
_bsl = resolve_baseline()   # idempotent — no-op if already downloaded
print(f"  Using checkpoint: {_ckpt_b} (label={_lbl_b})", flush=True)

# Run B sweep — --freeze-gaze MUST match training.
run(
    f"python3 phase2_attacks/eval_full_epsilon_sweep.py "
    f"--n-samples 500 "
    f"--pgd-steps 50 "
    f"--batch-size 32 "
    f"--output-dir report/sweep_isolation_run_b "
    f"--eps-norm-space "
    f"--eps-list 0.0 0.031 0.062 0.094 "
    f"--seed 42 "
    f"--freeze-gaze "
    f"--ckpt-specs {_lbl_b}:{_ckpt_b}:v11"
)

# Baseline sanity sweep — WITHOUT --freeze-gaze (baseline was not trained frozen).
print("\n" + "="*70)
print("  RUNNING BASELINE SANITY SWEEP ON KAGGLE: TRADES Large (no --freeze-gaze)")
print("="*70)
run(
    f"python3 phase2_attacks/eval_full_epsilon_sweep.py "
    f"--n-samples 500 "
    f"--pgd-steps 50 "
    f"--batch-size 32 "
    f"--output-dir report/sweep_trades_baseline "
    f"--eps-norm-space "
    f"--eps-list 0.0 0.031 0.062 0.094 "
    f"--seed 42 "
    f"--ckpt-specs trades_large_baseline:{_bsl}:large"
)

print("\n" + "="*70)
print("  ALL DONE: Run A + Run B training and matched evaluations complete.")
print("="*70)
