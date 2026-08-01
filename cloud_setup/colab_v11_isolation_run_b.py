#!/usr/bin/env python3
"""
Colab Notebook Pipeline: RHAN-v11 Isolation Run B (Freeze Foveal Gaze to Center)
================================================================================
Isolates whether the ACTIVE MOVEMENT of the foveal crop (vs fixed-center crop) is what matters.

Run B Configuration:
  - Base checkpoint: checkpoints/rhan_stl10_large_pseudolabel_best.pth
  - Curriculum: 60 epochs fresh start
  - Loss weights: --w-trades 0.55 --w-foraging 0 --w-precision 0 --w-halt 0 --w-recon 0.10
  - Flag: --freeze-gaze (Hardcodes fixation to center (0,0) for all samples/steps)
  - Output checkpoint: checkpoints/rhan_v11_isolation_fixedgaze_best.pth
  - Synced to HuggingFace: FerrariKazu/rhan-checkpoints

Followed by Matched Evaluation:
  - PGD-50, NORM-space eps=[0, 0.031, 0.062, 0.094], n=500
    (matched to Finding-17 baseline table; eval uses --freeze-gaze to match training)

Usage: Copy cells directly into Colab. Set HF_TOKEN in Colab Secrets.
"""

# %% [markdown]
# # RHAN-v11 Isolation Run B: Freeze Foveal Gaze to Center (--freeze-gaze)

# %% [markdown]
# ## Step 1: Install Dependencies

# %%
import os, sys, subprocess, time

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
# ## Step 4: Run Training — RUN B (Freeze Foveal Gaze to Center: --freeze-gaze)

# %%
print("\n" + "="*70)
print("  LAUNCHING RUN B: Freeze Foveal Gaze to Center (--freeze-gaze)")
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
    f"--ckpt-name rhan_v11_isolation_fixedgaze"
)

# %% [markdown]
# ## Step 5: Sync latest eval script and checkpoint, then matched sweep

# %%
# Force-sync latest code (pull can silently skip if refs are stale)
run("git fetch origin main && git reset --hard origin/main")

# Resolve checkpoint: try best first, fall back to rolling
import os as _os
from huggingface_hub import hf_hub_download as _hf_dl

_base    = "rhan_v11_isolation_fixedgaze"
_ckpt    = f"checkpoints/{_base}_best.pth"
_ckpt_lbl = f"{_base}_best"            # descriptive label for CSV

if _os.path.exists(_ckpt):
    print(f"  Checkpoint present locally: {_ckpt}", flush=True)
else:
    print(f"  Trying HF rhan-checkpoints ({_base}_best.pth)...", flush=True)
    try:
        _ckpt = _hf_dl(
            repo_id="FerrariKazu/rhan-checkpoints",
            repo_type="dataset",
            filename=f"{_base}_best.pth",
            local_dir="checkpoints",
        )
        print(f"  Downloaded best: {_ckpt}", flush=True)
    except Exception as _e1:
        print(f"  Not in rhan-checkpoints ({_e1.__class__.__name__}). Trying rolling...", flush=True)
        try:
            _ckpt = _hf_dl(
                repo_id="FerrariKazu/rhan-checkpoints-rolling",
                repo_type="dataset",
                filename=f"{_base}_rolling.pth",
                local_dir="checkpoints",
            )
            _ckpt_lbl = f"{_base}_rolling"
            print(f"  Downloaded rolling: {_ckpt}", flush=True)
        except Exception as _e2:
            raise RuntimeError(
                f"No checkpoint found for {_base} on HF or locally.\n"
                f"  rhan-checkpoints error:         {_e1}\n"
                f"  rhan-checkpoints-rolling error: {_e2}\n"
                "Run training (Step 4) before evaluating."
            )

print(f"\n  Using checkpoint: {_ckpt} (label={_ckpt_lbl})", flush=True)

print("\n" + "="*70)
print("  RUNNING MATCHED EVALUATION: Run B (rhan_v11_isolation_fixedgaze)")
print("="*70)

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
    f"--ckpt-specs {_ckpt_lbl}:{_ckpt}:v11"
)
