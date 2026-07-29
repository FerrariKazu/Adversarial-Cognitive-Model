#!/usr/bin/env python3
"""
Kaggle Notebook Pipeline: RHAN-v11 Isolation Run A (Zero Generative Prior)
===========================================================================
Isolates whether the Generative Prior is responsible for the high-epsilon robustness advantage.

Run A Configuration:
  - Base checkpoint: checkpoints/rhan_stl10_large_pseudolabel_best.pth
  - Curriculum: 60 epochs fresh start
  - Loss weights: --w-trades 0.55 --w-foraging 0 --w-precision 0 --w-halt 0 --w-recon 0
  - Output checkpoint: checkpoints/rhan_v11_isolation_norecon_best.pth
  - Synced to HuggingFace: FerrariKazu/rhan-checkpoints

Followed by Matched Evaluation:
  - PGD-50, eps=[0, 0.0313, 0.0625, 0.094], n=500

Usage: Copy cells directly into Kaggle Notebook. Set HF_TOKEN in Kaggle Secrets.
"""

# %% [markdown]
# # RHAN-v11 Isolation Run A on Kaggle: Zero Generative Prior (--w-recon 0)

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

run("pip install --quiet --upgrade pip setuptools wheel")
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

# %% [markdown]
# ## Step 3: Run Training — RUN A (Zero Generative Prior: --w-recon 0)

# %%
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
    f"--ckpt-name rhan_v11_isolation_norecon "
    f"--force-restart"
)

# %% [markdown]
# ## Step 4: Matched Evaluation (PGD-50, n=500)

# %%
print("\n" + "="*70)
print("  RUNNING MATCHED EVALUATION ON KAGGLE: Run A (rhan_v11_isolation_norecon)")
print("="*70)

run(
    f"python3 phase2_attacks/eval_empirical_sweep_verified.py "
    f"--n-samples 500 "
    f"--pgd-steps 50 "
    f"--output-json report/empirical_sweep_isolation_run_a.json"
)
