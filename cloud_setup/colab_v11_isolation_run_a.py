#!/usr/bin/env python3
"""
Colab Notebook Pipeline: RHAN-v11 Isolation Run A (Zero Generative Prior)
==========================================================================
Isolates whether the Generative Prior is responsible for the high-epsilon robustness advantage.

Run A Configuration:
  - Base checkpoint: checkpoints/rhan_stl10_large_pseudolabel_best.pth
  - Curriculum: 60 epochs fresh start
  - Loss weights: --w-trades 0.55 --w-foraging 0 --w-precision 0 --w-halt 0 --w-recon 0
  - Output checkpoint: checkpoints/rhan_v11_isolation_norecon_best.pth
  - Synced to HuggingFace: FerrariKazu/rhan-checkpoints

Followed by Matched Evaluation:
  - PGD-50, eps=[0, 0.0313, 0.0625, 0.094], n=500

Usage: Copy cells directly into Colab. Set HF_TOKEN in Colab Secrets.
"""

# %% [markdown]
# # RHAN-v11 Isolation Run A: Zero Generative Prior (--w-recon 0)

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
# ## Step 4: Run Training — RUN A (Zero Generative Prior: --w-recon 0)

# %%
print("\n" + "="*70)
print("  LAUNCHING RUN A: Zero Generative Prior (--w-recon 0)")
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
# ## Step 5: Matched Evaluation (PGD-50, n=500)

# %%
print("\n" + "="*70)
print("  RUNNING MATCHED EVALUATION: Run A (rhan_v11_isolation_norecon)")
print("="*70)

run(
    f"python3 phase2_attacks/eval_empirical_sweep_verified.py "
    f"--n-samples 500 "
    f"--pgd-steps 50 "
    f"--output-json report/empirical_sweep_isolation_run_a.json"
)
