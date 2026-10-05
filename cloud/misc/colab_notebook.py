#!/usr/bin/env python3
"""
Colab Notebook: RHAN-v11 Evaluation
=====================================
Downloads the best checkpoint from HuggingFace and runs the 4-suite
evaluation (statistical significance, SOTA comparison, biological claims,
diagnostic plots). Uses quick defaults (200 samples, 10 PGD steps) for
~30 min runtime on T4; pass --steps 20 --num-samples 500 for full eval.

Usage: Copy cells into Colab. Set HF_TOKEN in Colab Secrets.
"""

# %% [markdown]
# # RHAN-v11 Evaluation on Colab
#
# Quick evaluation (~30 min on T4). For full eval, edit Step 5 to add
# `--steps 20 --num-samples 500`.

# %% [markdown]
# ## Step 1: Install dependencies

# %%
import os, sys, subprocess, time

def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    rc = process.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
run("pip install --quiet huggingface_hub datasets Pillow matplotlib seaborn")

# %% [markdown]
# ## Step 2: Clone / sync repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'

if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
sys.path.insert(0, WORK_DIR)

# %% [markdown]
# ## Step 3: Set HF_TOKEN

# %%
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN not found. Set it in Colab Secrets (key icon in sidebar) "
        "as 'HF_TOKEN', or as an environment variable."
    )
print(f"HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")

# %% [markdown]
# ## Step 4: Verify GPU

# %%
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")

# %% [markdown]
# ## Step 5: Run Evaluation
#
# Downloads the best checkpoint from HuggingFace automatically.
# Uses quick defaults (200 samples, 10 PGD steps). For the full
# 4-suite evaluation (~2h), add `--steps 20 --num-samples 500`.

# %%
run(
    f"python3 phase1_training/eval_rhan_v11.py "
    f"--checkpoint checkpoints/rhan_stl10_v11_best.pth "
    f"--num-samples 200 "
    f"--steps 10 "
)

# %% [markdown]
# ## Step 6: Verify checkpoints on HuggingFace

# %%
print("\n--- Checkpoints on HuggingFace ---")
try:
    from huggingface_hub import list_repo_files
    for repo in ['FerrariKazu/rhan-checkpoints', 'FerrariKazu/rhan-checkpoints-rolling']:
        files = list_repo_files(repo, repo_type='dataset', token=hf_token)
        v11_files = [f for f in files if 'v11' in f]
        print(f"  {repo}: {v11_files}")
except Exception as e:
    print(f"  Error listing: {e}")

# %% [markdown]
# ## Pro Tip: Prevent Colab Disconnect
# Open browser console (Ctrl+Shift+I) → Console tab → paste:
# ```javascript
# function KeepAlive() {
#     document.querySelector("colab-connect-button").click();
# }
# setInterval(KeepAlive, 60000);
# ```
