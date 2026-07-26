#!/usr/bin/env python3
"""
Colab Notebook: RHAN-v11 Null Ablation Training (HF-persistent)
===============================================================
Runs RHAN-v11 training with foraging=0.0 as a null-ablation baseline.
Checkpoints auto-sync to HuggingFace so resume works across Colab
disconnects. Inspired by the Kaggle pipeline's HF persistence pattern.

Resume logic (built into train_rhan_v11.py):
  1. Check local rolling checkpoint
  2. Check HF for newer rolling checkpoint
  3. Resume from the latest
  4. After each epoch, sync rolling + best to HF

Usage: Copy cells into Colab. Set HF_TOKEN in Colab Secrets.
"""

# %% [markdown]
# # RHAN-v11 Null Ablation (foraging=0.0) on Colab
#
# Every epoch saves checkpoints to HuggingFace so you never lose progress.
# If Colab disconnects, just re-run all cells — it auto-resumes.

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

# Install system + Python deps
run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
run("pip install --quiet huggingface_hub datasets Pillow")

# %% [markdown]
# ## Step 2: Clone / sync repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'

if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')

# Add project to Python path
sys.path.insert(0, WORK_DIR)

# %% [markdown]
# ## Step 3: Set HF_TOKEN

# %%
# Try Colab Secrets first, then env var
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
        "HF_TOKEN not found. Set it in Colab Secrets (🔑 key icon in sidebar) "
        "as 'HF_TOKEN', or as an environment variable."
    )
print(f"HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")

# %% [markdown]
# ## Step 4: Clear cached weights (prevent corrupt partial downloads)

# %%
import shutil
for cache_dir in [os.path.expanduser("~/.cache/clip"),
                  os.path.expanduser("~/.cache/huggingface/hub")]:
    if os.path.exists(cache_dir):
        print(f"Clearing {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)

# %% [markdown]
# ## Step 5: Verify GPU

# %%
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")

# %% [markdown]
# ## Step 6: Run Null Ablation Training
#
# Key settings for null ablation:
#   - `--w-foraging 0.0` — disables foraging loss
#
# The training script auto-resumes from the latest rolling checkpoint
# on HuggingFace. If Colab dies mid-epoch, re-run this cell.

# %%
run(
    f"python3 phase1_training/train_rhan_v11.py "
    f"--batch-size 8 "
    f"--accum-steps 32 "
    f"--w-foraging 0.0 "
    f"--w-precision 0.0 "
    f"--w-halt 0.0 "
    f"--max-foraging-steps 4 "
    f"--fovea-size 48 "
    f"--metabolic-cost 0.05 "
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
