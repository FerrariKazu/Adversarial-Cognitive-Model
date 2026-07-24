#!/usr/bin/env python3
"""
Kaggle 2xT4 Notebook Execution Pipeline for Sprint 2 (Synthetic Generation & Filtering)
=======================================================================================
This script automates Sprint 2 Phase A (SDXL Turbo generation on T4x2) and Phase B
(CLIP quality & diversity filtering + HuggingFace dataset upload).

It is formatted with '# %%' markers for direct execution in Kaggle Notebook cells.
"""

# %% [markdown]
# # Step 1: Environment Setup & Dependencies
# Installs packages needed for SDXL Turbo generation, CLIP filtering, and WebDataset packaging.

# %%
import os
import sys
import subprocess
import shutil
import time

# Fetch HF_TOKEN from environment or Kaggle secrets
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("✓ HF_TOKEN successfully loaded and injected into environment.")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DIFFUSERS_NO_PROGRESS_BAR"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

def run_command(cmd, shell=True):
    print(f"\n[RUNNING]: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True, text=True)
    return result.returncode

print("\n>>> Installing Python Dependencies for Sprint 2...")
run_command("pip install --quiet --upgrade pip setuptools wheel")
# diffusers + transformers + accelerate for SDXL Turbo; webdataset for shard I/O
run_command("pip install --quiet diffusers transformers accelerate webdataset")
# CLIP for Phase B quality/diversity filtering (openai/CLIP — no PyPI release)
run_command("pip install --quiet git+https://github.com/openai/CLIP.git")
# opencv-python, datasets, huggingface_hub are standard; note: PIL is the import
# name — the PyPI package is Pillow (usually pre-installed on Kaggle, listed just in case)
run_command("pip install --quiet opencv-python datasets huggingface_hub Pillow")

# %% [markdown]
# # Step 2: Clone and Sync Repository
# Syncs latest code from GitHub to `/kaggle/working/Adversarial-Cognitive-Model`.

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
# # Step 3: Car-Only Regeneration with Improved Prompts + More Steps
# Car had 12.8% pass rate at threshold 0.25 — prompts were too elaborate for SDXL
# Turbo at 1 step. Regenerate with simpler concrete prompts and 3 inference steps.
# 
# ~4.5 hours on T4 for 20K images at 3 steps. Other classes are not regenerated
# (their raw data from the first pass is expected in ./data/synthetic_stl10_raw).

# %%
raw_output_dir = "./data/synthetic_stl10_raw"
os.makedirs(raw_output_dir, exist_ok=True)

print("--> Pre-downloading SDXL Turbo pipeline to HuggingFace cache...")
run_command("python3 -c \"from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', variant='fp16', low_cpu_mem_usage=True)\"")

print("\n============================================================")
print("  Sprint 2 Phase A.5: Car-Only Regeneration (3 steps, simpler prompts)")
print("============================================================")
cmd_car = f"CUDA_VISIBLE_DEVICES=0 python3 data_generation/generate_synthetic_stl10.py --output-dir {raw_output_dir} --class-index 2 --target-per-class 20000"
print("Regenerating car (class-index 2, target 20K raw for ~2-4K filtered at threshold 0.30)...")
subprocess.run(cmd_car, shell=True, check=True)
print("✓ Car regeneration complete.")

# %% [markdown]
# # Step 4: Phase B — CLIP Quality Gate at Threshold 0.30
# Filters ALL raw data (car from new generation + other classes from first pass)
# at the recalibrated threshold 0.30 (was 0.25, which let through 100% on some classes).

# %%
print("\n============================================================")
print("  Sprint 2 Phase B: CLIP Quality & Diversity Filtering (threshold=0.30)")
print("============================================================")

filtered_output_dir = "./data/synthetic_stl10_filtered"
cmd_filter = f"python3 data_generation/filter_synthetic_clip.py --input-dir {raw_output_dir} --output-dir {filtered_output_dir} --sim-threshold 0.30"

run_command(cmd_filter)

# %% [markdown]
# # Step 5: Phase B — HuggingFace Dataset Upload
# Uploads filtered shards to HuggingFace dataset repo `FerrariKazu/stl10-synthetic`.

# %%
print("\n============================================================")
print("  Sprint 2 Phase B: Uploading Dataset to HuggingFace")
print("============================================================")

cmd_upload = f"python3 data_generation/upload_synthetic_hf.py --input-dir {filtered_output_dir} --repo-id FerrariKazu/stl10-synthetic"

run_command(cmd_upload)

print("\n🎉 Sprint 2 Kaggle Pipeline Execution Completed Successfully!")
