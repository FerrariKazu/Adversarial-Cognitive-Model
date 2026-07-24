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
# # Step 3: Phase A — Parallel SDXL Turbo Synthetic Generation (T4×2)
# Runs GPU 0 for even class indices (0, 2, 4, 6, 8) and GPU 1 for odd class indices (1, 3, 5, 7, 9)
# simultaneously to maximize dual-GPU throughput on Kaggle.

# %%
print("\n============================================================")
print("  Sprint 2 Phase A: Parallel Generation on Kaggle T4x2")
print("============================================================")

raw_output_dir = "./data/synthetic_stl10_raw"
os.makedirs(raw_output_dir, exist_ok=True)

print("--> Pre-downloading SDXL Turbo pipeline to HuggingFace cache to prevent lockups...")
run_command("python3 -c \"from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', variant='fp16', low_cpu_mem_usage=True)\"")

# Parallel processes with strict CUDA_VISIBLE_DEVICES isolation
# GPU 0 handles even classes (0, 2, 4, 6, 8), GPU 1 handles odd classes (1, 3, 5, 7, 9)
cmd_gpu0 = f"CUDA_VISIBLE_DEVICES=0 python3 data_generation/generate_synthetic_stl10.py --output-dir {raw_output_dir} --gpu-split even --device cuda:0"
cmd_gpu1 = f"CUDA_VISIBLE_DEVICES=1 python3 data_generation/generate_synthetic_stl10.py --output-dir {raw_output_dir} --gpu-split odd --device cuda:0"

print(f"Launching GPU 0 process (Even classes: airplane, car, deer, horse, ship)...")
p0 = subprocess.Popen(cmd_gpu0, shell=True)

print("Waiting 10 seconds for GPU 0 initialization before launching GPU 1...")
time.sleep(10)

print(f"Launching GPU 1 process (Odd classes: bird, cat, dog, monkey, truck)...")
p1 = subprocess.Popen(cmd_gpu1, shell=True)

# Wait for both generation streams to complete
p0.wait()
p1.wait()

if p0.returncode != 0 or p1.returncode != 0:
    raise RuntimeError(f"Generation failed! GPU 0 code: {p0.returncode}, GPU 1 code: {p1.returncode}")

print("\n✓ Phase A Generation Finished!")

# %% [markdown]
# # Step 4: Phase A.5 — Car-Only Regeneration with Improved Prompts + More Steps
# Car had 12.8% pass rate at threshold 0.25. Regenerate with simpler prompts
# and 3 inference steps (instead of 1) for better fidelity.

# %%
print("\n============================================================")
print("  Car Regeneration with Improved Prompts + 3 Inference Steps")
print("============================================================")
cmd_car = f"CUDA_VISIBLE_DEVICES=0 python3 data_generation/generate_synthetic_stl10.py --output-dir {raw_output_dir} --class-index 2 --target-per-class 20000"
print("Regenerating car (class-index 2 = car, target 20K raw for ~2-4K filtered)...")
subprocess.run(cmd_car, shell=True, check=True)
print("✓ Car regeneration complete.")

# %% [markdown]
# # Step 5: Phase B — CLIP Quality Gate & Pairwise Diversity Filtering
# Filters raw images using CLIP similarity threshold (0.30, up from 0.25 after
# diagnosing 100% pass rates on deer/horse/monkey at the old threshold).

# %%
print("\n============================================================")
print("  Sprint 2 Phase B: CLIP Quality & Diversity Filtering")
print("============================================================")

filtered_output_dir = "./data/synthetic_stl10_filtered"
cmd_filter = f"python3 data_generation/filter_synthetic_clip.py --input-dir {raw_output_dir} --output-dir {filtered_output_dir} --sim-threshold 0.30"

run_command(cmd_filter)

# %% [markdown]
# # Step 6: Phase B — HuggingFace Dataset Upload
# Uploads filtered shards to HuggingFace dataset repo `FerrariKazu/stl10-synthetic`.

# %%
print("\n============================================================")
print("  Sprint 2 Phase B: Uploading Dataset to HuggingFace")
print("============================================================")

cmd_upload = f"python3 data_generation/upload_synthetic_hf.py --input-dir {filtered_output_dir} --repo-id FerrariKazu/stl10-synthetic"

run_command(cmd_upload)

print("\n🎉 Sprint 2 Kaggle Pipeline Execution Completed Successfully!")
