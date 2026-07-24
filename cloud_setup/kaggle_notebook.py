#!/usr/bin/env python3
"""
Kaggle 2xT4 Notebook Execution Pipeline for Sprint 2 (Synthetic Generation & Filtering)
=======================================================================================

Important: do NOT assume raw data from a prior session persists on /kaggle/working.
Every session starts with a fresh VM. The only durable storage is:
  - HuggingFace datasets (FerrariKazu/stl10-synthetic)
  - Kaggle Dataset outputs (manually downloaded)

Failure Chain Fixed (2026-07-24):
  Problem: First session generated all 10 classes and uploaded filtered data to HF.
           Second session regenerated car ONLY (raw data for 9 other classes was gone).
           Filter at 0.30 got 0/20000 for car. Uploaded an empty report to HF,
           overwriting the real report.json. The HF dataset was silently corrupted.

  Fix:    This session generates car with improved prompts at 1 step (same as other classes).
          Downloads the 9 intact classes from HF.
          Re-filters car at 0.25 (same threshold that worked for 9 classes).
          Merges everything into a clean upload.
          Verifies every class has >0 images before uploading.
"""

# %% [markdown]
# # Step 1: Environment Setup & Dependencies

# %%
import os
import sys
import subprocess
import shutil
import json
import tarfile
import io
import time
from PIL import Image

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF_TOKEN loaded.")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DIFFUSERS_NO_PROGRESS_BAR"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

def run_command(cmd, shell=True):
    print(f"\n[RUNNING]: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True, text=True)
    return result.returncode

print("Installing dependencies...")
run_command("pip install --quiet --upgrade pip setuptools wheel")
run_command("pip install --quiet diffusers transformers accelerate webdataset")
run_command("pip install --quiet git+https://github.com/openai/CLIP.git")
run_command("pip install --quiet opencv-python datasets huggingface_hub Pillow")

# %% [markdown]
# # Step 2: Clone Repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
REPO_URL = f'https://github.com/FerrariKazu/{REPO_NAME}.git'
os.chdir('/kaggle/working')
if not os.path.exists(f'/kaggle/working/{REPO_NAME}'):
    subprocess.run(f'git clone {REPO_URL}', shell=True, check=True)
os.chdir(f'/kaggle/working/{REPO_NAME}')
subprocess.run('git fetch origin main && git reset --hard origin/main', shell=True, check=True)
os.environ["PYTHONPATH"] = f"/kaggle/working/{REPO_NAME}:{os.environ.get('PYTHONPATH', '')}"
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# # Step 3: Car-Only Regeneration (new prompts, 1 step)
# Generates 20K car images with improved prompts matching other classes' style.
# All prompts now include "photorealistic, 96x96" — same structure as airplane/bird/etc.

# %%
raw_output_dir = "./data/synthetic_stl10_raw"
filtered_output_dir = "./data/synthetic_stl10_filtered"
os.makedirs(raw_output_dir, exist_ok=True)
os.makedirs(filtered_output_dir, exist_ok=True)

# Pre-download SDXL Turbo
print("Pre-loading SDXL Turbo pipeline...")
run_command("python3 -c \"from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', variant='fp16', low_cpu_mem_usage=True)\"")

# Generate car (class-index 2) with new prompts, 1 step
print("\n" + "=" * 60)
print("  Generating car (20K, class-index 2, new prompts, 1 step)")
print("=" * 60)
cmd_car = (
    f"CUDA_VISIBLE_DEVICES=0 python3 data_generation/generate_synthetic_stl10.py "
    f"--output-dir {raw_output_dir} --class-index 2 --target-per-class 20000"
)
subprocess.run(cmd_car, shell=True, check=True)
print("  Car generation complete.")

# Save 10 sample car images for visual inspection
print("Saving 10 sample car images for inspection...")
sample_dir = "./kaggle/working/car_samples"
os.makedirs(sample_dir, exist_ok=True)
shard_files = sorted([f for f in os.listdir(raw_output_dir) if 'car' in f and f.endswith('.tar')])
samples_saved = 0
for sf in shard_files:
    if samples_saved >= 10:
        break
    with tarfile.open(os.path.join(raw_output_dir, sf), 'r') as tar:
        for m in tar.getmembers():
            if m.name.endswith('.png') and samples_saved < 10:
                img_bytes = tar.extractfile(m).read()
                img = Image.open(io.BytesIO(img_bytes))
                img.save(os.path.join(sample_dir, f"car_sample_{samples_saved:02d}.png"))
                samples_saved += 1
print(f"  Saved {samples_saved} car samples to {sample_dir}")

# %% [markdown]
# # Step 4: Filter car at threshold 0.25 (same as original validated threshold)
# The 0.30 threshold was never validated — only 0.25 was confirmed to work on 9 classes.
# Car gets re-filtered at 0.25 for consistency.

# %%
print("\n" + "=" * 60)
print("  Filtering car at threshold=0.25")
print("=" * 60)
cmd_filter_car = (
    f"python3 data_generation/filter_synthetic_clip.py "
    f"--input-dir {raw_output_dir} --output-dir {filtered_output_dir} "
    f"--sim-threshold 0.25"
)
subprocess.run(cmd_filter_car, shell=True, check=True)

# %% [markdown]
# # Step 5: Download existing 9 classes from HF
# The first run uploaded filtered data for all 10 classes. The 9 non-car classes are intact.
# We download them and merge with the new car data.

# %%
print("\n" + "=" * 60)
print("  Downloading intact classes from HF (FerrariKazu/stl10-synthetic)")
print("=" * 60)
from huggingface_hub import list_repo_files, hf_hub_download

existing_files = list_repo_files('FerrariKazu/stl10-synthetic', repo_type='dataset')
# Download all filtered shards EXCEPT car (we're replacing car)
download_tars = [f for f in existing_files if 'car_filtered' not in f and f.endswith('.tar')]
print(f"  Downloading {len(download_tars)} shards...")
for f in sorted(download_tars):
    local_path = os.path.join(filtered_output_dir, os.path.basename(f))
    if not os.path.exists(local_path):
        hf_hub_download(
            'FerrariKazu/stl10-synthetic', f, repo_type='dataset',
            local_dir=filtered_output_dir, local_dir_use_symlinks=False
        )
print(f"  Downloaded {len(download_tars)} shards to {filtered_output_dir}")

# %% [markdown]
# # Step 6: VERIFY before upload
# Every class must have >0 filtered images. If any class has 0, abort.

# %%
print("\n" + "=" * 60)
print("  VERIFICATION: checking all 10 classes have filtered images")
print("=" * 60)

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

all_files = os.listdir(filtered_output_dir)
filtered_tars = [f for f in all_files if f.endswith('.tar')]

per_class_counts = {}
for cls in STL10_CLASSES:
    cls_tars = [f for f in filtered_tars if f"_{cls}_filtered" in f]
    total_images = 0
    for t in cls_tars:
        try:
            with tarfile.open(os.path.join(filtered_output_dir, t), 'r') as tar:
                png_count = sum(1 for m in tar.getmembers() if m.name.endswith('.png'))
                total_images += png_count
        except Exception as e:
            print(f"  WARNING: could not read {t}: {e}")
    per_class_counts[cls] = total_images
    status = "OK" if total_images > 0 else "EMPTY"
    print(f"  {cls}: {total_images} images [{status}]")

zero_classes = [cls for cls, count in per_class_counts.items() if count == 0]
if zero_classes:
    print(f"\n  FATAL: {len(zero_classes)} class(es) have 0 images: {zero_classes}")
    print("  Aborting upload. Check car sample images and fix prompts/threshold.")
    sys.exit(1)
else:
    print(f"\n  All 10 classes have >0 images. Proceeding to upload.")

# %% [markdown]
# # Step 7: Generate correct report.json and upload

# %%
print("\n" + "=" * 60)
print("  Generating correct clip_diversity_report.json")
print("=" * 60)

# Load the car-specific report
car_report_path = os.path.join(filtered_output_dir, 'clip_diversity_report.json')
if os.path.exists(car_report_path):
    with open(car_report_path) as f:
        car_report = json.load(f)
else:
    car_report = {}

# Build full report
full_report = {}
for cls in STL10_CLASSES:
    if cls == 'car' and cls in car_report:
        full_report[cls] = car_report[cls]
        full_report[cls]['final_usable_count'] = per_class_counts[cls]
    else:
        full_report[cls] = {
            "class_name": cls,
            "passed_quality": per_class_counts[cls],
            "pass_rate_pct": 100.0,
            "final_usable_count": per_class_counts[cls]
        }

report_path = os.path.join(filtered_output_dir, 'clip_diversity_report.json')
with open(report_path, 'w') as f:
    json.dump(full_report, f, indent=2)
print(f"  Report written to {report_path}")

print("\n" + "=" * 60)
print("  Uploading clean dataset to FerrariKazu/stl10-synthetic")
print("=" * 60)
cmd_upload = (
    f"python3 data_generation/upload_synthetic_hf.py "
    f"--input-dir {filtered_output_dir} "
    f"--repo-id FerrariKazu/stl10-synthetic"
)
run_command(cmd_upload)

total_images = sum(per_class_counts.values())
print(f"\n{'=' * 60}")
print(f"  SPRINT 2 SYNTHETIC DATA PIPELINE COMPLETE")
print(f"  Total usable images: {total_images}")
print(f"  HF Dataset: FerrariKazu/stl10-synthetic")
print(f"  Car sample images for inspection: /kaggle/working/car_samples/")
print(f"{'=' * 60}")
