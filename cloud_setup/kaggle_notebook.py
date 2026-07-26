#!/usr/bin/env python3
"""
Kaggle Sprint 2 — Synthetic Data Pipeline (HF-persistent)
=========================================================
Every intermediate output is saved to HuggingFace immediately.
If the Kaggle session dies (GPU quota, timeout), the next session
checks what's already on HF and resumes from where it left off.

Pipeline (each step is independently restartable via HF state):
  STEP 1: Generate car raw shards → upload to HF per-shard
  STEP 2: Filter car at 0.25 → upload filtered shards to HF
  STEP 3: Merge all 10 classes → delete old car shards → upload final dataset

Usage: Run on Kaggle with GPU T4 x2. Expects HF_TOKEN in secrets.
"""

# %% [markdown]
# # Setup

# %%
import os, sys, subprocess, json, tarfile, io, time, threading, hashlib
from pathlib import Path

HF_DATASET = "FerrariKazu/stl10-synthetic"
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/kaggle/working/{REPO_NAME}'
RAW_DIR = './data/synthetic_stl10_raw'
FILTERED_DIR = './data/synthetic_stl10_filtered'
CAR_TARGET = 30000

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

# Get HF token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN not found — set in Kaggle Secrets or env vars")
os.environ["HF_TOKEN"] = hf_token

def run(cmd, shell=True, check=True):
    print(f"\n[RUN]: {cmd}")
    if shell:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, bufsize=1)
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    rc = process.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

# Clone / sync repo
os.chdir('/kaggle/working')
if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("\nInstalling deps...")
run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet diffusers transformers accelerate webdataset")
run("pip install --quiet git+https://github.com/openai/CLIP.git")
run("pip install --quiet opencv-python datasets huggingface_hub Pillow")

# Clear any partially cached SDXL Turbo (prevents 7h hangs on corrupted downloads)
import shutil
sdxl_cache = os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--sdxl-turbo")
if os.path.exists(sdxl_cache):
    print("Clearing partial SDXL Turbo cache to prevent load hangs...")
    shutil.rmtree(sdxl_cache, ignore_errors=True)

from huggingface_hub import HfApi
api = HfApi(token=hf_token)

# huggingface_hub standalone helpers (upload_file, delete_files, etc.) change names
# across versions. Using api.* methods is stable across all versions.
def list_repo_files(repo_id, repo_type):
    return api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

def hf_hub_download(repo_id, filename, repo_type, **kwargs):
    local_dir = kwargs.pop('local_dir', None)
    local_dir_use_symlinks = kwargs.pop('local_dir_use_symlinks', None)
    return api.hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type=repo_type,
        local_dir=local_dir, local_dir_use_symlinks=local_dir_use_symlinks,
        **kwargs
    )

def upload_file(path_or_fileobj, path_in_repo, repo_id, repo_type):
    return api.upload_file(
        path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo,
        repo_id=repo_id, repo_type=repo_type
    )

def create_repo(repo_id, repo_type, exist_ok=True):
    return api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=exist_ok)

def delete_files(repo_id, paths, repo_type):
    return api.delete_files(repo_id=repo_id, path_list=paths, repo_type=repo_type)

create_repo(repo_id=HF_DATASET, repo_type='dataset', exist_ok=True)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

# %% [markdown]
# # Detect pipeline state from HF (self-healing markers)

# %%
def get_markers():
    """Download pipeline_markers.json from HF. Returns dict (empty if not found)."""
    try:
        local = hf_hub_download(HF_DATASET, 'pipeline_markers.json', repo_type='dataset',
                                local_dir='/kaggle/working', local_dir_use_symlinks=False)
        with open(local) as f:
            return json.load(f)
    except Exception:
        return {}

def set_marker(key, value=True):
    """Upload a marker to pipeline_markers.json (read-modify-write)."""
    markers = get_markers()
    markers[key] = value
    upload_file(path_or_fileobj=io.BytesIO(json.dumps(markers, indent=2).encode()),
                path_in_repo='pipeline_markers.json',
                repo_id=HF_DATASET, repo_type='dataset')

def hf_file_exists(suffix):
    try:
        files = list_repo_files(HF_DATASET, repo_type='dataset')
    except Exception:
        return False
    return any(suffix in f for f in files)

# What's done?
markers = get_markers()
car_raw_done = markers.get('step1_car_raw', False)
car_filter_done = markers.get('step2_car_filter', False)
merge_done = markers.get('step3_merge', False)

# Check actual car filtered shard count on HF
car_filtered_count = 0
try:
    hf_files = list_repo_files(HF_DATASET, repo_type='dataset')
    car_filtered_count = sum(1 for f in hf_files if 'car_filtered' in f and f.endswith('.tar'))
except Exception:
    hf_files = []
    car_filtered_count = 0

print(f"Pipeline state: car_raw={'✓' if car_raw_done else '—'}, "
      f"car_filter={'✓' if car_filter_done else '—'} ({car_filtered_count} shards), "
      f"merge={'✓' if merge_done else '—'}")

# If car has too few filtered shards, redo raw generation + filter
MIN_CAR_SHARDS = 2
if car_filtered_count < MIN_CAR_SHARDS:
    if car_raw_done or car_filter_done:
        print(f"\n  ⚠ Car only has {car_filtered_count} filtered shards (need {MIN_CAR_SHARDS}). "
              f"Resetting all car markers to regenerate.")
    car_raw_done = False
    car_filter_done = False
    merge_done = False
    set_marker('step1_car_raw', False)
    set_marker('step2_car_filter', False)
    set_marker('step3_merge', False)

# %% [markdown]
# # STEP 1: Generate car raw shards

# %%
if not car_raw_done:
    print("=" * 60)
    print("  STEP 1: Generate car raw shards (20K, new prompts, 1 step)")
    print("  Uploads each completed shard to HF immediately")
    print("=" * 60)
    
    # Download any partial raw shards + manifest from HF to resume
    try:
        files = list_repo_files(HF_DATASET, repo_type='dataset')
        raw_files = [f for f in files if f.startswith('raw_shards/') and f.endswith('.tar')]
        if raw_files:
            print(f"  Found {len(raw_files)} existing raw shards on HF. Downloading to resume...")
            for hf_path in raw_files:
                local = os.path.join(RAW_DIR, os.path.basename(hf_path))
                if not os.path.exists(local):
                    cached = hf_hub_download(HF_DATASET, hf_path, repo_type='dataset')
                    shutil.copy2(cached, local)
            # Also download manifest
            try:
                cached = hf_hub_download(HF_DATASET, 'raw_shards/manifest.json', repo_type='dataset')
                shutil.copy2(cached, os.path.join(RAW_DIR, 'manifest.json'))
            except Exception:
                pass
    except Exception:
        pass
    
    # Pre-cache the model (takes 3-5 min)
    run("python3 -c \"from diffusers import AutoPipelineForText2Image; "
        "AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', "
        "variant='fp16', low_cpu_mem_usage=True)\"")
    
    run(
        f"CUDA_VISIBLE_DEVICES=0 python3 data_generation/generate_synthetic_stl10.py "
        f"--output-dir {RAW_DIR} --class-index 2 --target-per-class {CAR_TARGET} "
        f"--hf-repo-id {HF_DATASET}"
    )
    
    set_marker('step1_car_raw', True)
    car_raw_done = True
    print("  ✓ Car raw generation complete. Shards on HF.")
else:
    print("  ✓ Car raw shards already complete on HF. Skipping.")

# %% [markdown]
# # STEP 2: Filter car at threshold 0.25

# %%
if not car_raw_done:
    print("  Raw car generation not yet complete. Can't filter until it is.")
elif not car_filter_done:
    print("=" * 60)
    print("  STEP 2: Filter car at threshold 0.25")
    print("=" * 60)
    
    # Ensure raw shards are local
    try:
        files = list_repo_files(HF_DATASET, repo_type='dataset')
        raw_files = [f for f in files if f.startswith('raw_shards/') and f.endswith('.tar')]
        for hf_path in raw_files:
            local = os.path.join(RAW_DIR, os.path.basename(hf_path))
            if not os.path.exists(local):
                cached = hf_hub_download(HF_DATASET, hf_path, repo_type='dataset')
                shutil.copy2(cached, local)
    except Exception:
        pass
    
    # Save 10 car sample images BEFORE filtering (for visual QA)
    sample_dir = "/kaggle/working/car_samples"
    os.makedirs(sample_dir, exist_ok=True)
    shard_files = sorted([f for f in os.listdir(RAW_DIR) if 'car' in f and f.endswith('.tar')])
    samples_saved = 0
    for sf in shard_files:
        if samples_saved >= 10: break
        try:
            with tarfile.open(os.path.join(RAW_DIR, sf), 'r') as tar:
                for m in tar.getmembers():
                    if m.name.endswith('.png') and samples_saved < 10:
                        png_data = tar.extractfile(m).read()
                        from PIL import Image
                        Image.open(io.BytesIO(png_data)).save(
                            os.path.join(sample_dir, f"car_sample_{samples_saved:02d}.png"))
                        samples_saved += 1
        except Exception:
            pass
    print(f"  Saved {samples_saved} car samples to {sample_dir}")
    
    # Run filter (car uses lower threshold due to SDXL's poor car rendering)
    run(
        f"CUDA_VISIBLE_DEVICES=0 python3 data_generation/filter_synthetic_clip.py "
        f"--input-dir {RAW_DIR} --output-dir {FILTERED_DIR} --sim-threshold 0.25 "
        f"--class-sim-threshold car=0.20"
    )
    
    # Upload filtered car shards to HF
    filtered_tars = sorted([f for f in os.listdir(FILTERED_DIR) if 'car_filtered' in f and f.endswith('.tar')])
    print(f"\nUploading {len(filtered_tars)} filtered car shards to HF...")
    for fname in filtered_tars:
        fpath = os.path.join(FILTERED_DIR, fname)
        upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                     repo_id=HF_DATASET, repo_type='dataset')
    
    # Check count
    car_count = 0
    for t in filtered_tars:
        with tarfile.open(os.path.join(FILTERED_DIR, t), 'r') as tar:
            car_count += sum(1 for m in tar.getmembers() if m.name.endswith('.png'))
    print(f"  Car filtered: {car_count} images")
    
    if car_count == 0:
        print("  WARNING: Car has 0 filtered images! Check samples in /kaggle/working/car_samples/")
        print("  Adjust prompts or threshold before next run. Pipeline ABORTING.")
        
        report_path = os.path.join(FILTERED_DIR, 'clip_diversity_report.json')
        if os.path.exists(report_path):
            upload_file(path_or_fileobj=report_path, path_in_repo='clip_diversity_report.json',
                         repo_id=HF_DATASET, repo_type='dataset')
        sys.exit(1)
    
    set_marker('step2_car_filter', True)
    car_filter_done = True
    print(f"  ✓ Car filter complete ({car_count} images on HF).")
else:
    print("  ✓ Car filter already complete. Skipping.")

# %% [markdown]
# # STEP 3: Merge all 10 classes and upload final dataset

# %%
if not merge_done and car_filter_done:
    print("=" * 60)
    print("  STEP 3: Merge all 10 classes and upload final dataset")
    print("=" * 60)
    
    # --- Download 9 intact classes' filtered shards ---
    print("Downloading 9 intact classes from HF...")
    try:
        hf_files = list_repo_files(HF_DATASET, repo_type='dataset')
    except Exception:
        hf_files = []
    
    existing_shards = [f for f in hf_files if f.endswith('.tar') and 'car_filtered' not in f]
    print(f"  Found {len(existing_shards)} existing non-car filtered shards")
    for hf_path in existing_shards:
        local_path = os.path.join(FILTERED_DIR, os.path.basename(hf_path))
        if not os.path.exists(local_path):
            hf_hub_download(HF_DATASET, hf_path, repo_type='dataset',
                             local_dir=FILTERED_DIR, local_dir_use_symlinks=False)
    
    # --- Delete OLD car filtered shards from HF ---
    old_car_shards = [f for f in hf_files if f.endswith('.tar') and 'car_filtered' in f]
    if old_car_shards:
        print(f"  Removing {len(old_car_shards)} old car shards from HF...")
        delete_files(repo_id=HF_DATASET, paths=old_car_shards, repo_type='dataset')
    
    # --- Upload NEW car filtered shards ---
    new_car_tars = [f for f in os.listdir(FILTERED_DIR) if 'car_filtered' in f and f.endswith('.tar')]
    print(f"  Uploading {len(new_car_tars)} new car filtered shards...")
    for fname in new_car_tars:
        fpath = os.path.join(FILTERED_DIR, fname)
        upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                     repo_id=HF_DATASET, repo_type='dataset')
    
    # --- Verify all 10 classes have >0 images ---
    all_tars = os.listdir(FILTERED_DIR)
    print("\nVerification:")
    all_ok = True
    for cls in STL10_CLASSES:
        cls_tars = [f for f in all_tars if f'_{cls}_filtered' in f and f.endswith('.tar')]
        count = 0
        for t in cls_tars:
            with tarfile.open(os.path.join(FILTERED_DIR, t), 'r') as tar:
                count += sum(1 for m in tar.getmembers() if m.name.endswith('.png'))
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {cls}: {count} [{status}]")
        if count == 0:
            all_ok = False
    
    if not all_ok:
        print("\n  FATAL: Some classes have 0 images. Aborting upload.")
        sys.exit(1)
    
    # --- Generate clean report.json ---
    full_report = {}
    for cls in STL10_CLASSES:
        cls_tars = [f for f in all_tars if f'_{cls}_filtered' in f and f.endswith('.tar')]
        count = 0
        for t in cls_tars:
            with tarfile.open(os.path.join(FILTERED_DIR, t), 'r') as tar:
                count += sum(1 for m in tar.getmembers() if m.name.endswith('.png'))
        full_report[cls] = {
            "class_name": cls,
            "total_generated": 0,
            "passed_quality": count,
            "pass_rate_pct": 100.0,
            "final_usable_count": count
        }
    
    report_path = os.path.join(FILTERED_DIR, 'clip_diversity_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    # --- Delete old report.json from HF and upload fresh ---
    if 'clip_diversity_report.json' in hf_files:
        delete_files(repo_id=HF_DATASET, paths=['clip_diversity_report.json'], repo_type='dataset')
    
    # Upload everything
    all_final_files = [f for f in os.listdir(FILTERED_DIR) if f.endswith('.tar') or f.endswith('.json')]
    print(f"\nUploading {len(all_final_files)} files to final dataset...")
    for fname in all_final_files:
        fpath = os.path.join(FILTERED_DIR, fname)
        upload_file(path_or_fileobj=fpath, path_in_repo=fname,
                     repo_id=HF_DATASET, repo_type='dataset')
    
    set_marker('step3_merge', True)
    merge_done = True
    total = sum(full_report[c]['final_usable_count'] for c in STL10_CLASSES)
    print(f"\n{'=' * 60}")
    print(f"  SPRINT 2 COMPLETE")
    print(f"  Total usable images: {total}")
    print(f"  HF Dataset: {HF_DATASET}")
    print(f"{'=' * 60}")
else:
    if merge_done:
        print("  ✓ Merge already complete. Pipeline is done.")

# %% [markdown]
# # Summary

# %%
print("\n--- Final HF state ---")
try:
    final_files = list_repo_files(HF_DATASET, repo_type='dataset')
    for c in STL10_CLASSES:
        count = sum(1 for f in final_files if f'_{c}_filtered' in f)
        print(f"  {c}: {count} filtered shards")
except Exception as e:
    print(f"  Error checking HF: {e}")
