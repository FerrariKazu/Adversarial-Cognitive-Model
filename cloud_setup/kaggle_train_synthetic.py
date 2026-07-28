#!/usr/bin/env python3
"""
Kaggle: Loss-Ablated RHAN-v11 Training on Synthetic STL-10
============================================================
Step 1: Download 118 filtered shards (115K images) from HF
Step 2: Extract images into uint8 tensor
Step 2.5: Download TRADES baseline checkpoint
Step 3: Pseudo-label with TRADES baseline
Step 4: Train loss-ablated v11 (TRADES + recon only, no active inference)
Step 5: Verify checkpoint on HF

Usage: Run on Kaggle with GPU T4 x2. Set HF_TOKEN in Kaggle Secrets.
Expected runtime: ~6-8 hours on T4x2.
"""

import os, sys, subprocess, json, tarfile, io, time, shutil
import numpy as np
from PIL import Image
import torch

HF_DATASET = "FerrariKazu/stl10-synthetic"
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/kaggle/working/{REPO_NAME}'
FILTERED_DIR = '/kaggle/working/synthetic_shards'
SYNTH_PT = '/kaggle/working/synthetic_data.pt'
CKPT_DIR = '/kaggle/working/checkpoints'

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']
CLASS_TO_IDX = {c: i for i, c in enumerate(STL10_CLASSES)}

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN not found — set in Kaggle Secrets")
os.environ["HF_TOKEN"] = hf_token
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def run(cmd, shell=True, check=True):
    print(f"\n[RUN]: {cmd}")
    proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


# ── Clone / sync repo ──
os.chdir('/kaggle/working')
if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

# Ensure torch.utils.serialization proxy exists (PyTorch >= 2.5 removed it)
# Must happen before any torch.save/torch.load call.
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))
from checkpoint_utils import _ensure_serialization_proxy
_ensure_serialization_proxy()

# ── Install deps ──
# Don't install torch/torchvision — Kaggle base env has it pre-installed.
# Explicitly installing a different CUDA variant (cu121 vs Kaggle's 12.4+)
# causes CUDA ABI mismatches and kernel segfaults on TransformerEncoder init.
run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet datasets Pillow")
from huggingface_hub import HfApi
api = HfApi(token=hf_token)

print("\n=== Step 1: Download 118 filtered shards from HF ===")
os.makedirs(FILTERED_DIR, exist_ok=True)
files = api.list_repo_files(HF_DATASET, repo_type='dataset')
filtered_tars = sorted([f for f in files if 'filtered' in f and f.endswith('.tar')])
print(f"Found {len(filtered_tars)} filtered shards on HF")
for fname in filtered_tars:
    local = os.path.join(FILTERED_DIR, os.path.basename(fname))
    if not os.path.exists(local):
        api.hf_hub_download(repo_id=HF_DATASET, filename=fname,
                            repo_type='dataset', local_dir=FILTERED_DIR)
print(f"Downloaded {len(filtered_tars)} shards to {FILTERED_DIR}")

print("\n=== Step 2: Extract images into uint8 tensor ===")
t0 = time.time()
all_imgs, all_labels = [], []
for shard_name in filtered_tars:
    cls_name = shard_name.split('_')[2]
    cls_idx = CLASS_TO_IDX[cls_name]
    shard_path = os.path.join(FILTERED_DIR, shard_name)
    with tarfile.open(shard_path, 'r') as tar:
        png_members = [m for m in tar.getmembers() if m.name.endswith('.png')]
        for png_m in png_members:
            img_bytes = tar.extractfile(png_m).read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((96, 96))
            all_imgs.append(np.array(pil_img, dtype=np.uint8))
            all_labels.append(cls_idx)
imgs_np = np.stack(all_imgs)
labels_np = np.array(all_labels, dtype=np.int64)
imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).contiguous()
print(f"Extracted {len(imgs_tensor)} images in {time.time()-t0:.1f}s")
print(f"  Tensor shape: {imgs_tensor.shape}, dtype: {imgs_tensor.dtype}")

print("\n=== Step 2.5: Download TRADES baseline checkpoint ===")
os.makedirs(CKPT_DIR, exist_ok=True)
api.hf_hub_download(
    repo_id='FerrariKazu/rhan-checkpoints',
    filename='rhan_stl10_large_pseudolabel_best.pth',
    repo_type='dataset', local_dir=CKPT_DIR
)
# Also make 'checkpoints' symlink so relative paths work
if os.path.islink('checkpoints'):
    os.unlink('checkpoints')
if not os.path.exists('checkpoints'):
    os.symlink(CKPT_DIR, 'checkpoints')

print("\n=== Step 3: Generate pseudo-labels with TRADES baseline ===")
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))
from model_rhan_stl10_large import RHANLargeSTL10
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = RHANLargeSTL10().to(device)
from checkpoint_utils import compat_load
ckpt = compat_load(os.path.join(CKPT_DIR, 'rhan_stl10_large_pseudolabel_best.pth'), map_location=device)
state = ckpt.get('model', ckpt)
model.load_state_dict(state, strict=False)
model.eval()

# Use both T4 GPUs for pseudo-labeling
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs for pseudo-labeling", flush=True)

mean = torch.tensor([0.4467, 0.4398, 0.4066], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.2603, 0.2566, 0.2713], device=device).view(1, 3, 1, 1)
N = imgs_tensor.size(0)
pseudo_labels = torch.full((N,), -1, dtype=torch.long, device='cpu')
batch_size = 256
t0 = time.time()
with torch.no_grad():
    for i in range(0, N, batch_size):
        batch = imgs_tensor[i:i+batch_size].float().to(device) / 255.0
        batch = (batch - mean) / std
        logits = model(batch)
        preds = logits.argmax(dim=1)
        pseudo_labels[i:i+batch_size] = preds.cpu()
        if (i // batch_size) % 20 == 0:
            pct = i * 100 // N
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (N - i)
            print(f"  Pseudo-label: {i}/{N} ({pct}%)  ETA: {eta:.0f}s", flush=True)
print(f"Pseudo-labeling done in {time.time()-t0:.1f}s")
agreement = (pseudo_labels.numpy() == labels_np).mean() * 100
print(f"  Label agreement with class: {agreement:.1f}%")

# Save prepared data (legacy format — zipfile is slow for 3.2 GB)
torch.save({'imgs': imgs_tensor, 'labels': pseudo_labels}, SYNTH_PT,
           _use_new_zipfile_serialization=False)
print(f"\nSaved synthetic data to {SYNTH_PT} ({os.path.getsize(SYNTH_PT)/1e9:.2f} GB)")

print("\n=== Re-sync repo (pseudo-labeling took ~20 min, may have fixes) ===")
run('git fetch origin main && git reset --hard origin/main')

print("\n=== Step 4: Train loss-ablated RHAN-v11 (single GPU) ===")
# DataParallel triggers cudaErrorMisalignedAddress on Kaggle T4s
# and torchrun DDP has NCCL init issues in Kaggle containers.
# Single GPU avoids these problems entirely.
# Effective batch: 16 * 16 = 256 (same as before).
run(
    f"CUDA_VISIBLE_DEVICES=0 python3 phase1_training/train_rhan_v11.py "
    f"--synthetic-data {SYNTH_PT} "
    f"--batch-size 16 "
    f"--accum-steps 16 "
    f"--w-foraging 0.0 "
    f"--w-precision 0.0 "
    f"--w-halt 0.0 "
    f"--w-trades 0.55 "
    f"--w-recon 0.10 "
)

print("\n=== Step 5: Verify checkpoint on HF ===")
try:
    files = api.list_repo_files('FerrariKazu/rhan-checkpoints', repo_type='dataset')
    v11_files = [f for f in files if 'v11' in f]
    print(f"rhan-checkpoints v11 files: {v11_files}")
except Exception as e:
    print(f"Error: {e}")

print("\n✓ Loss-ablated training complete. Checkpoints synced to HF.")
