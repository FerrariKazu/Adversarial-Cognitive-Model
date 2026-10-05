#!/usr/bin/env python3
"""
Build synthetic_stl10_115k_tradeslabels.pt from the 118 filtered HF shards.
Reproduces kaggle_train_synthetic.py Steps 1-3 exactly:
  - download filtered shards (115K images)
  - extract to a uint8 (N,3,96,96) tensor
  - pseudo-label every image with the TRADES Large baseline
  - save as .pt (legacy format) and upload to FerrariKazu/stl10-synthetic
"""
import io
import os
import re
import sys
import time
import tarfile
import shutil

import numpy as np
import torch
from PIL import Image

REPO = "FerrariKazu/stl10-synthetic"
OUT = "scratch/synthetic_stl10_115k_tradeslabels.pt"
SHARD_DIR = "scratch/synth_shards"

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']
CLASS_TO_IDX = {c: i for i, c in enumerate(STL10_CLASSES)}

env = open('.env').read()
tok = re.search(r'HF_TOKEN="?([^"\n]+)"?', env).group(1)
os.environ["HF_TOKEN"] = tok
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=tok)

# ── 1. Download filtered shards ──────────────────────────────────────────────
os.makedirs(SHARD_DIR, exist_ok=True)
files = api.list_repo_files(REPO, repo_type='dataset')
filtered_tars = sorted([f for f in files if 'filtered' in f and f.endswith('.tar')])
print(f"Found {len(filtered_tars)} filtered shards", flush=True)
for fname in filtered_tars:
    local = os.path.join(SHARD_DIR, os.path.basename(fname))
    if not os.path.exists(local):
        hf_hub_download(repo_id=REPO, filename=fname,
                        repo_type='dataset', local_dir=SHARD_DIR)
print("Shards downloaded.", flush=True)

# ── 2. Extract to uint8 tensor ───────────────────────────────────────────────
t0 = time.time()
total_imgs = 0
for shard_name in filtered_tars:
    with tarfile.open(os.path.join(SHARD_DIR, shard_name), 'r') as tar:
        total_imgs += sum(1 for m in tar.getmembers() if m.name.endswith('.png'))

imgs_tensor = torch.empty((total_imgs, 3, 96, 96), dtype=torch.uint8)
class_labels = torch.empty(total_imgs, dtype=torch.long)
idx = 0
for shard_name in filtered_tars:
    cls_name = shard_name.split('_')[2]
    cls_idx = CLASS_TO_IDX[cls_name]
    with tarfile.open(os.path.join(SHARD_DIR, shard_name), 'r') as tar:
        for m in tar.getmembers():
            if not m.name.endswith('.png'):
                continue
            img = Image.open(io.BytesIO(tar.extractfile(m).read()))
            img = img.convert('RGB').resize((96, 96))
            imgs_tensor[idx] = torch.from_numpy(
                np.array(img, dtype=np.uint8)).permute(2, 0, 1)
            class_labels[idx] = cls_idx
            idx += 1
print(f"Extracted {len(imgs_tensor)} images in {time.time()-t0:.1f}s "
      f"shape={imgs_tensor.shape}", flush=True)

# ── 3. Pseudo-label with TRADES Large ────────────────────────────────────────
sys.path.insert(0, 'phase1_training')
from model_rhan_stl10_large import RHANLargeSTL10
from checkpoint_utils import compat_load

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RHANLargeSTL10().to(dev).eval()
ckpt = compat_load('checkpoints/rhan_stl10_large_pseudolabel_best.pth', map_location=dev)
state = ckpt.get('model', ckpt)
model.load_state_dict(state, strict=False)
print(f"TRADES Large loaded on {dev}", flush=True)

mean = torch.tensor([0.4467, 0.4398, 0.4066], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.2603, 0.2566, 0.2713], device=dev).view(1, 3, 1, 1)
N = imgs_tensor.size(0)
pseudo_labels = torch.full((N,), -1, dtype=torch.long)
bs = 64
t0 = time.time()
with torch.no_grad():
    for i in range(0, N, bs):
        batch = imgs_tensor[i:i+bs].float().to(dev) / 255.0
        batch = (batch - mean) / std
        logits = model(batch)
        pseudo_labels[i:i+bs] = logits.argmax(dim=1).cpu()
        if (i // bs) % 50 == 0:
            pct = i * 100 // N
            print(f"  pseudo-label {i}/{N} ({pct}%)", flush=True)
print(f"Pseudo-labeling done in {time.time()-t0:.1f}s", flush=True)
agreement = (pseudo_labels == class_labels).float().mean().item() * 100
print(f"  Label agreement with class: {agreement:.1f}%", flush=True)

# ── 4. Save (legacy format, like the original synthetic_data.pt) ─────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({'imgs': imgs_tensor, 'labels': pseudo_labels, 'class_labels': class_labels},
           OUT, _use_new_zipfile_serialization=False)
print(f"Saved {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB)", flush=True)

# ── 5. Upload to HF ──────────────────────────────────────────────────────────
api.upload_file(path_or_fileobj=OUT,
                path_in_repo=os.path.basename(OUT),
                repo_id=REPO, repo_type='dataset', token=tok)
print("Uploaded to HF:", os.path.basename(OUT), flush=True)

# cleanup shards
shutil.rmtree(SHARD_DIR, ignore_errors=True)
print("DONE", flush=True)
