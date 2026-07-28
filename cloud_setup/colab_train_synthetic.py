#!/usr/bin/env python3
"""
Colab: Loss-Ablated RHAN-v11 Training on Synthetic STL-10
==========================================================
Step 1: Download 118 filtered shards (115K images) from HF
Step 2: Extract images into uint8 tensor
Step 2.5: Download TRADES baseline checkpoint
Step 3: Pseudo-label with TRADES baseline
Step 4: Train loss-ablated v11 (TRADES + recon only, no active inference)
Step 5: Verify checkpoint on HF

Usage: Copy cells into Colab. Set HF_TOKEN in Colab Secrets.
Expected runtime: ~8-10 hours on T4 (single GPU).
"""

# %% [markdown]
# # RHAN-v11 Loss-Ablated Training on Synthetic STL-10
#
# Trains v11 with active inference losses zeroed (TRADES + recon only)
# on the Sprint 2 synthetic dataset (115K images).
# Auto-resumes from HuggingFace rolling checkpoint if available.

# %% [markdown]
# ## Step 1: Install dependencies

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
run("pip install --quiet huggingface_hub datasets Pillow")

# %% [markdown]
# ## Step 2: Clone / sync repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'
FILTERED_DIR = '/content/synthetic_shards'
SYNTH_PT = '/content/synthetic_data.pt'
CKPT_DIR = '/content/checkpoints'

STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']
CLASS_TO_IDX = {c: i for i, c in enumerate(STL10_CLASSES)}

if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))

from checkpoint_utils import _ensure_serialization_proxy
_ensure_serialization_proxy()

# %% [markdown]
# ## Step 3: Set HF_TOKEN

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
    raise RuntimeError(
        "HF_TOKEN not found. Set it in Colab Secrets (key icon in sidebar)."
    )
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
print(f"HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from huggingface_hub import HfApi
api = HfApi(token=hf_token)

# %% [markdown]
# ## Step 4: Download 118 filtered shards from HF

# %%
print("\n=== Step 4: Download 118 filtered shards ===")
os.makedirs(FILTERED_DIR, exist_ok=True)
files = api.list_repo_files('FerrariKazu/stl10-synthetic', repo_type='dataset')
filtered_tars = sorted([f for f in files if 'filtered' in f and f.endswith('.tar')])
print(f"Found {len(filtered_tars)} filtered shards")
for fname in filtered_tars:
    local = os.path.join(FILTERED_DIR, os.path.basename(fname))
    if not os.path.exists(local):
        api.hf_hub_download(repo_id='FerrariKazu/stl10-synthetic', filename=fname,
                            repo_type='dataset', local_dir=FILTERED_DIR)
print(f"Downloaded {len(filtered_tars)} shards")

# %% [markdown]
# ## Step 5: Extract images into uint8 tensor

# %%
print("\n=== Step 5: Extract images ===")
t0 = time.time()
all_imgs, all_labels = [], []
for shard_name in filtered_tars:
    cls_name = shard_name.split('_')[2]
    cls_idx = CLASS_TO_IDX[cls_name]
    import tarfile, io
    from PIL import Image
    shard_path = os.path.join(FILTERED_DIR, shard_name)
    with tarfile.open(shard_path, 'r') as tar:
        for m in tar.getmembers():
            if not m.name.endswith('.png'):
                continue
            img_bytes = tar.extractfile(m).read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((96, 96))
            all_imgs.append(pil_img)
            all_labels.append(cls_idx)

import numpy as np
imgs_np = np.stack([np.array(img, dtype=np.uint8) for img in all_imgs])
labels_np = np.array(all_labels, dtype=np.int64)
imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).contiguous()
print(f"Extracted {len(imgs_tensor)} images in {time.time()-t0:.1f}s")
print(f"  Shape: {imgs_tensor.shape}, dtype: {imgs_tensor.dtype}")

# %% [markdown]
# ## Step 6: Download TRADES baseline checkpoint

# %%
print("\n=== Step 6: Download TRADES baseline checkpoint ===")
import torch.nn as nn
os.makedirs(CKPT_DIR, exist_ok=True)
api.hf_hub_download(
    repo_id='FerrariKazu/rhan-checkpoints',
    filename='rhan_stl10_large_pseudolabel_best.pth',
    repo_type='dataset', local_dir=CKPT_DIR
)
if os.path.islink('checkpoints'):
    os.unlink('checkpoints')
if not os.path.exists('checkpoints'):
    os.symlink(CKPT_DIR, 'checkpoints')

# %% [markdown]
# ## Step 7: Generate pseudo-labels

# %%
print("\n=== Step 7: Pseudo-label with TRADES baseline ===")
from model_rhan_stl10_large import RHANLargeSTL10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RHANLargeSTL10().to(device)
from checkpoint_utils import compat_load
ckpt = compat_load(os.path.join(CKPT_DIR, 'rhan_stl10_large_pseudolabel_best.pth'), map_location=device)
model.load_state_dict(ckpt.get('model', ckpt), strict=False)
model.eval()

mean = torch.tensor([0.4467, 0.4398, 0.4066], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.2603, 0.2566, 0.2713], device=device).view(1, 3, 1, 1)
N = imgs_tensor.size(0)
pseudo_labels = torch.full((N,), -1, dtype=torch.long, device='cpu')
batch_size = 128
t0 = time.time()
with torch.no_grad():
    for i in range(0, N, batch_size):
        batch = imgs_tensor[i:i+batch_size].float().to(device) / 255.0
        batch = (batch - mean) / std
        logits = model(batch)
        pseudo_labels[i:i+batch_size] = logits.argmax(dim=1).cpu()
        if (i // batch_size) % 20 == 0:
            pct = i * 100 // N
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (N - i)
            print(f"  {i}/{N} ({pct}%)  ETA: {eta:.0f}s", flush=True)
print(f"Done in {time.time()-t0:.1f}s")
agreement = (pseudo_labels.numpy() == labels_np).mean() * 100
print(f"  Label agreement: {agreement:.1f}%")

# Save
torch.save({'imgs': imgs_tensor, 'labels': pseudo_labels}, SYNTH_PT)
print(f"Saved {SYNTH_PT} ({os.path.getsize(SYNTH_PT)/1e9:.2f} GB)")

# %% [markdown]
# ## Step 8: Train loss-ablated v11
#
# Auto-resumes from HF rolling checkpoint if available
# (no --force-restart).

# %%
print("\n=== Step 8: Train ===")
run(
    f"python3 phase1_training/train_rhan_v11.py "
    f"--synthetic-data {SYNTH_PT} "
    f"--batch-size 16 "
    f"--accum-steps 16 "
    f"--w-foraging 0.0 "
    f"--w-precision 0.0 "
    f"--w-halt 0.0 "
    f"--w-trades 0.55 "
    f"--w-recon 0.10 "
)

# %% [markdown]
# ## Step 9: Verify checkpoints on HF

# %%
print("\n=== Step 9: Verify ===")
try:
    for repo in ['FerrariKazu/rhan-checkpoints', 'FerrariKazu/rhan-checkpoints-rolling']:
        files = api.list_repo_files(repo, repo_type='dataset')
        v11_files = [f for f in files if 'v11' in f]
        print(f"  {repo}: {v11_files}")
except Exception as e:
    print(f"  Error: {e}")

# %% [markdown]
# ## Prevent Colab Disconnect
# Open browser console (Ctrl+Shift+I) → Console tab → paste:
# ```javascript
# function KeepAlive() {
#     document.querySelector("colab-connect-button").click();
# }
# setInterval(KeepAlive, 60000);
# ```
