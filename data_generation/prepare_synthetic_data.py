#!/usr/bin/env python3
"""
Prepare Synthetic STL-10 Data for Training.

Downloads filtered shards from HuggingFace, extracts images into a uint8
tensor, generates pseudo-labels using the TRADES baseline, and saves as a
single .pt file for efficient training loading.

Usage:
  # Download from HF, pseudo-label, save
  python3 data_generation/prepare_synthetic_data.py \
    --output ./data/synthetic_pt/synthetic_data.pt

  # Skip download if shards are already local
  python3 data_generation/prepare_synthetic_data.py \
    --shard-dir ./data/synthetic_stl10_filtered \
    --output ./data/synthetic_pt/synthetic_data.pt
"""

import os, sys, json, io, tarfile, argparse, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../phase1_training'))

STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(STL10_CLASSES)}

HF_REPO = 'FerrariKazu/stl10-synthetic'


def download_shards(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type='dataset')
    filtered_tars = [f for f in files if 'filtered' in f and f.endswith('.tar')]
    print(f"Downloading {len(filtered_tars)} filtered shards from {HF_REPO}...", flush=True)
    for i, fname in enumerate(sorted(filtered_tars)):
        local_path = os.path.join(target_dir, os.path.basename(fname))
        if os.path.exists(local_path):
            continue
        api.hf_hub_download(repo_id=HF_REPO, filename=fname,
                            repo_type='dataset', local_dir=target_dir,
                            local_dir_use_symlinks=False)
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i+1}/{len(filtered_tars)} shards", flush=True)
    print(f"  Done. {len(filtered_tars)} shards in {target_dir}", flush=True)


def extract_from_shards(shard_dir):
    all_imgs = []
    all_labels = []
    all_keys = []
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith('.tar')])
    for shard_idx, shard_name in enumerate(shard_files):
        cls_name = shard_name.split('_')[2]
        cls_idx = CLASS_TO_IDX[cls_name]
        shard_path = os.path.join(shard_dir, shard_name)
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            png_members = [m for m in members if m.name.endswith('.png')]
            for png_m in png_members:
                key = png_m.name[:-4]
                img_file = tar.extractfile(png_m)
                img_bytes = img_file.read()
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                pil_img = pil_img.resize((96, 96))
                arr = np.array(pil_img, dtype=np.uint8)
                all_imgs.append(arr)
                all_labels.append(cls_idx)
                all_keys.append(key)
        if (shard_idx + 1) % 20 == 0:
            print(f"  Extracted {shard_idx+1}/{len(shard_files)} shards ({len(all_imgs)} images)", flush=True)
    imgs = np.stack(all_imgs)  # (N, 96, 96, 3) uint8
    labels = np.array(all_labels, dtype=np.int64)
    print(f"Extracted {len(imgs)} images. Shape: {imgs.shape}, labels: {labels.shape}", flush=True)
    return imgs, labels, all_keys


@torch.no_grad()
def generate_pseudo_labels(imgs_tensor, labels, model, device, batch_size=64):
    """Generate pseudo-labels using TRADES baseline model."""
    from model_rhan_stl10_large import RHANLargeSTL10
    ckpt_path = 'checkpoints/rhan_stl10_large_pseudolabel_best.pth'

    if model is None:
        print("Loading TRADES Large baseline for pseudo-labeling...", flush=True)
        model = RHANLargeSTL10().to(device)
        # Shim for PyTorch 2.5+ where torch.utils.serialization was removed
        try:
            from torch.utils import serialization as _us
        except ImportError:
            import torch.serialization as _ts
            class _SerialShim:
                StorageType = getattr(_ts, 'StorageType', type)
            sys.modules['torch.utils.serialization'] = _SerialShim()
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model' in state:
            state = state['model']
        elif 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=False)
        model.eval()

    mean = torch.tensor([0.4467, 0.4398, 0.4066], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2603, 0.2566, 0.2713], device=device).view(1, 3, 1, 1)

    N = imgs_tensor.size(0)
    pseudo_labels = torch.full((N,), -1, dtype=torch.long, device='cpu')
    confidences = torch.zeros(N, dtype=torch.float32, device='cpu')
    for i in range(0, N, batch_size):
        batch = imgs_tensor[i:i+batch_size].float().to(device) / 255.0
        batch = (batch - mean) / std
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)
        pseudo_labels[i:i+batch_size] = preds.cpu()
        confidences[i:i+batch_size] = max_probs.cpu()
        if (i // batch_size) % 20 == 0:
            print(f"  Pseudo-labeling: {i}/{N} ({i*100//N}%)", flush=True)
    return pseudo_labels, confidences


def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic STL-10 data")
    parser.add_argument('--shard-dir', type=str, default='./data/synthetic_stl10_filtered',
                        help='Directory with filtered .tar shards (downloads if absent)')
    parser.add_argument('--output', type=str, default='./data/synthetic_pt/synthetic_data.pt',
                        help='Output .pt file path')
    parser.add_argument('--skip-pseudo', action='store_true',
                        help='Skip pseudo-labeling (only extract images)')
    parser.add_argument('--max-class', type=int, default=0,
                        help='Max images per class (0 = all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Step 1: Get shards
    if not os.path.isdir(args.shard_dir) or not any(f.endswith('.tar') for f in os.listdir(args.shard_dir)):
        print(f"Shard dir {args.shard_dir} not found or empty. Downloading from HF...", flush=True)
        download_shards(args.shard_dir)

    # Step 2: Extract images
    t0 = time.time()
    print("Extracting images from shards...", flush=True)
    imgs, labels, keys = extract_from_shards(args.shard_dir)
    print(f"Extraction took {time.time()-t0:.1f}s", flush=True)

    # Subsampling
    if args.max_class > 0:
        from collections import Counter
        cls_counts = Counter(labels)
        keep_mask = np.zeros(len(labels), dtype=bool)
        for cls in range(10):
            cls_indices = np.where(labels == cls)[0]
            np.random.shuffle(cls_indices)
            n_keep = min(args.max_class, len(cls_indices))
            keep_mask[cls_indices[:n_keep]] = True
        imgs = imgs[keep_mask]
        labels = labels[keep_mask]
        keys = [k for i, k in enumerate(keys) if keep_mask[i]]
        print(f"Subsampled to {len(imgs)} images ({args.max_class} per class)", flush=True)

    imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # (N, 3, 96, 96) uint8
    print(f"Image tensor shape: {imgs_tensor.shape}, dtype: {imgs_tensor.dtype}", flush=True)

    # Step 3: Pseudo-labeling
    if not args.skip_pseudo:
        t0 = time.time()
        print("\nGenerating pseudo-labels with TRADES baseline...", flush=True)
        pseudo_labels, confidences = generate_pseudo_labels(imgs_tensor, labels, None, device)
        print(f"Pseudo-labeling took {time.time()-t0:.1f}s", flush=True)
        print(f"  Mean confidence: {confidences.mean():.3f}")
        print(f"  Label agreement with ground-truth class: {(pseudo_labels.numpy() == labels).mean()*100:.1f}%")
    else:
        pseudo_labels = torch.from_numpy(labels)
        confidences = torch.ones(len(labels), dtype=torch.float32)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_dict = {
        'imgs': imgs_tensor,          # (N, 3, 96, 96) uint8
        'labels': pseudo_labels,      # (N,) long
        'confidences': confidences,   # (N,) float
        'gt_labels': torch.from_numpy(labels),
        'keys': keys,
    }
    torch.save(save_dict, args.output)
    print(f"\nSaved to {args.output} ({os.path.getsize(args.output)/1e9:.2f} GB)", flush=True)
    print(f"  Images: {imgs_tensor.shape}")
    print(f"  Labels: {pseudo_labels.shape}")
    print(f"  GT labels: {len(labels)}")
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
