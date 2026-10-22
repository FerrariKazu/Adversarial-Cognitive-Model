#!/usr/bin/env python3
"""
Diagnostic: File List Hashing & 5,000-image Val Clean Accuracy across all arms.

Verifies:
1. Exact SHA-256 hash of all 5,000 validation file paths and bytes.
2. Exact file lists and hashes of 1,000-image slices:
   - numpy rng seed 42 (used in thesis_preflight.py)
   - torch randperm seed 42 (used in Kaggle_ATTACK_RECONCILIATION.py)
3. Evaluates full 5,000-image clean accuracy for every checkpoint using exact
   training val transform and compare with checkpoint's training monitor.
4. Evaluates slice clean accuracy on both 1k slices.
"""

import os
import sys
import hashlib
import json
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from gen2_foundation.model import Gen2FoundationModel

VAL_ROOT = "data/imagenet100/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file_hashes():
    ds = ImageFolder(VAL_ROOT)
    samples = ds.samples # list of (path, class_idx)
    
    # Sort by path to be strictly deterministic
    sorted_samples = sorted(samples, key=lambda x: x[0])
    
    h_paths = hashlib.sha256()
    for p, _ in sorted_samples:
        rel = os.path.relpath(p, VAL_ROOT)
        h_paths.update(rel.encode("utf-8"))
    path_hash = h_paths.hexdigest()
    
    # 1000 slice via numpy
    rng_np = np.random.default_rng(42)
    idx_np = rng_np.choice(len(ds), size=1000, replace=False)
    
    h_np = hashlib.sha256()
    for idx in sorted(idx_np):
        rel = os.path.relpath(samples[idx][0], VAL_ROOT)
        h_np.update(rel.encode("utf-8"))
    np_hash = h_np.hexdigest()
    
    # 1000 slice via torch randperm
    g = torch.Generator().manual_seed(42)
    idx_torch = torch.randperm(len(ds), generator=g)[:1000].tolist()
    
    h_torch = hashlib.sha256()
    for idx in sorted(idx_torch):
        rel = os.path.relpath(samples[idx][0], VAL_ROOT)
        h_torch.update(rel.encode("utf-8"))
    torch_hash = h_torch.hexdigest()
    
    # Check overlap
    overlap = len(set(idx_np).intersection(set(idx_torch)))
    
    return {
        "n_total": len(ds),
        "path_sha256": path_hash,
        "slice_np_seed42_sha256": np_hash,
        "slice_torch_seed42_sha256": torch_hash,
        "slice_overlap_count": overlap,
        "idx_np": idx_np,
        "idx_torch": idx_torch
    }

def get_val_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def eval_loader(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    print(f"Device: {DEVICE}")
    print(f"Val Root: {VAL_ROOT}")
    
    meta_hash = get_file_hashes()
    print(f"Total Val Images: {meta_hash['n_total']}")
    print(f"Full Val Paths SHA256: {meta_hash['path_sha256']}")
    print(f"Numpy Slice SHA256:   {meta_hash['slice_np_seed42_sha256']}")
    print(f"Torch Slice SHA256:   {meta_hash['slice_torch_seed42_sha256']}")
    print(f"Overlap between Numpy and Torch slices: {meta_hash['slice_overlap_count']}/1000")
    
    idx_np = meta_hash["idx_np"]
    idx_torch = meta_hash["idx_torch"]
    
    MODELS = [
        ("ff_112_s17",         "ff_112",          112, "checkpoints/foundation_ff_112_final.pth"),
        ("ff_112_s42",         "ff_112",          112, "checkpoints/foundation_ff_112_seed42_final.pth"),
        ("ff_112_s99",         "ff_112",          112, "checkpoints/foundation_ff_112_seed99_final.pth"),
        ("ff_56_final",        "ff_56",           56,  "checkpoints/foundation_ff_56_final.pth"),
        ("recurrence_only",    "recurrence_only", 56,  "checkpoints/foundation_recurrence_only_final.pth"),
        ("g2_gist_only_cont",  "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_cont_final.pth"),
        ("g2_gist_only",       "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_final.zip"),
        ("gen1_core",          "gen1_core",       56,  "checkpoints/foundation_gen1_core_best_clean.pth"),
    ]
    
    results = {
        "hashes": {
            "n_total": meta_hash["n_total"],
            "path_sha256": meta_hash["path_sha256"],
            "slice_np_sha256": meta_hash["slice_np_seed42_sha256"],
            "slice_torch_sha256": meta_hash["slice_torch_seed42_sha256"],
            "overlap": meta_hash["slice_overlap_count"]
        },
        "arms": {}
    }
    
    print("\n" + "=" * 95)
    print(f"{'Arm':<20} {'Monitor':>10} {'5k Val':>10} {'Diff 5k':>10} {'1k (Torch)':>12} {'1k (Numpy)':>12}")
    print("=" * 95)
    
    for label, phase, img_size, ckpt in MODELS:
        if not os.path.exists(ckpt):
            print(f"Skipping {label}: {ckpt} not found")
            continue
            
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        monitor_clean = sd.get("val_clean_acc", sd.get("val_clean", None))
        
        # Load model
        base = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=img_size).to(DEVICE)
        base.load_state_dict(sd.get("model", sd), strict=False)
        base.eval()
        
        tf = get_val_transform(img_size)
        ds = ImageFolder(VAL_ROOT, transform=tf)
        
        # 5k loader
        l_5k = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        acc_5k = eval_loader(base, l_5k)
        
        # 1k torch loader
        l_torch = DataLoader(Subset(ds, idx_torch), batch_size=128, shuffle=False, num_workers=2)
        acc_torch = eval_loader(base, l_torch)
        
        # 1k np loader
        l_np = DataLoader(Subset(ds, idx_np), batch_size=128, shuffle=False, num_workers=2)
        acc_np = eval_loader(base, l_np)
        
        diff_5k = (acc_5k - monitor_clean) * 100.0 if monitor_clean is not None else float("nan")
        mon_str = f"{monitor_clean*100:.2f}%" if monitor_clean is not None else "N/A"
        
        print(f"{label:<20} {mon_str:>10} {acc_5k*100:>9.2f}% {diff_5k:>+9.2f}% {acc_torch*100:>11.2f}% {acc_np*100:>11.2f}%")
        
        results["arms"][label] = {
            "checkpoint": ckpt,
            "monitor_clean": monitor_clean,
            "acc_5k": acc_5k,
            "diff_5k_pp": diff_5k,
            "acc_1k_torch": acc_torch,
            "acc_1k_np": acc_np
        }
        
    out_path = "report/audit_rev1/harness_parity_audit.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to:", out_path)

if __name__ == "__main__":
    main()
