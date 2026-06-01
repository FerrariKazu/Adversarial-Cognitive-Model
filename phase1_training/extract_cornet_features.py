#!/usr/bin/env python3
"""
Extract and Cache CORnet-S IT Features for CIFAR-10
=====================================================
One-time preprocessing step: passes CIFAR-10 through CORnet-S and extracts
IT cortex layer features. These serve as the "biological reference" targets
for the alignment loss.

CORnet-S models the primate visual pathway: V1 → V2 → V4 → IT.
We extract features from the IT layer (the highest visual area).

Saves to: checkpoints/cornet_it_features_train.pth
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    output_path = os.path.join(ckpt_dir, 'cornet_it_features_train.pth')

    if os.path.exists(output_path):
        print(f"Features already cached at {output_path}")
        print("Delete the file to re-extract.")
        return

    # Load CORnet-S
    print("Loading CORnet-S...")
    from model_cornets import CIFARCORnet
    cornet_model = CIFARCORnet(num_classes=10).to(device)
    cornet_model.eval()
    for p in cornet_model.parameters():
        p.requires_grad = False

    # CORnet-S needs 224×224 input
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 train set
    print("Loading CIFAR-10 train set...")
    trainset = datasets.CIFAR10(root='/tmp/cifar10_data', train=True,
                                 download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Extract IT features by hooking into CORnet-S
    # CORnet-S structure: model.model.V1, .V2, .V4, .IT
    it_features = []
    hook_handle = None

    def hook_fn(module, input, output):
        # output from IT area: we want the spatial features before the decoder
        it_features.append(output.detach().cpu())

    # Register hook on the IT module
    # CORnet-S uses a recurrent structure; we hook the IT area's output
    print("Registering IT layer hook...")
    try:
        it_module = cornet_model.model.module.IT
        hook_handle = it_module.register_forward_hook(hook_fn)
    except AttributeError:
        # Try alternative path
        print("Trying alternative IT module path...")
        it_module = cornet_model.model.IT
        hook_handle = it_module.register_forward_hook(hook_fn)

    print("Extracting IT features...")
    start = time.time()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(trainloader):
            imgs = imgs.to(device)
            _ = cornet_model(imgs)
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(trainloader)} | "
                      f"Features collected: {sum(f.size(0) for f in it_features)}")

    if hook_handle is not None:
        hook_handle.remove()

    elapsed = time.time() - start
    print(f"Extraction complete in {elapsed:.1f}s")

    if not it_features:
        print("ERROR: No features extracted. Check CORnet-S architecture.")
        return

    # Process features: flatten spatial dims and pool to fixed size
    all_features = torch.cat(it_features, dim=0)
    print(f"Raw IT features shape: {all_features.shape}")

    # If features are spatial (B, C, H, W), global average pool
    if all_features.dim() == 4:
        all_features = all_features.mean(dim=[-2, -1])  # (N, C)
    elif all_features.dim() == 3:
        all_features = all_features.mean(dim=1)  # (N, C)

    print(f"Pooled IT features shape: {all_features.shape}")

    # Project to 512-dim if needed
    if all_features.shape[-1] != 512:
        print(f"Projecting from {all_features.shape[-1]} to 512 dims...")
        proj = nn.Linear(all_features.shape[-1], 512).to(device)
        projected = []
        for i in range(0, all_features.size(0), 1000):
            batch = all_features[i:i+1000].to(device)
            with torch.no_grad():
                projected.append(proj(batch).cpu())
        all_features = torch.cat(projected, dim=0)
        print(f"Projected shape: {all_features.shape}")

    torch.save(all_features, output_path)
    print(f"Saved {all_features.shape[0]} IT features to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
