#!/usr/bin/env python3
"""Quick test: can we load the selfalign checkpoint?"""
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))

ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_selfalign_best.pth')
print(f"Checkpoint exists: {os.path.exists(ckpt_path)}")
print(f"Checkpoint size: {os.path.getsize(ckpt_path) / 1e6:.1f} MB")

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    print(f"Keys: {list(ckpt.keys())}")
    if 'model' in ckpt:
        ckpt = ckpt['model']
    print(f"Num parameters: {len(ckpt)}")
    for k, v in sorted(ckpt.items()):
        print(f"  {k}: {v.shape}")
else:
    print(f"Type: {type(ckpt)}")

# Try loading into model
from model_rhan_v5 import RHANv5
model = RHANv5()
try:
    model.load_state_dict(ckpt, strict=True)
    print("\nLoaded with strict=True — perfect match!")
except RuntimeError as e:
    print(f"\nstrict=True failed: {e}")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:10]}")
    if unexpected:
        print(f"  Unexpected: {unexpected[:10]}")

# Quick forward test
import torch.nn.functional as F
x = torch.randn(2, 3, 32, 32)
try:
    logits, features = model.forward_with_features(x)
    print(f"\nForward OK: logits={logits.shape}, features={features.shape}")
except Exception as e:
    print(f"\nForward failed: {e}")
