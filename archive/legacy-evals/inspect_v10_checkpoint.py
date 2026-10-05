#!/usr/bin/env python3
"""
Diagnose v10_final checkpoint: identify all 48 unmatched keys,
determine whether they're renamed layers or a different architecture.
"""

import torch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'phase1_training')

from model_rhan_stl10_large import RHANLargeSTL10
from model_rhan_v10 import RHANv10

ckpt_path = 'checkpoints/rhan_stl10_v10_best.pth'
device = 'cpu'

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
if isinstance(ckpt, dict):
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif 'model' in ckpt and isinstance(ckpt['model'], dict):
        sd = ckpt['model']
    else:
        sd = ckpt
else:
    sd = ckpt

if all(k.startswith('model.') for k in list(sd.keys())[:5]):
    sd = {k[len('model.'):]: v for k, v in sd.items()}

print(f"Checkpoint keys: {len(sd)}")
print()

# ── Check against RHANLargeSTL10 ──
print("=" * 70)
print("  RHANLARGE STL10 (55.6M base)")
print("=" * 70)
base = RHANLargeSTL10()
base_sd = base.state_dict()
base_keys = set(base_sd.keys())
ckpt_keys = set(sd.keys())

common = base_keys & ckpt_keys
missing_in_ckpt = base_keys - ckpt_keys
extra_in_ckpt = ckpt_keys - base_keys

print(f"  Common keys:          {len(common)}")
print(f"  Missing in checkpoint: {len(missing_in_ckpt)}")
print(f"  Extra in checkpoint:   {len(extra_in_ckpt)}")

if extra_in_ckpt:
    print(f"\n  Extra keys (in ckpt but not in RHANLargeSTL10):")
    for k in sorted(extra_in_ckpt):
        v = sd[k]
        print(f"    {k}  shape={tuple(v.shape)}")

if missing_in_ckpt:
    print(f"\n  Missing keys (in RHANLargeSTL10 but not in ckpt):")
    for k in sorted(missing_in_ckpt):
        v = base_sd[k]
        print(f"    {k}  shape={tuple(v.shape)}")

print()

# ── Check against RHANv10 ──
print("=" * 70)
print("  RHANV10 (55.6M + 4M active inference)")
print("=" * 70)
v10 = RHANv10()
v10_sd = v10.state_dict()
v10_keys = set(v10_sd.keys())

common_v10 = v10_keys & ckpt_keys
missing_in_ckpt_v10 = v10_keys - ckpt_keys
extra_in_ckpt_v10 = ckpt_keys - v10_keys

print(f"  Common keys:            {len(common_v10)}")
print(f"  Missing in checkpoint:   {len(missing_in_ckpt_v10)}")
print(f"  Extra in checkpoint:     {len(extra_in_ckpt_v10)}")

if extra_in_ckpt_v10:
    print(f"\n  Extra keys (in ckpt but not in RHANv10):")
    for k in sorted(extra_in_ckpt_v10):
        v = sd[k]
        print(f"    {k}  shape={tuple(v.shape)}")

if missing_in_ckpt_v10:
    print(f"\n  Missing keys (in RHANv10 but not in ckpt):")
    for k in sorted(missing_in_ckpt_v10):
        v = v10_sd[k]
        print(f"    {k}  shape={tuple(v.shape)}")

print()

# ── Check if extra keys have a prefix/stem pattern (like v11 preview) ──
print("=" * 70)
print("  PATTERN ANALYSIS")
print("=" * 70)
extra = sorted(extra_in_ckpt_v10)
if extra:
    prefixes = set(k.split('.')[0] for k in extra)
    print(f"  Prefixes of extra keys: {sorted(prefixes)}")
    for p in sorted(prefixes):
        pk = [k for k in extra if k.startswith(p)]
        print(f"    {p}: {len(pk)} keys")
        for k in pk[:5]:
            print(f"      {k}  {tuple(sd[k].shape)}")
        if len(pk) > 5:
            print(f"      ... and {len(pk)-5} more")

# ── Attempt a shape-based mapping heuristic ──
print("\n  Shape-based mapping candidates (extra → likely base layer):")
for ek in extra[:10]:
    ev = sd[ek]
    # Find base key with same shape
    candidates = [(bk, base_sd[bk]) for bk in base_keys if base_sd[bk].shape == ev.shape]
    if candidates:
        print(f"    {ek} {tuple(ev.shape)} → {[c[0] for c in candidates]}")
    else:
        print(f"    {ek} {tuple(ev.shape)} → (no shape match)")
