#!/usr/bin/env python3
"""Executable facts behind docs/research/RHAN_IM_01_FORENSIC_AUDIT.md (checks on RANDOM-INIT models; no weights needed).

1. parameter bookkeeping of the ff_112 and recurrence_112 phases (state_dict vs parameters(), aliased patch embed)
2. parameters that receive NO gradient after forward+backward (the unused refinement block / gist gate)
3. whether a 112-px "glimpse" of a 112-px frame can contain any pixel that is not in the frame
   (foveal_sample at window == frame is a shifted, border-replicated copy), versus a real 112-px crop of a
   224-px canvas, which is a pure native-resolution slice

Usage: python scripts/rhan_im/forensic_checks.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, collections, sys
torch.manual_seed(0)
from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import foveal_sample

def count(model, name):
    sd = model.state_dict()
    sd_numel = sum(v.numel() for v in sd.values())
    uniq = {}
    for k, v in sd.items():
        uniq.setdefault(v.data_ptr(), []).append(k)
    uniq_numel = sum(sd[ks[0]].numel() for ks in uniq.values())
    aliased = [ks for ks in uniq.values() if len(ks) > 1]
    params = list(model.parameters())
    p_numel = sum(p.numel() for p in params)
    rg = sum(p.numel() for p in params if p.requires_grad)
    print(f"[{name}] state_dict numel={sd_numel:,}  unique-storage numel={uniq_numel:,}  parameters() numel={p_numel:,}  requires_grad numel={rg:,}")
    print(f"   aliased groups: {aliased}")
    print(f"   diff state_dict - parameters() = {sd_numel - p_numel:,}  (patch-embed weight+bias = {384*3*14*14 + 384:,})")
    return sd_numel, p_numel

def dead_params(model, name, x, phase_forward=None):
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x)
    out.float().pow(2).mean().backward()
    dead = [(n, p.numel()) for n, p in model.named_parameters() if p.grad is None]
    groups = collections.OrderedDict()
    for n, k in dead:
        root = n.split('.')[0] + '.' + n.split('.')[1] if n.startswith('backbone.refinement') else n
        groups[root] = groups.get(root, 0) + k
    tot = sum(k for _, k in dead)
    print(f"[{name}] params with NO gradient after fwd+bwd: {len(dead)} tensors, {tot:,} values")
    for n, p in list(groups.items())[:6]:
        print(f"      {n}: {p:,}")
    # refinement-only count
    ref = sum(p.numel() for n, p in model.named_parameters() if n.startswith('backbone.refinement'))
    print(f"   refinement block total = {ref:,} values")

x112 = torch.randn(2, 3, 112, 112)
m_ff = Gen2FoundationModel(phase="ff_112", fovea_size=112)
count(m_ff, "ff_112")
dead_params(m_ff, "ff_112", x112)
m_rec = Gen2FoundationModel(phase="recurrence_only", fovea_size=112)
count(m_rec, "recurrence_112 (arch_phase=recurrence_only)")
dead_params(m_rec, "recurrence_112", x112)

# ---- Is a 112-px 'glimpse' of a 112-px frame new information?
print("\n--- foveal_sample at H=112, fovea=112 (the flagship configuration)")
x = torch.randn(1, 3, 112, 112)
g0 = foveal_sample(x, torch.zeros(1, 2), fovea_size=112)
print("gaze (0,0): max|crop - frame| =", float((g0 - x).abs().max()))
for g in [(0.3, 0.3), (-0.3, 0.3), (0.0, -0.3)]:
    gz = torch.tensor([g])
    c = foveal_sample(x, gz, fovea_size=112)
    # fraction of output pixels whose source coordinate lies outside the frame (border-replicated)
    shift_px = [abs(v) * 56 for v in g]
    outside = 1 - (1 - shift_px[0] / 112) * (1 - shift_px[1] / 112)
    print(f"gaze {g}: shift = {shift_px[0]:.1f}px,{shift_px[1]:.1f}px -> {100*outside:.1f}% of the crop is border-replicated; "
          f"no pixel outside the original 112x112 frame is ever sampled")
# genuine zoom case for contrast: 224 canvas, 112 window
x224 = torch.randn(1, 3, 224, 224)
c = foveal_sample(x224, torch.zeros(1, 2), fovea_size=112)
ref = x224[..., 56:168, 56:168]
print("\n--- contrast: canvas 224, window 112 at centre: max|crop - native 112 centre crop| =", float((c - ref).abs().max()), "(native-resolution crop of a larger frame)")
