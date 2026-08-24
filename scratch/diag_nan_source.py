#!/usr/bin/env python3
"""Isolate which loss term produces NaN backbone gradients."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = RHANNextConfig(enable_hpc=True, hpc_num_levels=1, enable_ais=True,
                     enable_sbr=False, enable_iwm=False)
model = RHANNext(cfg)
ckpt = torch.load('checkpoints/rhan_next_ais_v1_halting_only_best.pth',
                  map_location=dev, weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model = model.to(dev).train()

imgs = torch.randn(4, 3, 96, 96, device=dev)
lbls = torch.randint(0, 10, (4,), device=dev)

def nan_counts():
    nn_ = sum(int(p.grad is not None and torch.isnan(p.grad).any())
              for n, p in model.named_parameters() if 'hpc' not in n)
    nh = sum(int(p.grad is not None and torch.isnan(p.grad).any())
             for n, p in model.named_parameters() if 'hpc' in n)
    inf_nn = sum(int(p.grad is not None and torch.isinf(p.grad).any())
                 for n, p in model.named_parameters() if 'hpc' not in n)
    return nn_, nh, inf_nn

with torch.autocast('cuda', enabled=(dev == 'cuda')):
    # Term 1: l_trades CE on clean
    model.zero_grad(set_to_none=True)
    logits_c, traj_c = model(imgs, return_trajectory=True)
    F.cross_entropy(logits_c, lbls).backward()
    nn_, nh, inf_ = nan_counts()
    print(f'l_trades (CE clean):          nan_backbone={nn_} nan_hpc={nh} inf_backbone={inf_}')

    # Term 2: l_recon
    model.zero_grad(set_to_none=True)
    logits_c, traj_c = model(imgs, return_trajectory=True)
    l = model.get_reconstruction_loss(imgs, (logits_c, traj_c))
    print(f'  l_recon value: {float(l.detach()):.6f} finite={torch.isfinite(l).item()}')
    l.backward()
    nn_, nh, inf_ = nan_counts()
    print(f'l_recon:                      nan_backbone={nn_} nan_hpc={nh} inf_backbone={inf_}')

    # Term 3: l_hpc
    model.zero_grad(set_to_none=True)
    logits_c, traj_c = model(imgs, return_trajectory=True)
    l = model.get_hpc_loss(imgs, (logits_c, traj_c))
    print(f'  l_hpc value: {float(l.detach()):.6f} finite={torch.isfinite(l).item()}')
    l.backward()
    nn_, nh, inf_ = nan_counts()
    print(f'l_hpc:                         nan_backbone={nn_} nan_hpc={nh} inf_backbone={inf_}')

    # Term 4: full warmup loss (CE + recon + hpc)
    model.zero_grad(set_to_none=True)
    logits_c, traj_c = model(imgs, return_trajectory=True)
    l = (F.cross_entropy(logits_c, lbls)
         + model.get_reconstruction_loss(imgs, (logits_c, traj_c))
         + model.get_hpc_loss(imgs, (logits_c, traj_c)))
    print(f'  warmup total value: {float(l.detach()):.6f} finite={torch.isfinite(l).item()}')
    l.backward()
    nn_, nh, inf_ = nan_counts()
    print(f'warmup total (CE+recon+hpc):  nan_backbone={nn_} nan_hpc={nh} inf_backbone={inf_}')
