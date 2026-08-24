#!/usr/bin/env python3
"""Diagnose the GradScaler collapse: compute each loss term on a real batch.

Hypothesis: the TOTAL loss (w_trades*L_trades + w_recon_eff*L_recon + w_hpc*L_hpc)
is NaN while the printed per-batch "Loss:" (L_trades only) looks normal. NaN in any
term -> inf/nan grads -> GradScaler backs off every step -> scale -> 0 -> optimizer
never steps (matches: zero weight change over 15 epochs in the v2 smoke).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from phase1_training.train_rhan_next import dynamic_trades_loss_next

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# Build the model the same way the trainer does (HPC on).
from rhan_core.model import RHANNext
cfg = RHANNextConfig = None
from rhan_core.config.pillar_config import RHANNextConfig
cfg = RHANNextConfig(enable_hpc=True, hpc_num_levels=1, enable_ais=True,
                     enable_sbr=False, enable_iwm=False)
model = RHANNext(cfg)
ckpt = torch.load('checkpoints/rhan_next_ais_v1_halting_only_best.pth',
                  map_location=dev, weights_only=False)
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
print(f'load: {len(missing)} missing, {len(unexpected)} unexpected')
model = model.to(dev).train()

imgs = torch.randn(4, 3, 96, 96, device=dev)
lbls = torch.randint(0, 10, (4,), device=dev)
weights = torch.ones(4, device=dev)
x_adv = torch.randn(4, 3, 96, 96, device=dev)

with torch.autocast('cuda', enabled=(dev == 'cuda')):
    try:
        l_trades, traj_c, traj_a, beta_dyn, l_recon, l_hpc, w_recon_eff = \
            dynamic_trades_loss_next(model, imgs, lbls, weights, x_adv, 2.0,
                                     1.0, 0.10, precision_recon_enabled=True)
        terms = {
            'l_trades': l_trades,
            'l_recon': l_recon,
            'l_hpc': l_hpc,
            'w_recon_eff': w_recon_eff,
        }
        for k, v in terms.items():
            try:
                print(f'{k}: val={float(v.detach())!r} finite={torch.isfinite(v).item()}')
            except Exception as e:
                print(f'{k}: ERROR {e}')
        total = 1.0 * l_trades + w_recon_eff * l_recon + 0.10 * l_hpc
        print(f'TOTAL loss: {float(total.detach())!r} finite={torch.isfinite(total).item()}')
        # NaN component tracing: backward on total, check head + backbone grads
        total.backward()
        hn = sum(p.grad.abs().sum().item() if p.grad is not None else 0.0
                 for n, p in model.named_parameters() if 'hpc' in n)
        bn = sum(p.grad.abs().sum().item() if p.grad is not None else 0.0
                 for n, p in model.named_parameters() if 'hpc' not in n)
        nan_hpc = sum(int(p.grad is not None and torch.isnan(p.grad).any())
                      for n, p in model.named_parameters() if 'hpc' in n)
        nan_back = sum(int(p.grad is not None and torch.isnan(p.grad).any())
                       for n, p in model.named_parameters() if 'hpc' not in n)
        print(f'grad: |hpc|= {hn:.4e} |backbone|= {bn:.4e} '
              f'nan_hpc_params={nan_hpc} nan_backbone_params={nan_back}')
    except Exception as e:
        import traceback
        traceback.print_exc()
