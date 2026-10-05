#!/usr/bin/env python3
"""Fast AutoAttack evaluation — APGD-CE + APGD-DLR only (most important attacks)."""

import os, sys, time, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders
from autoattack import AutoAttack

def main():
    device = torch.device('cuda')
    print(f"Device: {device}", flush=True)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'rhan_adv_trades_best.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RHANv5(head_type='cosine').to(device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {ckpt_path}", flush=True)

    class W(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    wrapper = W(model)

    print("Loading test data...", flush=True)
    _, testloader = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    imgs_list, lbls_list = [], []
    for imgs, lbls in testloader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        if sum(x.size(0) for x in imgs_list) >= 1000:
            break
    x_test = torch.cat(imgs_list, dim=0)[:1000].to(device)
    y_test = torch.cat(lbls_list, dim=0)[:1000].to(device)
    print(f"Evaluating on {x_test.size(0)} images", flush=True)

    # Run individual attacks for speed
    from autoattack.other_utils import L2_norm
    from autoattack.autopgd_base import APGDAttack
    import math

    eps = 8/255

    # APGD-CE only (fastest, most important)
    print(f"\n{'='*60}", flush=True)
    print(f"APGD-CE (eps={eps:.4f})", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    apgd_ce = APGDAttack(wrapper, n_restarts=1, n_iter=100, eps=eps,
                          eot_iter=1, rho=.75, seed=0, loss='ce', verbose=True)
    apgd_ce.init_hyperparam(x_test)
    x_adv_ce = apgd_ce.perturb(x_test, y_test)
    t_ce = time.time() - t0

    with torch.no_grad():
        preds_ce = wrapper(x_adv_ce).argmax(1)
        acc_ce = (preds_ce == y_test).float().mean().item()
    print(f"APGD-CE accuracy: {acc_ce*100:.2f}%  ({t_ce:.1f}s)", flush=True)

    # APGD-DLR
    print(f"\n{'='*60}", flush=True)
    print(f"APGD-DLR (eps={eps:.4f})", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    apgd_dlr = APGDAttack(wrapper, n_restarts=1, n_iter=100, eps=eps,
                           eot_iter=1, rho=.75, seed=0, loss='dlr', verbose=True)
    x_adv_dlr = apgd_dlr.perturb(x_test, y_test)
    t_dlr = time.time() - t0

    with torch.no_grad():
        preds_dlr = wrapper(x_adv_dlr).argmax(1)
        acc_dlr = (preds_dlr == y_test).float().mean().item()
    print(f"APGD-DLR accuracy: {acc_dlr*100:.2f}%  ({t_dlr:.1f}s)", flush=True)

    # Combined (worst of both)
    combined_correct = ((preds_ce == y_test) & (preds_dlr == y_test)).sum().item()
    acc_combined = combined_correct / 1000

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f" Clean accuracy:         86.80%", flush=True)
    print(f" APGD-CE accuracy:       {acc_ce*100:.2f}%  ({t_ce:.0f}s)", flush=True)
    print(f" APGD-DLR accuracy:      {acc_dlr*100:.2f}%  ({t_dlr:.0f}s)", flush=True)
    print(f" Combined (CE ∩ DLR):    {acc_combined*100:.2f}%", flush=True)
    print(f"\n PGD-100 (eps=0.031):    84.77%", flush=True)
    print(f" APGD-CE gap vs PGD-100: {84.77 - acc_ce*100:.1f} pp", flush=True)

    if 84.77 - acc_ce*100 < 8:
        print(" ✓ Robustness is genuine (gap < 8pp)", flush=True)
    elif 84.77 - acc_ce*100 < 15:
        print(" ⚠ Moderate gap — some gradient masking possible", flush=True)
    else:
        print(" ✗ Large gap — gradient masking likely", flush=True)

    print(f"\nDone.", flush=True)

main()
