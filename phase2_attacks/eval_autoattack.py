#!/usr/bin/env python3
"""AutoAttack evaluation for RHAN-v5-TRADES."""

import os, sys, time, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders
from autoattack import AutoAttack

def main():
    device = torch.device('cuda')
    print(f"Device: {device}", flush=True)

    # Load model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'rhan_adv_trades_best.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RHANv5(head_type='cosine').to(device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {ckpt_path}", flush=True)

    # Logit-only wrapper
    class W(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    wrapper = W(model)

    # Load 1000 test images via DataLoader
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

    # AutoAttack
    print(f"\n{'='*60}", flush=True)
    print(f"AutoAttack standard (eps=8/255={8/255:.4f})", flush=True)
    print(f"Attacks: APGD-CE + APGD-DLR + FAB + Square", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    adversary = AutoAttack(wrapper, norm='Linf', eps=8/255, version='standard', device=device, verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
    aa_time = time.time() - t0

    with torch.no_grad():
        logits = wrapper(x_adv)
        preds = logits.argmax(1)
        correct = (preds == y_test).sum().item()
    aa_acc = correct / 1000

    print(f"\n{'='*60}", flush=True)
    print(f"AutoAttack accuracy (eps=8/255): {aa_acc*100:.2f}% ({correct}/1000)", flush=True)
    print(f"Time: {aa_time:.1f}s ({aa_time/60:.1f}m)", flush=True)

    # Gap analysis
    pgd100_at_eps031 = 84.77  # from training eval
    gap = pgd100_at_eps031 - aa_acc * 100
    print(f"\nPGD-100 (eps=0.031): {pgd100_at_eps031}%", flush=True)
    print(f"AutoAttack (eps=0.031): {aa_acc*100:.2f}%", flush=True)
    print(f"Gap: {gap:.1f} pp", flush=True)
    if gap < 8:
        print("Robustness is genuine (gap < 8pp)", flush=True)
    elif gap < 15:
        print("Moderate gap — some gradient masking possible", flush=True)
    else:
        print("Large gap — gradient masking likely", flush=True)

    # Per-class
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print(f"\nPer-class AutoAttack accuracy:", flush=True)
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            acc_c = (preds[mask] == y_test[mask]).float().mean().item()
            print(f"  {classes[c]:>12s}: {acc_c*100:.1f}%", flush=True)

    print(f"\nDone.", flush=True)

main()
