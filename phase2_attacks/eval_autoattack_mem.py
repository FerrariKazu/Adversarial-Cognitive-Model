#!/usr/bin/env python3
"""Memory-efficient AutoAttack evaluation for RHAN-v5-TRADES."""

import os, sys, time, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1_training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, '..'))))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders

def main():
    # Memory optimization settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    print(f"Device: {device}", flush=True)
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'rhan_adv_trades_best.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = RHANv5(head_type='cosine')
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    del ckpt
    torch.cuda.empty_cache()
    print(f"Loaded: {ckpt_path}", flush=True)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    print(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    # Wrapper returning logits only
    class W(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    # Enable gradient checkpointing to save memory during attack
    # This recomputes forward passes during backward but saves massive memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    # Also try to enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    wrapper = W(model)

    # Load test data in smaller batches to save memory
    print("Loading test data...", flush=True)
    _, testloader = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')

    n_eval = 1000
    imgs_list, lbls_list = [], []
    for imgs, lbls in testloader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        if sum(x.size(0) for x in imgs_list) >= n_eval:
            break
    x_test = torch.cat(imgs_list, dim=0)[:n_eval].to(device, non_blocking=True)
    y_test = torch.cat(lbls_list, dim=0)[:n_eval].to(device, non_blocking=True)
    print(f"Evaluating on {x_test.size(0)} images", flush=True)
    print(f"GPU memory with data: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    from autoattack import AutoAttack

    eps = 8 / 255

    print(f"\n{'='*60}", flush=True)
    print(f"AutoAttack (standard, eps={eps:.4f})", flush=True)
    print(f"Attacks: APGD-CE + APGD-DLR + FAB + Square", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    # Run with smaller batch size to avoid OOM
    adversary = AutoAttack(
        wrapper,
        norm='Linf',
        eps=eps,
        version='standard',
        device=device,
        verbose=True
    )

    # Run attack in chunks if needed
    try:
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=64)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM with bs=64, retrying with bs=32...", flush=True)
            torch.cuda.empty_cache()
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=32)
        else:
            raise

    aa_time = time.time() - t0

    # Evaluate
    with torch.no_grad():
        logits = wrapper(x_adv)
        preds = logits.argmax(1)
        correct = (preds == y_test).sum().item()
    aa_acc = correct / n_eval

    print(f"\n{'='*60}", flush=True)
    print(f"AUTOATTACK RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f" AutoAttack accuracy (eps=8/255): {aa_acc*100:.2f}% ({correct}/{n_eval})", flush=True)
    print(f" Time: {aa_time:.1f}s ({aa_time/60:.1f}m)", flush=True)

    # Gap analysis
    pgd100_031 = 84.77
    pgd100_05 = 65.82
    gap = pgd100_031 - aa_acc * 100

    print(f"\nCOMPARISON", flush=True)
    print(f" PGD-100   (eps=0.031): {pgd100_031}%", flush=True)
    print(f" PGD-100   (eps=0.050): {pgd100_05}%", flush=True)
    print(f" AutoAttack (eps=0.031): {aa_acc*100:.2f}%", flush=True)
    print(f" Gap PGD-100 vs AA:      {gap:.1f} pp", flush=True)

    if gap < 8:
        print(f" -> Robustness is GENUINE (gap < 8pp)", flush=True)
    elif gap < 15:
        print(f" -> Moderate gap — some masking possible", flush=True)
    else:
        print(f" -> Large gap — gradient masking likely", flush=True)

    # Per-class
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print(f"\nPer-class accuracy:", flush=True)
    for c in range(10):
        mask = y_test == c
        n = mask.sum().item()
        if n > 0:
            acc_c = (preds[mask] == y_test[mask]).float().mean().item()
            print(f"  {classes[c]:>12s}: {acc_c*100:.1f}%  (n={n})", flush=True)

    print(f"\nDone.", flush=True)

main()
