#!/usr/bin/env python3
"""Find exact PGD-20 eps_thresh for RHAN-v5 selfalign checkpoint.
Tests fine-grained epsilons between 0.30 and 0.50."""
import os, sys, time, io
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))
from model_rhan_v5 import RHANv5

def log(msg):
    print(msg, flush=True)

def main():
    device = torch.device('cuda')
    log(f"Device: {device}")

    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_selfalign_best.pth')
    model = RHANv5().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"Loaded: {ckpt_path}")

    class W(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out
    wrapper = W(model)

    parquet_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--cifar10/snapshots/"
        "0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/test-00000-of-00001.parquet"
    )
    df = pd.read_parquet(parquet_path)
    n = len(df)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    log("Decoding images...")
    imgs = []
    labels = []
    for i in range(n):
        row = df.iloc[i]
        img_data = row['img']
        img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        imgs.append(transform(img))
        labels.append(int(row['label']))
    x_all = torch.stack(imgs).to(device)
    y_all = torch.tensor(labels).to(device)
    del imgs, labels
    torch.cuda.empty_cache()

    # Fine-grained epsilon sweep around the threshold
    # From previous sweep: 0.20->29.84%, 0.30->12.74%
    # Threshold is between 0.30 and 0.50
    epsilons = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    steps = 20
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1,3,1,1).to(device)

    log(f"\n{'='*60}")
    log(f"Fine-grained PGD-20 epsilon sweep")
    log(f"{'='*60}")
    log(f"{'Epsilon':>10s} | {'PGD-20 Acc':>12s} | {'Time':>8s}")
    log(f"{'-'*40}")

    results = []
    for eps in epsilons:
        alpha = max(eps / 10, 0.001)
        t0 = time.time()
        correct_adv = total_adv = 0
        bs = 50

        for i in range(0, n, bs):
            xb = x_all[i:i+bs]
            yb = y_all[i:i+bs]
            B = xb.size(0)

            delta = torch.empty_like(xb).uniform_(-eps, eps)
            xa = (xb + delta).clamp(cifar_min, cifar_max).detach()

            for _ in range(steps):
                xa.requires_grad_(True)
                loss = F.cross_entropy(wrapper(xa), yb)
                grad = torch.autograd.grad(loss, xa)[0]
                xa = xa.detach() + alpha * grad.sign()
                delta = torch.clamp(xa - xb, -eps, eps)
                xa = torch.clamp(xb + delta, cifar_min, cifar_max).detach()

            with torch.no_grad():
                preds = wrapper(xa).argmax(1)
                correct_adv += (preds == yb).sum().item()
                total_adv += B

        adv_acc = 100. * correct_adv / total_adv
        elapsed = time.time() - t0
        log(f"{eps:>10.3f} | {adv_acc:>11.2f}% | {elapsed:>7.0f}s")
        results.append((eps, adv_acc))
        torch.cuda.empty_cache()

    # Find eps_thresh (interpolate where acc = 0)
    log(f"\n{'='*60}")
    for i in range(len(results)-1):
        eps1, acc1 = results[i]
        eps2, acc2 = results[i+1]
        if acc1 > 0 and acc2 == 0:
            # Linear interpolation
            eps_thresh = eps1 + (eps2 - eps1) * (0 - acc1) / (acc2 - acc1)
            log(f"eps_thresh (PGD-20) = {eps_thresh:.4f} ({eps_thresh*255:.1f}/255)")
            break
        elif acc1 == 0:
            log(f"eps_thresh (PGD-20) <= {eps1:.4f} ({eps1*255:.1f}/255)")
            break
    else:
        if results[-1][1] > 0:
            log(f"eps_thresh (PGD-20) > {results[-1][0]:.4f} — need higher epsilons")
        else:
            # Interpolate between last two
            eps1, acc1 = results[-2]
            eps2, acc2 = results[-1]
            if acc1 > 0:
                eps_thresh = eps1 + (eps2 - eps1) * (0 - acc1) / (acc2 - acc1)
                log(f"eps_thresh (PGD-20) = {eps_thresh:.4f} ({eps_thresh*255:.1f}/255)")
            else:
                log(f"Could not determine eps_thresh from sweep range")
    log(f"{'='*60}")
    log("\nDone.")

main()
