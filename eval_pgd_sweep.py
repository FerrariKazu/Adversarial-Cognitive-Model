#!/usr/bin/env python3
"""PGD-20 epsilon sweep for RHAN-v5 selfalign checkpoint."""
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
    log(f"Reading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    n = len(df)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    log("Decoding images...")
    imgs = []
    labels = []
    t0 = time.time()
    for i in range(n):
        row = df.iloc[i]
        img_data = row['img']
        img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        imgs.append(transform(img))
        labels.append(int(row['label']))
    log(f"Decoded {n} images in {time.time()-t0:.1f}s")

    x_all = torch.stack(imgs).to(device)
    y_all = torch.tensor(labels).to(device)
    del imgs, labels
    torch.cuda.empty_cache()

    # Clean accuracy
    correct = total = 0
    with torch.no_grad():
        for i in range(0, n, 100):
            logits = wrapper(x_all[i:i+100])
            correct += logits.argmax(1).eq(y_all[i:i+100]).sum().item()
            total += min(100, n - i)
    clean_acc = 100. * correct / total
    log(f"Clean accuracy: {clean_acc:.2f}%")
    torch.cuda.empty_cache()

    # Epsilon sweep
    epsilons = [0.00, 0.01, 0.02, 0.031, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    steps = 20
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1,3,1,1).to(device)

    log(f"\n{'='*60}")
    log(f"{'Epsilon':>10s} | {'PGD-20 Acc':>12s} | {'Gap':>8s} | {'Time':>8s}")
    log(f"{'-'*50}")

    for eps in epsilons:
        if eps == 0.0:
            log(f"{'0.000':>10s} | {clean_acc:>11.2f}% | {'0.00':>7s}pp | {'--':>8s}")
            continue

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
        gap = clean_acc - adv_acc
        elapsed = time.time() - t0
        log(f"{eps:>10.3f} | {adv_acc:>11.2f}% | {gap:>7.2f}pp | {elapsed:>7.0f}s")
        torch.cuda.empty_cache()

    log(f"{'='*60}")
    log("\nDone.")

main()
