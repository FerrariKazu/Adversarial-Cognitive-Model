#!/usr/bin/env python3
"""PGD-20 eval for RHAN-v5 selfalign checkpoint.
Reads CIFAR-10 directly from cached HF parquet files.
Small batch size to fit in 8GB VRAM."""
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

    # Load model
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

    # Read from cached parquet
    parquet_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--cifar10/snapshots/"
        "0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/test-00000-of-00001.parquet"
    )
    log(f"Reading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    n = len(df)
    log(f"Rows: {n}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Decode ALL images upfront (CPU memory is fine, 10K * 32*32*3 ~ 30MB)
    imgs = []
    labels = []
    t0 = time.time()
    for i in range(n):
        row = df.iloc[i]
        label = int(row['label'])
        img_data = row['img']
        img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_t = transform(img)
        imgs.append(img_t)
        labels.append(label)
    log(f"Decoded {n} images in {time.time()-t0:.1f}s")

    # Clean accuracy (small batches)
    x_all = torch.stack(imgs).to(device)
    y_all = torch.tensor(labels).to(device)
    del imgs, labels

    correct = total = 0
    with torch.no_grad():
        for i in range(0, n, 100):
            xb = x_all[i:i+100]
            yb = y_all[i:i+100]
            logits = wrapper(xb)
            correct += logits.argmax(1).eq(yb).sum().item()
            total += xb.size(0)
    clean_acc = 100. * correct / total
    log(f"Clean accuracy: {clean_acc:.2f}% ({correct}/{total})")

    # PGD-20 attack — batch size 50 to fit in 8GB
    epsilon = 8/255
    alpha = 2/255
    steps = 20
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1,3,1,1).to(device)

    log(f"Running PGD-20 (eps={epsilon:.4f}, bs=50)...")
    t0 = time.time()
    correct_adv = total_adv = 0
    bs = 50

    for i in range(0, n, bs):
        xb = x_all[i:i+bs]
        yb = y_all[i:i+bs]
        B = xb.size(0)

        delta = torch.empty_like(xb).uniform_(-epsilon, epsilon)
        xa = (xb + delta).clamp(cifar_min, cifar_max).detach()

        for _ in range(steps):
            xa.requires_grad_(True)
            loss = F.cross_entropy(wrapper(xa), yb)
            grad = torch.autograd.grad(loss, xa)[0]
            xa = xa.detach() + alpha * grad.sign()
            delta = torch.clamp(xa - xb, -epsilon, epsilon)
            xa = torch.clamp(xb + delta, cifar_min, cifar_max).detach()

        with torch.no_grad():
            preds = wrapper(xa).argmax(1)
            correct_adv += (preds == yb).sum().item()
            total_adv += B

        if (i // bs) % 50 == 0:
            log(f"  [{total_adv}/{n}] acc={100.*correct_adv/total_adv:.2f}% ({time.time()-t0:.0f}s)")

    adv_acc = 100. * correct_adv / total_adv
    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"PGD-20 (eps=8/255): {adv_acc:.2f}% ({correct_adv}/{total_adv})")
    log(f"Clean: {clean_acc:.2f}%  |  Gap: {clean_acc-adv_acc:.2f}pp  |  Time: {elapsed:.0f}s")
    log(f"{'='*60}")

    # Per-class
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    log("\nPer-class PGD-20:")
    for c in range(10):
        mask = y_all == c
        if mask.sum() == 0: continue
        idx = mask.nonzero(as_tuple=True)[0]
        xc, yc = x_all[idx], y_all[idx]
        delta = torch.empty_like(xc).uniform_(-epsilon, epsilon)
        xa = (xc + delta).clamp(cifar_min, cifar_max).detach()
        for _ in range(steps):
            xa.requires_grad_(True)
            loss = F.cross_entropy(wrapper(xa), yc)
            grad = torch.autograd.grad(loss, xa)[0]
            xa = xa.detach() + alpha * grad.sign()
            delta = torch.clamp(xa - xc, -epsilon, epsilon)
            xa = torch.clamp(xc + delta, cifar_min, cifar_max).detach()
        with torch.no_grad():
            ac = (wrapper(xa).argmax(1) == yc).float().mean().item()
        log(f"  {classes[c]:>12s}: {ac*100:.1f}%")

    log("\nDone.")

main()
