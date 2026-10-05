#!/usr/bin/env python3
"""PGD-20 eval for RHAN-v5 selfalign checkpoint.
Minimal memory footprint — bs=20, explicit cache cleanup."""
import os, sys, time, io
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))
from model_rhan_v5 import RHANv5

def log(msg):
    print(msg, flush=True)

def pgd_attack(wrapper, xb, yb, epsilon, alpha, steps, cifar_min, cifar_max):
    delta = torch.empty_like(xb).uniform_(-epsilon, epsilon)
    xa = (xb + delta).clamp(cifar_min, cifar_max).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(wrapper(xa), yb)
        grad = torch.autograd.grad(loss, xa)[0]
        xa = xa.detach() + alpha * grad.sign()
        delta = torch.clamp(xa - xb, -epsilon, epsilon)
        xa = torch.clamp(xb + delta, cifar_min, cifar_max).detach()
    return xa

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

    # Read parquet
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
    log(f"Clean accuracy: {clean_acc:.2f}% ({correct}/{total})")
    torch.cuda.empty_cache()

    # PGD-20
    epsilon = 8/255
    alpha = 2/255
    steps = 20
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1,3,1,1).to(device)

    log(f"Running PGD-20 (eps={epsilon:.4f}, bs=20)...")
    t0 = time.time()
    correct_adv = total_adv = 0
    bs = 20

    for i in range(0, n, bs):
        xb = x_all[i:i+bs]
        yb = y_all[i:i+bs]
        xa = pgd_attack(wrapper, xb, yb, epsilon, alpha, steps, cifar_min, cifar_max)
        with torch.no_grad():
            preds = wrapper(xa).argmax(1)
            correct_adv += (preds == yb).sum().item()
            total_adv += xb.size(0)
        if (i // bs) % 100 == 0:
            log(f"  [{total_adv}/{n}] acc={100.*correct_adv/total_adv:.2f}% ({time.time()-t0:.0f}s)")

    adv_acc = 100. * correct_adv / total_adv
    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"PGD-20 (eps=8/255): {adv_acc:.2f}% ({correct_adv}/{total_adv})")
    log(f"Clean: {clean_acc:.2f}%  |  Gap: {clean_acc-adv_acc:.2f}pp  |  Time: {elapsed:.0f}s")
    log(f"{'='*60}")
    torch.cuda.empty_cache()

    # Per-class (very small batches)
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    log("\nPer-class PGD-20:")
    for c in range(10):
        mask = y_all == c
        if mask.sum() == 0: continue
        idx = mask.nonzero(as_tuple=True)[0]
        xc, yc = x_all[idx], y_all[idx]
        c_correct = 0
        for j in range(0, len(xc), 20):
            xb = xc[j:j+20]
            yb = yc[j:j+20]
            xa = pgd_attack(wrapper, xb, yb, epsilon, alpha, steps, cifar_min, cifar_max)
            with torch.no_grad():
                c_correct += (wrapper(xa).argmax(1) == yb).sum().item()
        ac = c_correct / len(xc)
        log(f"  {classes[c]:>12s}: {ac*100:.1f}%")
        torch.cuda.empty_cache()

    log("\nDone.")

main()
