#!/usr/bin/env python3
"""AutoAttack evaluation for RHAN-v5 selfalign checkpoint.
Single epsilon (8/255) — the standard benchmark."""
import os, sys, time, io
import torch, torch.nn as nn
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

    x_test = torch.stack(imgs).to(device)
    y_test = torch.tensor(labels).to(device)
    del imgs, labels

    # Clean accuracy
    correct = total = 0
    with torch.no_grad():
        for i in range(0, n, 100):
            logits = wrapper(x_test[i:i+100])
            correct += logits.argmax(1).eq(y_test[i:i+100]).sum().item()
            total += min(100, n - i)
    clean_acc = 100. * correct / total
    log(f"Clean accuracy: {clean_acc:.2f}%")

    # AutoAttack at eps=8/255 only
    from autoattack import AutoAttack

    eps = 8/255
    log(f"\n{'='*60}")
    log(f"AutoAttack standard (eps={eps:.4f} = 8/255)")
    log(f"Attacks: APGD-CE + APGD-DLR + FAB + Square")
    log(f"{'='*60}")

    t0 = time.time()
    adversary = AutoAttack(wrapper, norm='Linf', eps=eps, version='standard', device=device, verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
    aa_time = time.time() - t0

    with torch.no_grad():
        logits = wrapper(x_adv)
        preds = logits.argmax(1)
        correct = (preds == y_test).sum().item()
    aa_acc = correct / n

    log(f"\n{'='*60}")
    log(f"AutoAttack (eps=8/255): {aa_acc*100:.2f}% ({correct}/{n})")
    log(f"Clean: {clean_acc:.2f}%  |  Gap: {clean_acc - aa_acc*100:.2f}pp")
    log(f"Time: {aa_time:.1f}s ({aa_time/60:.1f}m)")
    log(f"{'='*60}")

    # Per-class
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    log(f"\nPer-class AutoAttack accuracy:")
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            acc_c = (preds[mask] == y_test[mask]).float().mean().item()
            log(f"  {classes[c]:>12s}: {acc_c*100:.1f}%")

    log(f"\nDone.")

main()
