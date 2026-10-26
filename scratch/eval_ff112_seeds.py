#!/usr/bin/env python3
"""
Evaluate the three ff_112 training seeds (original seed 17, seed 42, seed 99)
on the reconciled harness (NormalizedModel, eps = 5.417/255, n=1000, seed=42)
to calculate the TRUE training-seed standard deviation (sigma_train).
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchattacks

sys.path.insert(0, ".")
from gen2_foundation.model import Gen2FoundationModel
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

DATA_ROOT = "data/imagenet100"
EVAL_SEED = 42
EVAL_N = 1000
EVAL_BATCH = 32
EPS_PIXEL = 5.417 / 255.0


class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.base_model((x - self.mean) / self.std)


def build_raw_loader(img_size=112):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
    g = torch.Generator().manual_seed(EVAL_SEED)
    indices = torch.randperm(len(val_ds), generator=g)[:EVAL_N].tolist()
    sub_ds = Subset(val_ds, indices)
    return DataLoader(sub_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)


def run_pgd_pixel(wrapped_model, x, y, eps=EPS_PIXEL, steps=50):
    alpha = eps / 4.0
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = wrapped_model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
    return x_adv


def eval_seed(ckpt_path):
    print(f"\nLoading {ckpt_path}...")
    base = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    base.load_state_dict(sd.get("model", sd), strict=False)
    base.eval()

    wrapped = NormalizedModel(base).to(device)
    wrapped.eval()

    loader = build_raw_loader(112)
    apgd_ce = torchattacks.APGD(wrapped, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)

    outcomes = {}
    for atk in ["clean", "pgd50", "apgd_ce"]:
        preds_list, labels_list = [], []
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if atk == "clean":
                x_adv = x
            elif atk == "pgd50":
                x_adv = run_pgd_pixel(wrapped, x, y, eps=EPS_PIXEL, steps=50)
            elif atk == "apgd_ce":
                x_adv = apgd_ce(x, y)
            with torch.no_grad():
                preds = wrapped(x_adv).argmax(-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(y.cpu().numpy().tolist())
        corr = (np.array(preds_list) == np.array(labels_list)).astype(np.int8)
        acc = float(np.mean(corr) * 100.0)
        print(f"  {atk:<10} Acc: {acc:.2f}% ({time.time()-t0:.1f}s)")
        outcomes[atk] = acc

    worst = min(outcomes["pgd50"], outcomes["apgd_ce"])
    outcomes["worst_case"] = worst
    return outcomes


def main():
    seeds_config = {
        "ff_112_seed17": "checkpoints/foundation_ff_112_final.pth",
        "ff_112_seed42": "checkpoints/foundation_ff_112_seed42_final.pth",
        "ff_112_seed99": "checkpoints/foundation_ff_112_seed99_final.pth",
    }

    results = {}
    for name, path in seeds_config.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found! Ensure checkpoint is downloaded.")
            return
        results[name] = eval_seed(path)

    # Compute training seed statistics across the 3 seeds
    metrics = ["clean", "pgd50", "apgd_ce", "worst_case"]
    stats = {}

    print("\n" + "=" * 70)
    print("ff_112 TRAINING SEED VARIANCE (N=3 seeds on identical 1000-image slice)")
    print("=" * 70)
    print(f"  {'Metric':<12} {'Seed 17':>9} {'Seed 42':>9} {'Seed 99':>9} {'Mean':>9} {'Std (sigma)':>12}")
    print("  " + "-" * 66)

    for m in metrics:
        vals = [results[s][m] for s in ["ff_112_seed17", "ff_112_seed42", "ff_112_seed99"]]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        stats[m] = {
            "per_seed": {
                "seed17": vals[0],
                "seed42": vals[1],
                "seed99": vals[2],
            },
            "mean": mean,
            "std_sample": std,
            "two_sigma": 2.0 * std,
        }
        print(f"  {m:<12} {vals[0]:>8.2f}% {vals[1]:>8.2f}% {vals[2]:>8.2f}% {mean:>8.2f}% {std:>10.2f}% (2σ={2*std:.2f}%)")

    out_file = "report/audit_rev1/ff112_training_seed_variance.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved training seed variance to {out_file}")


if __name__ == "__main__":
    main()
