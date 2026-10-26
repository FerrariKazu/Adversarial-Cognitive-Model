#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = "/home/ferrarikazu/Adversarial Cognitive Model"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.train_generation1_foundation import FoundationConfig, build_model
from evaluation.imagenet100_loader import make_imagenet100_loaders
from evaluation.clean_and_robust import generic_pgd

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.path.join(REPO_ROOT, "data", "imagenet100")
    loaders = make_imagenet100_loaders(data_root, batch_size=64, num_workers=2)
    val_loader = loaders["val"]
    val_dataset = val_loader.dataset
    print(f"Full val dataset size: {len(val_dataset)}")

    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "foundation_backbone_only_rolling.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, "recovery_artifacts", "checkpoints", "foundation_backbone_only_rolling_fef50f3_epoch49.pth")
    
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint epoch: {state.get('epoch')}, commit: {state.get('code_commit')}")

    cfg = FoundationConfig()
    model = build_model(cfg, phase="backbone_only").to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # 1. Full clean validation accuracy
    correct_clean = 0
    total_clean = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct_clean += (preds == y).sum().item()
            total_clean += y.numel()
    clean_top1 = correct_clean / total_clean

    # 2. Robust top-1 on a fixed 512-image subset
    g = torch.Generator().manual_seed(41)
    indices = torch.randperm(len(val_dataset), generator=g)[:512].tolist()
    subset_loader = DataLoader(Subset(val_dataset, indices), batch_size=64, shuffle=False)

    eps = 0.094  # epoch 49 curriculum point
    steps = 10   # pgd_steps for epoch 49
    correct_robust = 0
    total_robust = 0
    for x, y in subset_loader:
        x, y = x.to(device), y.to(device)
        x_adv = generic_pgd(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=-1)
            correct_robust += (preds == y).sum().item()
            total_robust += y.numel()
    robust_top1 = correct_robust / total_robust

    print("=" * 50)
    print(f"clean top-1:  {clean_top1 * 100:.2f}% ({clean_top1:.4f}) [total={total_clean}]")
    print(f"robust top-1: {robust_top1 * 100:.2f}% ({robust_top1:.4f}) [eps={eps}, steps={steps}, total={total_robust}]")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()
