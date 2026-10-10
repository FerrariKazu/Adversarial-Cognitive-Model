#!/usr/bin/env python3
"""Gate W Probe: Linear evaluation of frozen DINOv2 trunk at 56x56 gist view.
================================================================================

Freezes the DINOv2-small trunk (with LayerScale and interpolated pos_embed),
attaches a fresh linear head, and trains 5 epochs on ImageNet-100 at 56x56.
Threshold: clean val top-1 > 31.3%.
"""

from __future__ import annotations

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.backbone import CompactViTGen2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GATE_W_THRESHOLD = 0.313


class GistLinearProbe(nn.Module):
    def __init__(self, trunk: CompactViTGen2, num_classes: int = 100):
        super().__init__()
        self.trunk = trunk
        # Freeze trunk
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.head = nn.Linear(trunk.d_z, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full 56x56 image passed through trunk
        x_prep = self.trunk._prep(x)
        x_out = self.trunk._trunk_forward(x_prep)
        cls_feat = x_out[:, 0]  # (B, D_z)
        return self.head(cls_feat)


def get_loaders(data_root: str, batch_size: int = 128, num_workers: int = 4):
    # Gist view: full image resized to 56x56
    tf_train = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf_train)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def run_gate_w(
    ckpt_path: str = "checkpoints/dinov2_vits14_pretrain.pth",
    data_root: str = "data/imagenet100",
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gate W Probe running on: {device}")

    trunk = CompactViTGen2(img_size=56, use_layer_scale=True)
    report = trunk.load_dino_warm_start(ckpt_path)
    print(f"Warm-start load report: accepted={report['accepted']}, "
          f"matched={report['matched_trunk_keys']}/{report['trunk_keys']} "
          f"({report['matched_fraction']*100:.1f}%)")

    model = GistLinearProbe(trunk, num_classes=100).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders(data_root, batch_size=batch_size)
    print(f"Dataset: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=-1) == y).sum().item()
            train_total += y.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(x)
                val_correct += (logits.argmax(dim=-1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s): "
              f"train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%, "
              f"best_val={best_val_acc*100:.2f}%")

    passed = best_val_acc > GATE_W_THRESHOLD
    print("=" * 60)
    print(f"GATE W RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"Required threshold: > {GATE_W_THRESHOLD*100:.1f}%")
    print(f"Achieved clean top-1: {best_val_acc*100:.2f}%")
    print("=" * 60)
    return passed, best_val_acc


if __name__ == "__main__":
    passed, acc = run_gate_w()
    if not passed:
        sys.exit(1)
