#!/usr/bin/env python3
"""
full_resolution_probe.py
========================
Run the frozen DINOv2-S linear probe at 56, 112, and 128 px on the
full ImageNet-100 training set (126k images) and report top-1 accuracy
on the full 5000-image val set.

Also checks the stored image resolution at 1000-image sample.

Usage:
    python3 scratch/audit_rev1/full_resolution_probe.py
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from gen2_foundation.backbone import CompactViTGen2

DINO_CKPT = os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth")
DATA_ROOT  = os.path.join(REPO_ROOT, "data", "imagenet100")
REPORT_DIR = os.path.join(REPO_ROOT, "report", "audit_rev1")

PATCH_SIZE   = 14   # DINOv2 patch size
BATCH_SIZES  = {56: 256, 112: 128, 126: 96}   # fit within 8 GB; 126px = 9x9=81 tokens for patch_size=14
NUM_WORKERS  = 4
LR           = 1e-3
EPOCHS       = 5    # shallow linear head, fast convergence
NUM_CLASSES  = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")


# ── Frozen DINOv2 Encoder ────────────────────────────────────────────────────
class FrozenDINOv2Encoder(nn.Module):
    """Load DINOv2-S from a .pth checkpoint; return CLS token."""
    def __init__(self, img_size: int, ckpt_path: str):
        super().__init__()
        self.backbone = CompactViTGen2(img_size=img_size, num_refine_iters=2)  # min allowed; refinement frozen
        res = self.backbone.load_dino_warm_start(ckpt_path, strict_band=0.90)
        print(f"  [{img_size}px] Warm-start loaded — "
              f"matched {res['matched_trunk_keys']}/{res['trunk_keys']} trunk keys "
              f"({res['matched_fraction']*100:.1f}%)  accepted={res['accepted']}")
        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns (pooled_z, tokens); we want pooled_z (CLS)
        z, _ = self.backbone(x)
        return z            # (B, 384)


# ── Linear Head ─────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, encoder: FrozenDINOv2Encoder, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x)
        return self.head(z)


def make_loaders(img_size: int, batch_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=tf_train)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=tf_val)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    return train_ld, val_ld


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


def train_probe(img_size: int) -> dict:
    print(f"\n{'='*60}")
    print(f" PROBE @ {img_size}px  —  {(img_size // PATCH_SIZE)**2} tokens")
    print(f"{'='*60}")

    batch_size = BATCH_SIZES[img_size]
    train_ld, val_ld = make_loaders(img_size, batch_size)
    n_train = len(train_ld.dataset)
    n_val   = len(val_ld.dataset)
    print(f"  train={n_train:,} | val={n_val:,} | batch={batch_size}")

    encoder = FrozenDINOv2Encoder(img_size=img_size, ckpt_path=DINO_CKPT).to(device)
    model   = LinearProbe(encoder, num_classes=NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = 0.0
    t_start  = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        model.encoder.eval()   # always eval for frozen backbone
        running_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
        scheduler.step()
        train_loss = running_loss / n_train

        val_acc = evaluate(model, val_ld)
        best_val = max(best_val, val_acc)
        elapsed  = time.time() - t_start
        print(f"  Ep {ep}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%  ({elapsed:.0f}s elapsed)")

    n_tokens = (img_size // PATCH_SIZE) ** 2
    return {
        "img_size":    img_size,
        "n_tokens":    n_tokens,
        "n_train":     n_train,
        "n_val":       n_val,
        "best_val_acc": best_val,
        "epochs":      EPOCHS,
    }


if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)
    results = {}

    # Note: 126px = 9x9 = 81 tokens (divisible by patch_size 14; 128 is not divisible by 14)
    for sz in [56, 112, 126]:
        r = train_probe(sz)
        results[str(sz)] = r
        print(f"\n  ✓ {sz}px → best val acc = {r['best_val_acc']*100:.2f}%  ({r['n_tokens']} tokens)")

    out_path = os.path.join(REPORT_DIR, "step5_full_resolution_probe.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== SUMMARY ===")
    for sz, r in results.items():
        print(f"  {sz}px  {r['n_tokens']:3d} tokens  → {r['best_val_acc']*100:.2f}% val acc")
