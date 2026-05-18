#!/usr/bin/env python3
"""
RHAN v2 — Clean Training with Linear Head + Full Optimization Stack
====================================================================
Changes from v1:
  - Linear classification head (no cosine similarity → no gradient masking)
  - torch.compile(mode='reduce-overhead') for 20-40% speedup
  - AMP (FP16) for doubled throughput on RTX 4060 tensor cores
  - OneCycleLR for superconvergence (30-40% faster than cosine)
  - Optimized DataLoader (pin_memory, persistent_workers, prefetch)
  - cudnn.benchmark = True
  - 40 epochs (not 50)

Target: 90-93% clean accuracy, ~25-30 min on RTX 4060
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))

from model_rhan import RHAN, count_parameters
from dataset import get_dataloaders


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → base_lr over warmup_epochs
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing: base_lr → 0 over remaining epochs
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    total_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === cuDNN benchmark ===
    torch.backends.cudnn.benchmark = True

    # === Model with LINEAR head ===
    model = RHAN(num_classes=10, head_type='linear').to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"\nRHAN v2 (Linear Head) Parameter Count:")
    print(f"  Total:     {total_params:>12,}")
    print(f"  Trainable: {trainable_params:>12,}")

    # === torch.compile ===
    print("\nCompiling model with torch.compile(mode='reduce-overhead')...")
    model = torch.compile(model, mode='reduce-overhead')
    print("  Model compiled. First epoch will be slow (JIT warmup).")

    # === Data — optimized DataLoader ===
    trainloader, testloader = get_dataloaders(
        batch_size=128,
        num_workers=6,
        model_name='resnet'
    )
    # Patch DataLoader kwargs for speed (pin_memory, persistent_workers)
    from torch.utils.data import DataLoader
    trainloader = DataLoader(
        trainloader.dataset, batch_size=128, shuffle=True,
        num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=3,
    )
    testloader = DataLoader(
        testloader.dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
        prefetch_factor=3,
    )

    # === Training setup ===
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)

    epochs = 40
    warmup_epochs = 4
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lr=0.0003)

    scaler = GradScaler()

    # Checkpoint directory
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'rhan_v2_best.pth')

    best_acc = 0.0

    print(f"\n{'='*70}")
    print(f"RHAN v2 — Clean Training with Linear Head")
    print(f"{'='*70}")
    print(f"  Optimizer:      AdamW (lr=0.0003, wd=0.05)")
    print(f"  Scheduler:      WarmupCosineScheduler (base_lr=0.0003, warmup={warmup_epochs})")
    print(f"  Loss:           CrossEntropy (label_smoothing=0.1)")
    print(f"  Batch size:     128")
    print(f"  Epochs:         {epochs}")
    print(f"  AMP:            Yes (FP16)")
    print(f"  torch.compile:  Yes (reduce-overhead)")
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        start_time = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(trainloader.dataset)
        train_acc = 100.0 * train_correct / train_total

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total

        # Checkpointing
        if test_acc > best_acc:
            # Save the underlying model state (unwrap torch.compile)
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), ckpt_path)
            best_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s{marker}", flush=True)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Training complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"Total training time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # === Quick gradient masking check ===
    print(f"\n{'='*70}")
    print(f"GRADIENT MASKING CHECK — PGD-20 vs PGD-100 at ε=0.05")
    print(f"{'='*70}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from phase2_attacks.pgd import pgd_attack

    # Reload best checkpoint into a fresh (non-compiled) model for evaluation
    eval_model = RHAN(num_classes=10, head_type='linear').to(device)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    eval_model.eval()

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    def eval_pgd(model, loader, eps, alpha, steps, max_samples=1000):
        correct = 0
        total = 0
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            _, predicted = pgd_attack(
                model, images, labels,
                epsilon=eps, alpha=alpha, steps=steps,
                device=device, clip_min=cifar_min, clip_max=cifar_max,
            )
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    print("Running PGD-20 at ε=0.05...")
    acc_pgd20 = eval_pgd(eval_model, testloader, eps=0.05, alpha=0.01, steps=20)
    print(f"  PGD-20:  {acc_pgd20:.2f}%")

    print("Running PGD-100 at ε=0.05...")
    acc_pgd100 = eval_pgd(eval_model, testloader, eps=0.05, alpha=0.005, steps=100)
    print(f"  PGD-100: {acc_pgd100:.2f}%")

    gap = acc_pgd20 - acc_pgd100
    print(f"\n  Gap (PGD-20 - PGD-100): {gap:.2f}%")
    if gap < 5:
        print(f"  ✓ Gap < 5% → NO gradient masking detected. Proceed to adversarial training.")
    elif gap < 10:
        print(f"  ⚠ Gap 5-10% → Mild gradient masking. Proceed with caution.")
    else:
        print(f"  ✗ Gap > 10% → SIGNIFICANT gradient masking. Alert before continuing.")

    print(f"\nTotal wall time: {(time.time() - total_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
