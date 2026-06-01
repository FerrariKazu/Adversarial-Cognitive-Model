#!/usr/bin/env python3
"""
RHAN-Split Fine-Tuning (Trial 3)
================================
Fine-tunes the RHANSplit model (Ventral/Dorsal Stream Split)
using 50% clean and 50% PGD-5 mixed adversarial training.
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Silence TF32 warning
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_split import RHANSplit
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


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

    # Optimize GPU operations
    torch.backends.cudnn.benchmark = True

    # Checkpoint paths
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    rhan_ckpt = os.path.join(ckpt_dir, 'rhan_adv_best.pth')

    if not os.path.exists(rhan_ckpt):
        print(f"ERROR: Base RHAN-adv checkpoint not found at {rhan_ckpt}")
        return

    # 1. Initialize RHANSplit and load base weights
    model = RHANSplit(head_type='cosine').to(device)
    print("Loading base weights from RHAN-adv checkpoint...")
    rhan_state = torch.load(rhan_ckpt, map_location=device)
    filtered_state = {k: v for k, v in rhan_state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"  Loaded {len(filtered_state)} layers from {rhan_ckpt}.")
    print(f"  Missing (initialized new): {missing}")

    # 2. Setup DataLoaders
    trainloader_raw, testloader_raw = get_dataloaders(
        batch_size=64, num_workers=4, model_name='resnet'
    )
    trainloader = DataLoader(
        trainloader_raw.dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
        prefetch_factor=2,
    )

    # CIFAR bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # 3. Setup Optimizer & Scheduler
    epochs = 40
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # TensorBoard setup
    tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'rhan_split'))

    # Compile model
    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0
    output_ckpt_path = os.path.join(ckpt_dir, 'rhan_split_best.pth')

    print(f"\n{'='*70}")
    print(f"RHAN Ventral/Dorsal Split Fine-Tuning (Trial 3)")
    print(f"{'='*70}")
    print(f"  Base checkpoint:   {rhan_ckpt}")
    print(f"  Optimizer:         AdamW (lr=5e-5, wd=0.05)")
    print(f"  Scheduler:         CosineAnnealingLR (T_max={epochs})")
    print(f"  Batch size:        64")
    print(f"  Epochs:            {epochs}")
    print(f"  Adv Training:      50% PGD-5 (eps=0.031) + 50% clean")
    print(f"  AMP & Compile:     Enabled")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()
        
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Mixed adversarial training: 50% PGD-5, 50% clean ──
            B = imgs.size(0)
            half = B // 2
            if half > 0:
                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        compiled_model, imgs[:half], labels[:half],
                        epsilon=0.031, alpha=0.031/4, steps=5,
                        device=device, clip_min=cifar_min, clip_max=cifar_max,
                        random_start=True
                    )
                mixed_imgs = torch.cat([adv_imgs.detach(), imgs[half:]], dim=0)
                mixed_labels = torch.cat([labels[:half], labels[half:]], dim=0)
            else:
                mixed_imgs = imgs
                mixed_labels = labels

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = compiled_model(mixed_imgs)
                loss = F.cross_entropy(logits, mixed_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * B
            _, predicted = logits.max(1)
            train_total += mixed_labels.size(0)
            train_correct += predicted.eq(mixed_labels).sum().item()

        scheduler.step()

        # Normalize metrics
        N = len(trainloader.dataset)
        epoch_loss /= N
        train_acc = 100.0 * train_correct / train_total

        # Evaluation
        compiled_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs = compiled_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total

        # Logging to TensorBoard
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)

        if test_acc >= best_test_acc:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), output_ckpt_path)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.2f}% | "
              f"LR: {current_lr:.7f} | Time: {elapsed:.1f}s{marker}", flush=True)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Training complete. Model saved successfully to: {output_ckpt_path}")
    print(f"Total training time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")
    tb_writer.close()


if __name__ == '__main__':
    main()
