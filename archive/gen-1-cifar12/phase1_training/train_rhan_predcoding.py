#!/usr/bin/env python3
"""
RHAN Predictive Coding — Clean Training
========================================
Stage 1: Train RHAN_PredCoding on clean CIFAR-10.
Initializes from rhan_adv_best.pth (all weights except stem_predictor).
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

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_predcoding import RHAN_PredCoding
from dataset import get_dataloaders


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
    torch.backends.cudnn.benchmark = True

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    rhan_ckpt = os.path.join(ckpt_dir, 'rhan_adv_best.pth')

    if not os.path.exists(rhan_ckpt):
        print(f"ERROR: Base checkpoint not found at {rhan_ckpt}")
        return

    # 1. Initialize model and load weights
    model = RHAN_PredCoding(num_classes=10, embed_dim=512, num_heads=8,
                            ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                            num_recurrent_steps=2).to(device)
    model.load_from_rhan_adv(rhan_ckpt, device=device)

    # 2. DataLoaders
    trainloader_raw, testloader_raw = get_dataloaders(
        batch_size=64, num_workers=4, model_name='resnet'
    )
    from torch.utils.data import DataLoader
    trainloader = DataLoader(
        trainloader_raw.dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2,
    )

    # 3. Optimizer & Scheduler
    epochs = 40
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'rhan_predcoding'))

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0
    output_ckpt = os.path.join(ckpt_dir, 'rhan_predcoding_clean_best.pth')

    print(f"\n{'='*70}")
    print(f"RHAN Predictive Coding — Clean Training")
    print(f"{'='*70}")
    print(f"  Epochs: {epochs} | LR: 3e-4 | Batch: 64 | wd: 0.05")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()

        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, _ = compiled_model(imgs)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()
        N = len(trainloader.dataset)
        epoch_loss /= N
        train_acc = 100.0 * train_correct / train_total

        # Validation
        compiled_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs, _ = compiled_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)

        if test_acc > best_test_acc:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), output_ckpt)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}s{marker}", flush=True)

    total_elapsed = time.time() - total_start
    print(f"\nClean training complete. Best: {best_test_acc:.2f}%")
    print(f"Saved to: {output_ckpt}")
    print(f"Total time: {total_elapsed/60:.1f} min\n")
    tb_writer.close()


if __name__ == '__main__':
    main()
