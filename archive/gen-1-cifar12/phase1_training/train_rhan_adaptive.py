#!/usr/bin/env python3
"""
RHAN with Adaptive Computation Time Fine-Tuning (Trial 2)
=========================================================

Mixed adversarial training (50% PGD-5 + 50% clean) with ponder cost:
  total_loss = ce_loss + 0.01 * ponder_cost
where:
  ponder_cost = mean(steps_used) per batch

This enforces efficiency on easy inputs (clean) while preserving 
robustness on harder inputs by dynamically adjusting recurrent cycles.
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

# Silence TF32 warning and optimise matmul
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_adaptive import AdaptiveRHAN
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

    # 1. Initialize AdaptiveRHAN model and load weights
    model = AdaptiveRHAN(max_steps=6, epsilon_halt=0.01).to(device)
    model.load_from_rhan_adv(rhan_ckpt, device=device)

    # 2. Setup DataLoaders (batch size 64 as requested)
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

    # 3. Setup Optimizer & Scheduler
    epochs = 40
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # TensorBoard setup
    tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'rhan_adaptive'))

    # Compile model using torch.compile
    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0
    output_ckpt_path = os.path.join(ckpt_dir, 'rhan_adaptive_best.pth')

    # CIFAR normalisation bounds for PGD clamping
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # Wrapper that returns only logits (needed for pgd_attack)
    class LogitsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _, _ = self.m(x)
            return logits

    print(f"\n{'='*70}")
    print(f"RHAN Adaptive Computation Time Fine-Tuning (Trial 2)")
    print(f"{'='*70}")
    print(f"  Base checkpoint:   {rhan_ckpt}")
    print(f"  Optimizer:         AdamW (lr=5e-5, wd=0.05)")
    print(f"  Scheduler:         CosineAnnealingLR (T_max={epochs})")
    print(f"  Batch size:        64")
    print(f"  Epochs:            {epochs}")
    print(f"  Ponder Weight:     0.01")
    print(f"  Adv Training:      50% PGD-5 (eps=0.031) + 50% clean")
    print(f"  AMP & Compile:     Enabled")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()
        
        epoch_ce_loss = 0.0
        epoch_ponder_loss = 0.0
        epoch_total_loss = 0.0
        epoch_steps_used = 0.0
        
        train_correct = 0
        train_total = 0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Mixed adversarial training: 50% PGD-5, 50% clean ──
            B = imgs.size(0)
            half = B // 2
            if half > 0:
                # Generate PGD-5 adversarial examples on the first half
                attack_wrapper = LogitsWrapper(compiled_model)
                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        attack_wrapper, imgs[:half], labels[:half],
                        epsilon=0.031, alpha=0.031/4, steps=5,
                        device=device, clip_min=cifar_min, clip_max=cifar_max,
                        random_start=True
                    )
                # Concatenate: [adversarial | clean]
                mixed_imgs = torch.cat([adv_imgs.detach(), imgs[half:]], dim=0)
                mixed_labels = torch.cat([labels[:half], labels[half:]], dim=0)
            else:
                mixed_imgs = imgs
                mixed_labels = labels

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Forward pass returns logits, steps_used, and cumulative_halt
                logits, steps_used, cumulative_halt = compiled_model(mixed_imgs)
                
                # 1. Classification Loss
                loss_ce = F.cross_entropy(logits, mixed_labels)
                
                # 2. Ponder cost: mean steps used (encourages efficiency)
                ponder_cost = steps_used.mean()
                
                # Total dynamic loss composition
                total_loss = loss_ce + 0.01 * ponder_cost

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Stats gathering
            epoch_ce_loss += loss_ce.item() * B
            epoch_ponder_loss += ponder_cost.item() * B
            epoch_total_loss += total_loss.item() * B
            epoch_steps_used += steps_used.mean().item() * B

            _, predicted = logits.max(1)
            train_total += mixed_labels.size(0)
            train_correct += predicted.eq(mixed_labels).sum().item()

        scheduler.step()

        # Normalize metrics
        N = len(trainloader.dataset)
        epoch_ce_loss /= N
        epoch_ponder_loss /= N
        epoch_total_loss /= N
        epoch_steps_used /= N
        train_acc = 100.0 * train_correct / train_total

        # Real-time evaluation on clean validation set
        compiled_model.eval()
        test_correct = 0
        test_total = 0
        test_steps = 0.0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs, steps, _ = compiled_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                test_steps += steps.mean().item() * targets.size(0)

        test_acc = 100.0 * test_correct / test_total
        test_steps /= len(testloader.dataset)

        # Logging to TensorBoard
        tb_writer.add_scalar('Loss/CrossEntropy', epoch_ce_loss, epoch)
        tb_writer.add_scalar('Loss/PonderCost', epoch_ponder_loss, epoch)
        tb_writer.add_scalar('Loss/Total', epoch_total_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)
        tb_writer.add_scalar('Steps/Train', epoch_steps_used, epoch)
        tb_writer.add_scalar('Steps/Test', test_steps, epoch)

        # Checkpoint saving based on clean validation accuracy
        if test_acc > best_test_acc:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), output_ckpt_path)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Total Loss: {epoch_total_loss:.4f} | CE: {epoch_ce_loss:.4f} | Ponder: {epoch_ponder_loss:.4f} | "
              f"Steps: {epoch_steps_used:.2f} | Train: {train_acc:.1f}% | Test: {test_acc:.2f}% (Steps: {test_steps:.2f}) | "
              f"LR: {current_lr:.7f} | Time: {elapsed:.1f}s{marker}", flush=True)

    # Save final checkpoint (only if it's the best; otherwise the best was already saved)
    if test_acc >= best_test_acc:
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(raw_model.state_dict(), output_ckpt_path)
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Adaptive training complete. Model saved successfully to: {output_ckpt_path}")
    print(f"Total training time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    tb_writer.close()


if __name__ == '__main__':
    main()
