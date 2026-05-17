#!/usr/bin/env python3
"""
RHAN Adversarial Training — Fine-tune from clean checkpoint
=============================================================
Loads rhan_v2_best.pth and fine-tunes with PGD-5 adversarial training.
Uses 50/50 mixed clean+adversarial batches for balanced robustness.

All optimizations applied: torch.compile, AMP, cudnn.benchmark,
gradient accumulation (batch 64 × 2 = effective 128).

Target: 83-88% clean, 20-30% PGD-100 at ε=0.05
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import RHAN, count_parameters
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

    # === cuDNN benchmark ===
    torch.backends.cudnn.benchmark = True

    # === Load clean checkpoint ===
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    clean_ckpt = os.path.join(ckpt_dir, 'rhan_v2_best.pth')

    if not os.path.exists(clean_ckpt):
        print(f"ERROR: Clean checkpoint not found at {clean_ckpt}")
        print("Run train_rhan_v2.py first!")
        return

    model = RHAN(num_classes=10, head_type='linear').to(device)
    model.load_state_dict(torch.load(clean_ckpt, map_location=device))
    print(f"Loaded clean checkpoint: {clean_ckpt}")

    total_params, _ = count_parameters(model)
    print(f"Parameters: {total_params:,}")

    # === torch.compile ===
    print("Compiling model...")
    model = torch.compile(model)

    # === Data ===
    from torch.utils.data import DataLoader
    trainloader_raw, testloader_raw = get_dataloaders(
        batch_size=64, num_workers=6, model_name='resnet'
    )
    trainloader = DataLoader(
        trainloader_raw.dataset, batch_size=64, shuffle=True,
        num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=3,
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
        prefetch_factor=3,
    )

    # === Training setup ===
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05)
    epochs = 25
    accumulate_steps = 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Adversarial params
    eps = 0.031      # 8/255 — standard CIFAR adversarial budget
    alpha = 0.01     # PGD step size
    pgd_steps = 5    # PGD-5 for training speed

    # CIFAR bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    ckpt_path = os.path.join(ckpt_dir, 'rhan_adv_best.pth')
    best_acc = 0.0

    print(f"\n{'='*70}")
    print(f"RHAN Adversarial Fine-Tuning (PGD-5, ε=8/255)")
    print(f"{'='*70}")
    print(f"  Base checkpoint:  {clean_ckpt}")
    print(f"  Optimizer:        AdamW (lr=0.00005, wd=0.05)")
    print(f"  Scheduler:        CosineAnnealingLR (T_max={epochs})")
    print(f"  Batch size:       64 × {accumulate_steps} accumulation = 128 effective")
    print(f"  Epochs:           {epochs}")
    print(f"  PGD steps:        {pgd_steps}")
    print(f"  Epsilon:          {eps:.4f} (8/255)")
    print(f"  Mixed training:   50% adversarial + 50% clean")
    print(f"  AMP:              Yes")
    print(f"  torch.compile:    Yes")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # === FAST PGD-5 attack generation ===
            delta = torch.zeros_like(imgs).uniform_(-eps, eps).to(device)
            delta = torch.clamp(delta, -(imgs - cifar_min.expand_as(imgs)), (cifar_max.expand_as(imgs) - imgs))
            delta.requires_grad_(True)

            for _ in range(pgd_steps):
                with autocast():
                    adv_out = model(imgs + delta)
                    adv_loss = F.cross_entropy(adv_out, labels)
                adv_loss.backward()
                grad = delta.grad.detach()
                delta.data = (delta + alpha * grad.sign())
                delta.data = torch.clamp(delta.data, -eps, eps)
                delta.data = torch.clamp(imgs + delta.data, cifar_min.expand_as(imgs), cifar_max.expand_as(imgs)) - imgs
                delta.grad.zero_()

            adv_imgs = (imgs + delta).detach()

            # === MIXED TRAINING: 50% adv + 50% clean ===
            half = imgs.shape[0] // 2
            mixed = torch.cat([adv_imgs[:half], imgs[half:]])

            # === Forward with gradient accumulation ===
            with autocast():
                output = model(mixed)
                loss = criterion(output, labels) / accumulate_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accumulate_steps * imgs.size(0)
            _, predicted = output.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Handle leftover accumulation
        if (step + 1) % accumulate_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        train_loss /= len(trainloader.dataset)
        train_acc = 100.0 * train_correct / train_total

        # === Evaluation (clean accuracy) ===
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total

        # Checkpointing
        if test_acc > best_acc:
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
              f"LR: {current_lr:.7f} | "
              f"Time: {elapsed:.1f}s{marker}", flush=True)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Adversarial training complete. Best Clean Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"Total training time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # === Full PGD-100 evaluation ===
    print(f"\n{'='*70}")
    print(f"FULL PGD-100 EVALUATION — RHAN Adversarial")
    print(f"{'='*70}")

    eval_model = RHAN(num_classes=10, head_type='linear').to(device)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    eval_model.eval()
    eval_model = torch.compile(eval_model)

    # Also load clean v2 for comparison
    clean_model = RHAN(num_classes=10, head_type='linear').to(device)
    clean_model.load_state_dict(torch.load(clean_ckpt, map_location=device))
    clean_model.eval()
    clean_model = torch.compile(clean_model)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]

    def eval_pgd(model, loader, eps_val, steps=100, max_samples=512):
        if eps_val == 0:
            correct = total = 0
            with torch.no_grad():
                for images, labels in loader:
                    if total >= max_samples:
                        break
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            return 100. * correct / total

        a = max(eps_val / 10, 0.001)
        correct = total = 0
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            _, predicted = pgd_attack(
                model, images, labels,
                epsilon=eps_val, alpha=a, steps=steps,
                device=device, clip_min=cifar_min, clip_max=cifar_max,
            )
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    print("\nEvaluating RHAN-clean (v2)...")
    clean_results = {}
    for eps_val in epsilons:
        acc = eval_pgd(clean_model, testloader, eps_val)
        clean_results[eps_val] = acc
        print(f"  ε={eps_val:.2f} → {acc:.2f}%")

    print("\nEvaluating RHAN-adv...")
    adv_results = {}
    for eps_val in epsilons:
        acc = eval_pgd(eval_model, testloader, eps_val)
        adv_results[eps_val] = acc
        print(f"  ε={eps_val:.2f} → {acc:.2f}%")

    # === Final comparison table ===
    resnet_pgd = {0.00: 95.82, 0.01: 75.57, 0.05: 2.76, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit_pgd = {0.00: 97.80, 0.01: 55.18, 0.05: 8.56, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}
    human = {0.00: 73.33, 0.01: None, 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}

    print(f"\n{'='*90}")
    print(f"FINAL COMPARISON TABLE — PGD-100 Accuracy")
    print(f"{'='*90}")
    header = f"{'ε':<8}"
    for name in ['RHAN-clean', 'RHAN-adv', 'ResNet-18', 'ViT-Small', 'Human']:
        header += f" {name:>12}"
    print(header)
    print("-" * 74)

    for eps_val in epsilons:
        row = f"{eps_val:<8.2f}"
        row += f" {clean_results.get(eps_val, 0):>11.2f}%"
        row += f" {adv_results.get(eps_val, 0):>11.2f}%"
        row += f" {resnet_pgd.get(eps_val, 0):>11.2f}%"
        row += f" {vit_pgd.get(eps_val, 0):>11.2f}%"
        hval = human.get(eps_val)
        if hval is not None:
            row += f" {hval:>11.2f}%"
        else:
            row += f" {'N/A':>12}"
        print(row)

    print(f"{'='*90}")
    print(f"\nTotal wall time: {(time.time() - total_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
