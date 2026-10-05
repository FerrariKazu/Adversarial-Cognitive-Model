#!/usr/bin/env python3
"""
RHAN Dual-Stream — Adversarial Fine-Tuning
=============================================
Loads from rhan_dualstream_clean_best.pth, fine-tunes with PGD-5 curriculum.
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

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_dualstream import RHAN_DualStream
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
    torch.backends.cudnn.benchmark = True

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    clean_ckpt = os.path.join(ckpt_dir, 'rhan_dualstream_clean_best.pth')

    if not os.path.exists(clean_ckpt):
        print(f"ERROR: Clean checkpoint not found at {clean_ckpt}")
        print("Run train_rhan_dualstream.py first!")
        return

    model = RHAN_DualStream(
        num_classes=10, embed_dim=512, num_heads=8,
        ff_dim=2048, dropout=0.1, num_layers_per_stream=2,
        num_recurrent_steps=2,
    ).to(device)
    model.load_state_dict(torch.load(clean_ckpt, map_location=device))
    print(f"Loaded clean checkpoint from {clean_ckpt}")

    trainloader_raw, testloader_raw = get_dataloaders(
        batch_size=64, num_workers=4, model_name='resnet'
    )
    trainloader = DataLoader(
        trainloader_raw.dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2,
    )

    epochs = 30
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'rhan_dualstream_adv'))

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0
    output_ckpt = os.path.join(ckpt_dir, 'rhan_dualstream_adv_best.pth')

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    class LogitsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _ = self.m(x)
            return logits

    def get_epsilon(epoch, total_epochs):
        progress = epoch / max(total_epochs - 1, 1)
        return 0.031 + (0.10 - 0.031) * progress

    print(f"\n{'='*70}")
    print(f"RHAN Dual-Stream — Adversarial Fine-Tuning")
    print(f"{'='*70}")
    print(f"  Epochs: {epochs} | LR: 1e-4 | Batch: 64")
    print(f"  Epsilon curriculum: 0.031 → 0.10")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()

        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        current_eps = get_epsilon(epoch, epochs)
        alpha = max(current_eps / 4, 0.001)

        for imgs, labels in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            B = imgs.size(0)
            half = B // 2
            if half > 0:
                attack_wrapper = LogitsWrapper(compiled_model)
                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        attack_wrapper, imgs[:half], labels[:half],
                        epsilon=current_eps, alpha=alpha, steps=5,
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
                logits, _ = compiled_model(mixed_imgs)
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
        N = len(trainloader.dataset)
        epoch_loss /= N
        train_acc = 100.0 * train_correct / train_total

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
        tb_writer.add_scalar('Epsilon', current_eps, epoch)

        if test_acc > best_test_acc:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), output_ckpt)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | ε={current_eps:.3f} | Loss: {epoch_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}s{marker}", flush=True)

    # Gradient masking check
    print(f"\n{'='*70}")
    print("Gradient Masking Check (PGD-20 vs PGD-100)")
    print(f"{'='*70}")
    eps_check = 0.031
    alpha_check = eps_check / 4
    for steps_check, label_check in [(20, 'PGD-20'), (100, 'PGD-100')]:
        correct = 0
        total = 0
        for imgs, targets in testloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            attack_wrapper = LogitsWrapper(compiled_model)
            with torch.enable_grad():
                adv_imgs, _ = pgd_attack(
                    attack_wrapper, imgs, targets,
                    epsilon=eps_check, alpha=alpha_check, steps=steps_check,
                    device=device, clip_min=cifar_min, clip_max=cifar_max,
                    random_start=True
                )
            with torch.no_grad():
                with autocast():
                    logits, _ = compiled_model(adv_imgs)
                _, preds = logits.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        acc = 100.0 * correct / total
        print(f"  {label_check}: {acc:.2f}%")

    total_elapsed = time.time() - total_start
    print(f"\nAdversarial fine-tuning complete. Best: {best_test_acc:.2f}%")
    print(f"Saved to: {output_ckpt}")
    print(f"Total time: {total_elapsed/60:.1f} min\n")
    tb_writer.close()


if __name__ == '__main__':
    main()
