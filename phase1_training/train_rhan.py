"""
RHAN Training Script
====================
Training configuration for the Recurrent Hybrid Attention Network.
Uses native 32×32 CIFAR-10 input (no resize needed).

Key design choices:
  - AdamW with weight_decay=0.05 for strong regularisation
  - CosineAnnealingLR with 5-epoch linear warmup
  - Label smoothing 0.1 to improve generalisation
  - Batch size 128 (RHAN is lightweight enough for RTX 4060 8GB)
  - 50 epochs (sufficient for convergence with cosine schedule)
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
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))

from model_rhan import RHAN, count_parameters
from dataset import get_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = RHAN(num_classes=10).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"\nRHAN Parameter Count:")
    print(f"  Total:     {total_params:>12,}")
    print(f"  Trainable: {trainable_params:>12,}")
    print(f"  Size (MB): {total_params * 4 / 1024**2:>12.1f}")

    assert total_params < 15_000_000, \
        f"RHAN has {total_params:,} params — exceeds 15M target!"

    # -------------------------------------------------------------------------
    # Data — native 32×32 CIFAR-10 (no resize needed)
    # -------------------------------------------------------------------------
    trainloader, testloader = get_dataloaders(
        batch_size=128,
        num_workers=4,
        model_name='resnet'  # Uses standard 32×32 CIFAR-10 transforms
    )

    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.05,
    )

    epochs = 50
    warmup_epochs = 5
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lr=0.0003)

    # TensorBoard
    writer = SummaryWriter('../runs/rhan_cifar10')
    os.makedirs('checkpoints', exist_ok=True)

    best_acc = 0.0

    print(f"\nStarting training for RHAN ({epochs} epochs)...")
    print(f"  Optimizer:  AdamW (lr={0.0003}, wd={0.05})")
    print(f"  Scheduler:  CosineAnnealing + {warmup_epochs}-epoch warmup")
    print(f"  Loss:       CrossEntropy (label_smoothing=0.1)")
    print(f"  Batch size: 128")
    print(f"  Input size: 32×32 (native CIFAR-10)")
    print()

    for epoch in range(epochs):
        start_time = time.time()

        # Update learning rate
        current_lr = scheduler.step(epoch)

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_ortho_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)

            # Encourage class prototypes to be orthogonal (spread in feature space)
            # This prevents prototype collapse in the cosine similarity head
            prototypes = model.head.class_prototypes  # shape: (10, 512)
            proto_norm = F.normalize(prototypes, dim=1)
            similarity_matrix = proto_norm @ proto_norm.T  # (10, 10)
            # Penalize off-diagonal similarity (want identity matrix)
            identity = torch.eye(10, device=device)
            ortho_loss = ((similarity_matrix - identity) ** 2).sum()
            ortho_weight = 0.01  # small weight — don't let it dominate

            loss = ce_loss + ortho_weight * ortho_loss
            loss.backward()

            # Gradient clipping for stability (recurrent architectures benefit)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_ortho_loss += ortho_loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(trainloader.dataset)
        train_ortho_loss /= len(trainloader.dataset)
        train_acc = 100.0 * train_correct / train_total

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total

        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Train_Ortho', train_ortho_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Checkpointing
        if test_acc > best_acc:
            torch.save(model.state_dict(), 'checkpoints/rhan_best.pth')
            best_acc = test_acc
            marker = ' ★ NEW BEST'
        else:
            marker = ''

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s{marker}", flush=True)

    writer.close()
    print(f"\nTraining complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: checkpoints/rhan_best.pth")


if __name__ == '__main__':
    main()
