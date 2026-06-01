#!/usr/bin/env python3
"""
RHAN with Neural Representation Alignment
============================================
Adds alignment loss on top of RHAN-adv, using cached CORnet-S IT features
as biological reference targets.

Total loss = CE_loss + alpha * alignment_loss
where alpha = 0.1

Two stages:
  1. Clean training with alignment (20 epochs)
  2. Adversarial fine-tuning with alignment (3 epochs)
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
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import RHAN
from alignment_head import AlignmentHead, alignment_loss
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_stage(model, alignment_head, bio_features, trainloader, testloader,
                epochs, lr, device, ckpt_path, tb_suffix, adv=False):
    """Generic training stage (clean or adversarial)."""
    optimizer = optim.AdamW(
        list(model.parameters()) + list(alignment_head.parameters()),
        lr=lr, weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    tb_writer = SummaryWriter(log_dir=os.path.join(
        os.path.dirname(__file__), '..', 'runs', f'rhan_alignment_{tb_suffix}'
    ))

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0
    align_weight = 0.1

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    class LogitsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

    def get_epsilon(epoch, total):
        progress = epoch / max(total - 1, 1)
        return 0.031 + (0.10 - 0.031) * progress

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()
        alignment_head.train()

        epoch_ce = 0.0
        epoch_align = 0.0
        train_correct = 0
        train_total = 0
        current_eps = get_epsilon(epoch, epochs) if adv else 0.0
        alpha = max(current_eps / 4, 0.001) if adv else 0.0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0)

            # Get bio features for this batch (by index)
            batch_start = step * trainloader.batch_size
            batch_end = min(batch_start + B, bio_features.size(0))
            bio_batch = bio_features[batch_start:batch_end].to(device, non_blocking=True)

            if adv and current_eps > 0:
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
                    imgs = torch.cat([adv_imgs.detach(), imgs[half:]], dim=0)
                    labels = torch.cat([labels[:half], labels[half:]], dim=0)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = compiled_model(imgs)
                ce_loss = F.cross_entropy(logits, labels)

                # Alignment loss
                cls_features = compiled_model.get_feature_vector(imgs)
                align_loss_val = alignment_loss(cls_features, bio_batch, alignment_head)

                total_loss = ce_loss + align_weight * align_loss_val

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(alignment_head.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_ce += ce_loss.item() * B
            epoch_align += align_loss_val.item() * B
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()
        N = len(trainloader.dataset)
        epoch_ce /= N
        epoch_align /= N
        train_acc = 100.0 * train_correct / train_total

        # Validation
        compiled_model.eval()
        alignment_head.eval()
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
        tb_writer.add_scalar('Loss/CE', epoch_ce, epoch)
        tb_writer.add_scalar('Loss/Align', epoch_align, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)

        if test_acc > best_test_acc:
            torch.save({
                'model': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
                'alignment_head': alignment_head._orig_mod.state_dict() if hasattr(alignment_head, '_orig_mod') else alignment_head.state_dict(),
            }, ckpt_path)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - epoch_start
        eps_str = f"ε={current_eps:.3f} | " if adv else ""
        print(f"Epoch {epoch+1:02d}/{epochs} | {eps_str}CE: {epoch_ce:.4f} | "
              f"Align: {epoch_align:.4f} | Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.2f}% | {elapsed:.1f}s{marker}", flush=True)

    tb_writer.close()
    return best_test_acc


def main():
    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load cached bio features
    bio_features_path = os.path.join(ckpt_dir, 'cornet_it_features_train.pth')
    if not os.path.exists(bio_features_path):
        print(f"ERROR: Bio features not found at {bio_features_path}")
        print("Run extract_cornet_features.py first!")
        return

    bio_features = torch.load(bio_features_path, map_location='cpu')
    print(f"Loaded bio features: {bio_features.shape}")

    # Load base RHAN-adv
    rhan_ckpt = os.path.join(ckpt_dir, 'rhan_adv_best.pth')
    if not os.path.exists(rhan_ckpt):
        print(f"ERROR: RHAN-adv checkpoint not found at {rhan_ckpt}")
        return

    model = RHAN(num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='linear').to(device)
    model.load_state_dict(torch.load(rhan_ckpt, map_location=device))
    print(f"Loaded RHAN-adv from {rhan_ckpt}")

    alignment_head = AlignmentHead(rhan_dim=512, bio_dim=512, hidden_dim=256).to(device)

    # DataLoaders
    trainloader_raw, testloader_raw = get_dataloaders(
        batch_size=64, num_workers=4, model_name='resnet'
    )
    trainloader = DataLoader(
        trainloader_raw.dataset, batch_size=64, shuffle=False,  # shuffle=False to align with bio_features
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2,
    )

    # Stage 1: Clean training with alignment
    print(f"\n{'='*70}")
    print(f"Stage 1: Clean Training with Alignment Loss")
    print(f"{'='*70}")
    clean_ckpt = os.path.join(ckpt_dir, 'rhan_alignment_clean_best.pth')
    train_stage(model, alignment_head, bio_features, trainloader, testloader,
                epochs=20, lr=1e-4, device=device, ckpt_path=clean_ckpt,
                tb_suffix='clean', adv=False)

    # Reload best clean checkpoint
    if os.path.exists(clean_ckpt):
        ckpt = torch.load(clean_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        alignment_head.load_state_dict(ckpt['alignment_head'])

    # Stage 2: Adversarial fine-tuning with alignment
    print(f"\n{'='*70}")
    print(f"Stage 2: Adversarial Fine-Tuning with Alignment Loss")
    print(f"{'='*70}")
    adv_ckpt = os.path.join(ckpt_dir, 'rhan_alignment_adv_best.pth')
    train_stage(model, alignment_head, bio_features, trainloader, testloader,
                epochs=20, lr=5e-5, device=device, ckpt_path=adv_ckpt,
                tb_suffix='adv', adv=True)

    total_elapsed = time.time() - total_start
    print(f"\nAlignment training complete.")
    print(f"Clean checkpoint: {clean_ckpt}")
    print(f"Adv checkpoint:   {adv_ckpt}")
    print(f"Total time: {total_elapsed/60:.1f} min\n")


if __name__ == '__main__':
    main()
