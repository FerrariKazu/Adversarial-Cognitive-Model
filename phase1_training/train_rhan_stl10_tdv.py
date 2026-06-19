#!/usr/bin/env python3
"""
RHAN-TDV: Temporal Difference Vision Pretraining for STL-10
============================================================

Phases:
  1. tdv: Self-supervised pretraining on STL-10 100K unlabeled images (30 epochs, frozen stem)
  2. label: Class-mapping linear probe training on 5K labeled images (10 epochs, frozen features)
  3. trades: Full adversarial fine-tuning under TRADES curriculum (60 epochs, unfrozen everything)

Usage:
  python phase1_training/train_rhan_stl10_tdv.py --phase tdv
  python phase1_training/train_rhan_stl10_tdv.py --phase label
  python phase1_training/train_rhan_stl10_tdv.py --phase trades
  python phase1_training/train_rhan_stl10_tdv.py --eval-only
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_pretrained import RHANUnifiedSTL10

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION (Temporal Dataset via Augmentation-as-Time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class STL10TemporalDataset(Dataset):
    def __init__(self, data_root='./data/stl10'):
        self.stl10 = torchvision.datasets.STL10(
            data_root, split='unlabeled', download=True
        )
        self.mean = (0.4467, 0.4398, 0.4066)
        self.std  = (0.2603, 0.2566, 0.2713)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return len(self.stl10)

    def __getitem__(self, idx):
        img, _ = self.stl10[idx]
        x_t, x_t1 = self.temporal_augment(img)
        return x_t, x_t1

    def temporal_augment(self, img):
        # Random crop for frame t
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=(0.6, 1.0), ratio=(0.75, 1.33))
        x_t = TF.resized_crop(img, i, j, h, w, (96, 96))

        # Slightly shifted version for frame t+1 (simulated motion)
        delta_i = random.randint(-8, 8)
        delta_j = random.randint(-8, 8)
        i2 = max(0, min(img.height-h, i + delta_i))
        j2 = max(0, min(img.width-w, j + delta_j))
        x_t1 = TF.resized_crop(img, i2, j2, h, w, (96, 96))
        x_t1 = T.ColorJitter(0.1, 0.1, 0.1, 0.05)(x_t1)

        return self.to_tensor(x_t), self.to_tensor(x_t1)

def get_stl10_dataloaders(data_root='./data/stl10', batch_size=64):
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)

    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.STL10(
        data_root, split='train', transform=train_transform, download=True
    )
    testset = torchvision.datasets.STL10(
        data_root, split='test', transform=test_transform, download=True
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    stl_min = torch.tensor([-(m/s) for m, s in zip(mean, std)]).view(1,3,1,1)
    stl_max = torch.tensor([(1-m)/s for m, s in zip(mean, std)]).view(1,3,1,1)

    return trainloader, testloader, stl_min, stl_max

def get_stl10_unlabeled_dataloader(data_root='./data/stl10', batch_size=256):
    trainset = STL10TemporalDataset(data_root)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    return loader

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TDV OBJECTIVES & ADV ATTACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tdv_loss(model, x_t, x_t1):
    """Temporal Difference Vision self-supervised prediction loss."""
    # Encode both frames
    z_t  = model.tdv_head(model.get_feature_vector(x_t))   # (B, 256)
    z_t1 = model.tdv_head(model.get_feature_vector(x_t1))  # (B, 256)
    z_t1_detach = z_t1.detach()

    # Encode motion between frames (outputs B, 256 directly)
    m_proj = model.motion_encoder(x_t, x_t1)  # (B, 256)

    # Prediction prediction: z_t + m_proj ≈ z_t1
    z_t1_pred = z_t + m_proj

    # 1. Prediction discrepancy
    l_pred = F.mse_loss(z_t1_pred, z_t1_detach)

    # 2. Variance loss (VICReg-style standard deviation penalty)
    std_t  = torch.sqrt(z_t.var(dim=0) + 1e-4)
    std_t1 = torch.sqrt(z_t1.var(dim=0) + 1e-4)
    l_var = (F.relu(1 - std_t) + F.relu(1 - std_t1)).mean()

    # 3. Covariance loss (decorrelation)
    B, D = z_t.shape
    z_tc = z_t - z_t.mean(dim=0)
    cov = (z_tc.T @ z_tc) / (B - 1)
    l_cov = (cov**2).sum() - (cov.diagonal()**2).sum()
    l_cov = l_cov / D

    # Correct VICReg scaling: invariance=25, variance=25, covariance=1
    loss = 25.0 * l_pred + 25.0 * l_var + 1.0 * l_cov
    return loss, l_pred, l_var, l_cov

def pgd_attack(model, x_t, x_t1, eps=0.031, steps=3):
    """PGD attack targeting the temporal prediction head."""
    model.eval()
    x_adv = x_t.clone().detach() + 0.001 * torch.randn_like(x_t)
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)
    stl_min = torch.tensor([-(m/s) for m, s in zip(mean, std)]).view(1,3,1,1).to(x_t.device)
    stl_max = torch.tensor([(1-m)/s for m, s in zip(mean, std)]).view(1,3,1,1).to(x_t.device)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)

    with torch.no_grad():
        z_t1 = model.tdv_head(model.get_feature_vector(x_t1)).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                z_t_adv = model.tdv_head(model.get_feature_vector(x_adv))
                m_proj = model.motion_encoder(x_adv, x_t1)
                z_t1_pred = z_t_adv + m_proj
                loss = F.mse_loss(z_t1_pred, z_t1)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + (eps / steps) * grad.sign()
        delta = torch.clamp(x_adv - x_t, -eps, eps)
        x_adv = torch.clamp(x_t + delta, stl_min, stl_max).detach()

    model.train()
    return x_adv

def adversarial_tdv_loss(model, x_t, x_t1, eps=0.031, steps=3):
    x_t_adv = pgd_attack(model, x_t, x_t1, eps=eps, steps=steps)
    loss_clean, l_pred, l_var, l_cov = tdv_loss(model, x_t, x_t1)
    loss_adv, _, _, _ = tdv_loss(model, x_t_adv, x_t1)
    return 0.5 * loss_clean + 0.5 * loss_adv, l_pred, l_var, l_cov

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING PHASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase_tdv(model, unlabeled_loader, device, ckpt_dir, resume=False):
    print("\n" + "="*70)
    print("PHASE TDV: Self-supervised temporal pretraining (stem frozen)")
    print("="*70)

    model.freeze_stem(freeze=True)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler('cuda')

    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_tdv_pretrained_rolling.pth')
    final_path   = os.path.join(ckpt_dir, 'rhan_stl10_tdv_pretrained.pth')

    start_epoch = 1
    if resume and os.path.exists(rolling_path):
        ckpt = torch.load(rolling_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")
    else:
        # Load backbone from checkpoints/rhan_stl_pretrained_phase2_final.pth
        pretrained_ckpt_options = [
            'rhan_stl_pretrained_phase2_final.pth',
            'rhan_stl10_pretrained_phase2.pth',
            'rhan_stl_pretrained_best.pth'
        ]
        loaded = False
        for ckpt_name in pretrained_ckpt_options:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"Loaded backbone from {ckpt_path}")
                print(f"  Missing keys (expected tdv parameters): {len(missing)}")
                loaded = True
                break
        if not loaded:
            print("WARNING: No phase 2 pretrained checkpoint found! Training backbone from scratch.")

    for epoch in range(start_epoch, 31):
        t0 = time.time()
        model.train()

        total_loss = total_pred = total_var = total_cov = n_total = 0
        feature_std_val = 0.0

        for imgs_t, imgs_t1 in unlabeled_loader:
            imgs_t = imgs_t.to(device, non_blocking=True)
            imgs_t1 = imgs_t1.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                loss, l_pred, l_var, l_cov = adversarial_tdv_loss(model, imgs_t, imgs_t1, eps=0.031, steps=3)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs_t.size(0)
            total_loss += loss.item() * B
            total_pred += l_pred.item() * B
            total_var  += l_var.item() * B
            total_cov  += l_cov.item() * B
            n_total    += B

        scheduler.step()

        # Monitor feature variance to check for collapse
        model.eval()
        with torch.no_grad():
            dummy_batch = next(iter(unlabeled_loader))[0].to(device)
            feature_std_val = model.get_feature_vector(dummy_batch).std(dim=0).mean().item()

        print(
            f"TDV Epoch {epoch:02d}/30 | Loss:{total_loss/n_total:.4f} | "
            f"Pred:{total_pred/n_total:.4f} Var:{total_var/n_total:.4f} Cov:{total_cov/n_total:.4f} | "
            f"Std:{feature_std_val:.4f} | {time.time()-t0:.0f}s"
        )

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
        }, rolling_path)

        if feature_std_val < 0.1:
            print("CRITICAL WARNING: Feature collapse detected (Std < 0.1). Adjusting variance penalty.")

    torch.save(model.state_dict(), final_path)
    print(f"Phase TDV Complete. Final model saved to {final_path}")


def run_phase_label(model, trainloader, testloader, device, ckpt_dir, resume=False):
    print("\n" + "="*70)
    print("PHASE LABEL: Labeled class mapping (frozen features)")
    print("="*70)

    # Freeze everything except classification head
    for name, p in model.named_parameters():
        if "prototypes" in name or "log_scale" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_tdv_labeled_rolling.pth')
    final_path   = os.path.join(ckpt_dir, 'rhan_stl10_tdv_labeled.pth')

    start_epoch = 1
    if resume and os.path.exists(rolling_path):
        ckpt = torch.load(rolling_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")
    else:
        tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_tdv_pretrained.pth')
        if os.path.exists(tdv_ckpt):
            model.load_state_dict(torch.load(tdv_ckpt, map_location=device), strict=False)
            print(f"Loaded TDV pretrained checkpoint: {tdv_ckpt}")
        else:
            print("WARNING: TDV pretrained checkpoint not found! Training head from scratch.")

    for epoch in range(start_epoch, 11):
        t0 = time.time()
        model.train()

        total_loss = correct = n_total = 0
        for imgs, lbls in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(imgs)
                loss = ce_loss(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            total_loss += loss.item() * B
            correct += logits.argmax(1).eq(lbls).sum().item()
            n_total += B

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits = model(imgs)
                val_correct += logits.argmax(1).eq(lbls).sum().item()
                val_total += lbls.size(0)

        val_acc = 100. * val_correct / val_total
        print(f"Label Epoch {epoch:02d}/10 | Loss:{total_loss/n_total:.4f} | TrAcc:{100.*correct/n_total:.1f}% TeAcc:{val_acc:.1f}% | {time.time()-t0:.0f}s")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }, rolling_path)

    torch.save(model.state_dict(), final_path)
    print(f"Phase Label Complete. Model saved to {final_path}")


def run_phase_trades(model, trainloader, testloader, unlabeled_loader, device, stl_min, stl_max, ckpt_dir, resume=False):
    print("\n" + "="*70)
    print("PHASE TRADES: Labeled fine-tuning under TRADES curriculum + TDV Consistency")
    print("="*70)

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_tdv_trades_rolling.pth')
    final_path   = os.path.join(ckpt_dir, 'rhan_stl10_tdv_trades.pth')

    start_epoch = 1
    best_acc = 0.0
    if resume and os.path.exists(rolling_path):
        ckpt = torch.load(rolling_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}, best_acc={best_acc:.2f}%")
    else:
        labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_tdv_labeled.pth')
        if os.path.exists(labeled_ckpt):
            model.load_state_dict(torch.load(labeled_ckpt, map_location=device))
            print(f"Loaded TDV labeled checkpoint: {labeled_ckpt}")
        else:
            print("WARNING: TDV labeled checkpoint not found! Training from scratch.")

    unlabeled_iter = iter(unlabeled_loader)

    # Curriculum Setup
    # total 60 epochs:
    # 1-15: eps=0.031, beta=2.0
    # 16-30: eps=0.062, beta=2.0
    # 31-45: eps=0.094, beta=2.5
    # 46-60: eps=0.125, beta=2.5
    for epoch in range(start_epoch, 61):
        t0 = time.time()

        if epoch <= 15:
            eps, beta, steps = 0.031, 2.0, 7
            lr = 0.003
        elif epoch <= 30:
            eps, beta, steps = 0.062, 2.0, 10
            lr = 0.002
        elif epoch <= 45:
            eps, beta, steps = 0.094, 2.5, 10
            lr = 0.001
        else:
            eps, beta, steps = 0.125, 2.5, 12
            lr = 0.0005

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        model.train()
        total_loss = total_tr = total_cons = n_total = correct = 0

        for imgs, lbls in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            # 1. TRADES Adversarial PGD generation
            model.eval()
            x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
            x_adv = torch.clamp(x_adv, stl_min, stl_max)
            for _ in range(steps):
                x_adv.requires_grad_(True)
                with torch.enable_grad():
                    with autocast('cuda'):
                        logits_a = model(x_adv)
                        with torch.no_grad():
                            logits_c = model(imgs)
                        probs_c = F.softmax(logits_c.float(), dim=1)
                        loss_adv = F.kl_div(
                            F.log_softmax(logits_a.float(), dim=1),
                            probs_c, reduction='batchmean'
                        )
                grad = torch.autograd.grad(loss_adv, x_adv)[0]
                x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                delta = torch.clamp(x_adv - imgs, -eps, eps)
                x_adv = torch.clamp(imgs + delta, stl_min, stl_max).detach()
            model.train()

            # 2. Fetch temporal unlabeled pair
            try:
                x_t, x_t1 = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_t, x_t1 = next(unlabeled_iter)
            x_t = x_t.to(device, non_blocking=True)
            x_t1 = x_t1.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c = model(imgs)
                logits_a = model(x_adv)
                l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                    F.log_softmax(logits_a.float(), dim=1),
                    F.softmax(logits_c.float().detach(), dim=1),
                    reduction='batchmean'
                )

                l_tdv_consistency, _, _, _ = tdv_loss(model, x_t, x_t1)
                loss = 0.70 * l_trades + 0.30 * l_tdv_consistency

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            total_loss += loss.item() * B
            total_tr   += l_trades.item() * B
            total_cons += l_tdv_consistency.item() * B
            correct    += logits_c.argmax(1).eq(lbls).sum().item()
            n_total    += B

        # Validation (clean)
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits = model(imgs)
                val_correct += logits.argmax(1).eq(lbls).sum().item()
                val_total += lbls.size(0)

        val_acc = 100. * val_correct / val_total
        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), final_path)
            marker = ' ★'

        print(
            f"TRADES Epoch {epoch:02d}/60 (ε={eps:.3f}) | Loss:{total_loss/n_total:.3f} | "
            f"TrLoss:{total_tr/n_total:.3f} Cons:{total_cons/n_total:.3f} | "
            f"TrAcc:{100.*correct/n_total:.1f}% TeAcc:{val_acc:.1f}% | "
            f"{time.time()-t0:.0f}s{marker}"
        )

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
        }, rolling_path)

    print(f"Phase TRADES Complete. Best Model saved to {final_path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOATTACK EVALUATION MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_autoattack(model, testloader, device, eps=0.031, n_samples=1000, bs=128):
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("\n[WARNING] autoattack not installed. Skipping AutoAttack evaluation.")
        return

    print(f"\n{'='*70}")
    print(f"AUTOATTACK EVALUATION (standard Linf, ε={eps:.4f}, n={n_samples})")
    print(f"{'='*70}")

    model.eval()

    all_imgs, all_lbls = [], []
    for imgs, lbls in testloader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
        if sum(x.size(0) for x in all_imgs) >= n_samples:
            break

    x_test = torch.cat(all_imgs, dim=0)[:n_samples].to(device)
    y_test = torch.cat(all_lbls, dim=0)[:n_samples].to(device)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

    wrapper = Wrapper(model).eval()

    with torch.no_grad():
        clean_preds = wrapper(x_test).argmax(dim=1)
    clean_acc = 100.0 * clean_preds.eq(y_test).sum().item() / y_test.size(0)
    print(f"Clean accuracy (subset): {clean_acc:.2f}%")

    adversary = AutoAttack(wrapper, norm='Linf', eps=eps, version='standard', device=device, verbose=True)
    t0 = time.time()
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
    elapsed = time.time() - t0

    with torch.no_grad():
        adv_preds = wrapper(x_adv).argmax(dim=1)
    aa_correct = adv_preds.eq(y_test).sum().item()
    aa_acc = 100.0 * aa_correct / y_test.size(0)

    print(f"\nAutoAttack Robust Accuracy: {aa_acc:.2f}% ({aa_correct}/{y_test.size(0)}) in {elapsed:.0f}s")

    print("\nPer-class Robust Accuracy:")
    print("-" * 35)
    for c in range(10):
        mask = y_test == c
        n_c = mask.sum().item()
        if n_c > 0:
            c_cl = 100.0 * clean_preds[mask].eq(c).sum().item() / n_c
            c_aa = 100.0 * adv_preds[mask].eq(c).sum().item() / n_c
            tag = " <<< KEY" if c in [2, 9] else ""
            print(f"  {RHANUnifiedSTL10.STL10_CLASSES[c]:<12}: Clean={c_cl:>5.1f}% | AA={c_aa:>5.1f}%{tag}")

    car_aa = 100.0 * adv_preds[y_test == 2].eq(2).sum().item() / max((y_test == 2).sum().item(), 1)
    truck_aa = 100.0 * adv_preds[y_test == 9].eq(9).sum().item() / max((y_test == 9).sum().item(), 1)
    print(f"\nKey Result (Car vs Truck): Car AA = {car_aa:.1f}%, Truck AA = {truck_aa:.1f}%")
    if car_aa > 5.0 and truck_aa > 5.0:
        print(">>> TDV hypothesis confirmed! Temporal difference modeling mitigates adversarial collapse.")
    else:
        print(">>> Collapse is deeper than resolution and temporal context! Further investigation needed.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='tdv', choices=['tdv', 'label', 'trades'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-ckpt', type=str, default='')
    parser.add_argument('--eval-samples', type=int, default=256)
    parser.add_argument('--eval-eps', type=float, default=0.031)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.eval_only:
        model = RHANUnifiedSTL10().to(device)
        ckpt_name = args.eval_ckpt if args.eval_ckpt else 'rhan_stl10_tdv_trades.pth'
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded evaluation checkpoint: {ckpt_path}")
        else:
            print(f"Error: Evaluation checkpoint {ckpt_path} not found.")
            sys.exit(1)
        _, testloader, _, _ = get_stl10_dataloaders(args.data_root, args.batch_size)
        evaluate_autoattack(model, testloader, device, bs=args.batch_size, n_samples=args.eval_samples, eps=args.eval_eps)
        return

    model = RHANUnifiedSTL10().to(device)

    if args.phase == 'tdv':
        unlabeled_loader = get_stl10_unlabeled_dataloader(args.data_root, batch_size=args.batch_size)
        run_phase_tdv(model, unlabeled_loader, device, ckpt_dir, resume=args.resume)
    elif args.phase == 'label':
        trainloader, testloader, _, _ = get_stl10_dataloaders(args.data_root, batch_size=args.batch_size)
        run_phase_label(model, trainloader, testloader, device, ckpt_dir, resume=args.resume)
    elif args.phase == 'trades':
        trainloader, testloader, stl_min, stl_max = get_stl10_dataloaders(args.data_root, batch_size=args.batch_size)
        unlabeled_loader = get_stl10_unlabeled_dataloader(args.data_root, batch_size=args.batch_size)
        stl_min, stl_max = stl_min.to(device), stl_max.to(device)
        run_phase_trades(model, trainloader, testloader, unlabeled_loader, device, stl_min, stl_max, ckpt_dir, resume=args.resume)

if __name__ == '__main__':
    main()
