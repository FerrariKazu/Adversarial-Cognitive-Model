#!/usr/bin/env python3
"""
RHAN-UNIFIED: Full training pipeline for STL-10 96x96.

Phase 0: CLIP semantic initialization (30 epochs, clean only)
Phase 1-6: TRADES adversarial curriculum (20 epochs each = 120 total)

Loss (Phases 1-6): 0.60*TRADES + 0.25*alignment + 0.15*freq_consistency

Curriculum:
  Phase 1: eps=0.031  (8/255)
  Phase 2: eps=0.062  (16/255)
  Phase 3: eps=0.100  (26/255)
  Phase 4: eps=0.150  (38/255)
  Phase 5: eps=0.200  (51/255)
  Phase 6: eps=0.250  (64/255)
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
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_unified import RHANUnified
from dataset_stl10 import get_stl10_loaders, STL10_MIN, STL10_MAX


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps,
                        clip_min, clip_max):
    """Standard PGD attack for TRADES."""
    x_natural = x_natural.detach()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()
    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    with torch.no_grad():
        logits_clean = model(x_natural)
        probs_clean = F.softmax(logits_clean, dim=1).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = model(x_adv)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1), probs_clean,
                reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
        x_adv = (x_natural + delta).detach()
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for m in bn_modules:
        m.train()
    return x_adv


def train_phase0(device, args):
    """Phase 0: Clean training with semantic initialization."""
    print("\n" + "=" * 60)
    print("PHASE 0: SEMANTIC INITIALIZATION")
    print("=" * 60)

    model = RHANUnified(head_type='cosine').to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: {:,}".format(total_params))

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    train_loader, test_loader = get_stl10_loaders(batch_size=64)

    best_acc = 0.0
    for epoch in range(1, 31):
        model.train()
        train_loss = 0; train_correct = 0; total_b = 0
        t0 = time.time()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(imgs)
                loss = ce_loss(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * B
            train_correct += logits.argmax(1).eq(lbls).sum().item()
            total_b += B

        scheduler.step()
        model.eval()
        test_correct = 0; test_total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                test_correct += logits.argmax(1).eq(lbls).sum().item()
                test_total += lbls.size(0)

        test_acc = 100. * test_correct / test_total
        print("Phase 0 | Ep {:02d}/30 | Loss:{:.4f} | TrAcc:{:.1f}% TeAcc:{:.1f}% | {:.0f}s".format(
            epoch, train_loss/total_b,
            100.*train_correct/total_b, test_acc, time.time()-t0))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, 'rhan_unified_best.pth'))

    torch.save(model.state_dict(),
               os.path.join(args.ckpt_dir, 'rhan_unified_phase0_final.pth'))
    print("Phase 0 complete. Best test acc: {:.2f}%".format(best_acc))
    return model


def train_phases1_6(device, args, model=None):
    """Phases 1-6: Full TRADES adversarial curriculum."""
    print("\n" + "=" * 60)
    print("PHASES 1-6: TRADES ADVERSARIAL CURRICULUM")
    print("=" * 60)

    if model is None:
        model = RHANUnified(head_type='cosine').to(device)
        phase0_ckpt = os.path.join(args.ckpt_dir, 'rhan_unified_phase0_final.pth')
        if os.path.exists(phase0_ckpt):
            model.load_state_dict(
                torch.load(phase0_ckpt, map_location=device, weights_only=False))
            print("Loaded Phase 0 checkpoint")
        else:
            print("WARNING: No Phase 0 checkpoint. Starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: {:,}".format(total_params))

    ce_loss = nn.CrossEntropyLoss()
    train_loader, test_loader = get_stl10_loaders(batch_size=64)

    clip_min_t = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
    clip_max_t = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)

    curriculum = [
        (1, 0.031, 6.0, 10),
        (2, 0.062, 6.0, 10),
        (3, 0.100, 6.0, 10),
        (4, 0.150, 5.0, 10),
        (5, 0.200, 5.0, 10),
        (6, 0.250, 4.5, 10),
    ]

    best_acc = 0.0

    for phase, eps, beta, steps in curriculum:
        print("\nPhase {}: eps={:.3f}, beta={}, steps={}".format(phase, eps, beta, steps))

        if phase <= 4:
            optimizer = optim.SGD(model.parameters(), lr=0.01,
                                  momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.005,
                                  momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        scaler = GradScaler('cuda')
        step_size = eps / 4

        for epoch in range(1, 21):
            model.train()
            train_loss = 0; train_correct = 0; total_b = 0
            l_trades_s = 0; l_align_s = 0; l_freq_s = 0
            t0 = time.time()

            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                B = imgs.size(0)

                x_adv = generate_trades_adv(
                    model, imgs, step_size, eps, steps,
                    clip_min_t, clip_max_t)

                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    # Clean features + logits
                    feat_c = model.get_feature_vector(imgs)
                    logits_c = model.head(feat_c)

                    # Adversarial features + logits
                    feat_a = model.get_feature_vector(x_adv)
                    logits_a = model.head(feat_a)

                    # 1. TRADES (60%)
                    l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                        F.log_softmax(logits_a, dim=1),
                        F.softmax(logits_c, dim=1),
                        reduction='batchmean')

                    # 2. Alignment (25%) - feature consistency
                    l_align = 1.0 - F.cosine_similarity(
                        feat_c.detach(), feat_a, dim=-1).mean()

                    # 3. Frequency consistency (15%)
                    stem_c = model.stem(imgs)
                    stem_a = model.stem(x_adv)
                    l_freq = F.mse_loss(stem_a, stem_c.detach())

                    loss = 0.60 * l_trades + 0.25 * l_align + 0.15 * l_freq

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * B
                train_correct += logits_c.argmax(1).eq(lbls).sum().item()
                total_b += B
                l_trades_s += l_trades.item() * B
                l_align_s += l_align.item() * B
                l_freq_s += l_freq.item() * B

            scheduler.step()
            model.eval()
            test_correct = 0; test_total = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    logits = model(imgs)
                    test_correct += logits.argmax(1).eq(lbls).sum().item()
                    test_total += lbls.size(0)

            test_acc = 100. * test_correct / test_total

            print("Phase {} (eps={:.3f}) | Ep {:02d}/20 | L:{:.3f} | "
                  "T:{:.3f} A:{:.4f} F:{:.4f} | "
                  "TrAcc:{:.1f}% TeAcc:{:.1f}% | {:.0f}s".format(
                phase, eps, epoch, train_loss/total_b,
                l_trades_s/total_b, l_align_s/total_b, l_freq_s/total_b,
                100.*train_correct/total_b, test_acc, time.time()-t0))

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),
                           os.path.join(args.ckpt_dir, 'rhan_unified_best.pth'))

        torch.save(model.state_dict(),
                   os.path.join(args.ckpt_dir, 'rhan_unified_phase{}_final.pth'.format(phase)))
        print("Phase {} complete. Best so far: {:.2f}%".format(phase, best_acc))

    print("\nAll phases complete. Best test accuracy: {:.2f}%".format(best_acc))
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all',
                        choices=['0', '1-6', 'all'])
    args = parser.parse_args()

    set_seed(42)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    args.ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model = None
    if args.phase in ['0', 'all']:
        model = train_phase0(device, args)

    if args.phase in ['1-6', 'all']:
        train_phases1_6(device, args, model=model)


if __name__ == '__main__':
    main()
