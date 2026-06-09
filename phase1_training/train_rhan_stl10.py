#!/usr/bin/env python3
"""
RHAN-STL-10: Full training pipeline with proven fixes.
Based on proven v5 architecture adapted for 96x96 STL-10.

Phase 0: Clean pretraining (30 epochs, labeled only, CutMix)
Phases 1-8: TRADES adversarial curriculum (108 epochs total)
  - Beta=2.0 for phases 1-4, 2.5 for 5-6, 3.0 for 7-8
  - 3-epoch beta warmup at each phase transition
  - CutMix augmentation throughout
  - Rolling checkpoint every epoch
  - Resume capability

Total: 138 epochs (~20-27 hours)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10 import RHANSTL10
from dataset_stl10 import get_stl10_loaders, STL10_MIN, STL10_MAX


# =============================================================================
# CutMix Augmentation (CHANGE 4)
# =============================================================================

def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: paste random patches between training images.
    Applied 50% of the time to prevent overfitting on 5K samples.
    """
    lam = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(x.size(0)).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x_mix = x.clone()
    x_mix[:, :, x1:x2, y1:y2] = x[rand_idx, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x_mix, y, y[rand_idx], lam


# =============================================================================
# Attack Generation
# =============================================================================

def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps, clip_min, clip_max):
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
            loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), probs_clean, reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
        x_adv = (x_natural + delta).detach()
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for m in bn_modules:
        m.train()
    return x_adv


# =============================================================================
# Curriculum Definition (CHANGE 1)
# =============================================================================

# Phase-based curriculum with gentle beta values
# (start_epoch, end_epoch, eps, beta, steps)
CURRICULUM = [
    (31,  45,  0.016, 2.0, 10),   # Phase 1: very gentle start
    (46,  60,  0.031, 2.0, 10),   # Phase 2: standard CIFAR equiv
    (61,  75,  0.047, 2.0, 10),   # Phase 3: moderate
    (76,  90,  0.062, 2.0, 10),   # Phase 4: 16/255
    (91,  102, 0.094, 2.5, 10),   # Phase 5: harder
    (103, 114, 0.125, 2.5, 10),   # Phase 6: 32/255
    (115, 126, 0.150, 3.0, 10),   # Phase 7: 38/255
    (127, 138, 0.200, 3.0, 10),   # Phase 8: 51/255
]
PHASE0_EPOCHS = 30
TRADES_EPOCHS = 108
TOTAL_EPOCHS = PHASE0_EPOCHS + TRADES_EPOCHS  # 138


def get_phase_params(epoch):
    """Get epsilon, beta, steps for a given TRADES epoch (1-indexed within TRADES)."""
    for start, end, eps, beta, steps in CURRICULUM:
        if start <= epoch <= end:
            return eps, beta, steps
    # Default to last phase
    return CURRICULUM[-1][2], CURRICULUM[-1][3], CURRICULUM[-1][4]


def get_effective_beta(epoch, beta):
    """
    CHANGE 2: 3-epoch beta warmup at phase transitions.
    First 3 epochs of each phase use 30% of target beta.
    """
    for start, end, eps, b, steps in CURRICULUM:
        if start <= epoch <= start + 2:
            return b * 0.3
    return beta


# =============================================================================
# Phase 0: Clean Pretraining (CHANGE 3)
# =============================================================================

def train_phase0(model, train_loader, test_loader, device, ckpt_dir, epochs=30):
    """
    Phase 0: Pure supervised pretraining on labeled data only.
    No unlabeled. No pseudo-labels. Clean CE + CutMix.
    """
    print("\n" + "=" * 60)
    print("PHASE 0: CLEAN PRETRAINING (labeled + CutMix)")
    print("=" * 60)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0; train_correct = 0; total_b = 0
        t0 = time.time()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)

            # CHANGE 4: Apply CutMix 50% of the time
            use_cutmix = np.random.random() > 0.5
            if use_cutmix:
                imgs_m, y_a, y_b, lam = cutmix_data(imgs, lbls, alpha=1.0)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    logits = model(imgs_m)
                    loss = lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    logits = model(imgs)
                    loss = ce(logits, lbls)

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
        print("Phase 0 | Ep {:02d}/{} | Loss:{:.4f} | TrAcc:{:.1f}% TeAcc:{:.1f}% | {:.0f}s".format(
            epoch, epochs, train_loss / total_b,
            100. * train_correct / total_b, test_acc, time.time() - t0))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'rhan_stl10_phase0.pth'))

    print("Phase 0 complete. Best: {:.2f}%".format(best_acc))
    return model


# =============================================================================
# Phases 1-8: TRADES Adversarial Curriculum
# =============================================================================

def train_trades(model, train_loader, test_loader, device, ckpt_dir,
                 start_epoch=1):
    """
    Phases 1-8: TRADES adversarial curriculum with gentle beta schedule,
    3-epoch warmup at phase transitions, CutMix, and rolling checkpoints.
    """
    print("\n" + "=" * 60)
    print("PHASES 1-8: TRADES ADVERSARIAL CURRICULUM")
    print("=" * 60)

    ce_loss = nn.CrossEntropyLoss()
    clip_min_t = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
    clip_max_t = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)

    best_acc = 0.0

    for epoch in range(start_epoch, TRADES_EPOCHS + 1):
        eps, beta, steps = get_phase_params(epoch)
        effective_beta = get_effective_beta(epoch, beta)
        step_size = eps / 4

        # Phase-specific optimizer (reset each phase)
        phase_num = 1
        for i, (s, e, _, _, _) in enumerate(CURRICULUM):
            if s <= epoch <= e:
                phase_num = i + 1
                break

        if epoch == 1 or any(epoch == s for s, e, _, _, _ in CURRICULUM):
            # New phase: reset optimizer with lower LR
            if phase_num <= 4:
                optimizer = optim.SGD(model.parameters(), lr=0.005,
                                      momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.SGD(model.parameters(), lr=0.003,
                                      momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(CURRICULUM[phase_num-1][1] - CURRICULUM[phase_num-1][0] + 1))
            scaler = GradScaler('cuda')

        warmup_tag = " [WARMUP]" if effective_beta != beta else ""

        model.train()
        train_loss = 0; train_correct = 0; total_b = 0
        t0 = time.time()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)

            # CHANGE 4: Apply CutMix 50% of the time
            use_cutmix = np.random.random() > 0.5
            if use_cutmix:
                imgs_m, y_a, y_b, lam = cutmix_data(imgs, lbls, alpha=1.0)
                x_adv = generate_trades_adv(model, imgs_m, step_size, eps, steps,
                                            clip_min_t, clip_max_t)
            else:
                x_adv = generate_trades_adv(model, imgs, step_size, eps, steps,
                                            clip_min_t, clip_max_t)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c = model(imgs if not use_cutmix else imgs_m)
                logits_a = model(x_adv)

                if use_cutmix:
                    l_trades = (lam * ce_loss(logits_c, y_a) + (1 - lam) * ce_loss(logits_c, y_b)) + \
                               effective_beta * F.kl_div(
                                   F.log_softmax(logits_a, dim=1),
                                   F.softmax(logits_c, dim=1),
                                   reduction='batchmean')
                else:
                    l_trades = ce_loss(logits_c, lbls) + effective_beta * F.kl_div(
                        F.log_softmax(logits_a, dim=1),
                        F.softmax(logits_c, dim=1),
                        reduction='batchmean')

            scaler.scale(l_trades).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += l_trades.item() * B
            train_correct += logits_c.argmax(1).eq(lbls).sum().item()
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

        print("Phase {} (eps={:.3f}) | Ep {:02d}/{} |{} L:{:.3f} | "
              "TrAcc:{:.1f}% TeAcc:{:.1f}% | {:.0f}s".format(
            phase_num, eps, epoch, TRADES_EPOCHS, warmup_tag,
            train_loss / total_b,
            100. * train_correct / total_b, test_acc, time.time() - t0))

        # Health checks
        if epoch <= 3 and train_loss / total_b > 5.0:
            print("WARNING: TRADES loss > 5.0 on epoch 1 — beta may still be too high")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'rhan_stl10_best.pth'))

        # CHANGE 5: Rolling checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'phase_params': (eps, beta, steps),
        }, os.path.join(ckpt_dir, 'rhan_stl10_rolling.pth'))

        # Phase checkpoint at midpoint and end
        for phase_idx, (s, e, eps_val, beta_val, steps_val) in enumerate(CURRICULUM):
            if epoch == s + 9:  # midpoint (~epoch 10 of phase)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'rhan_stl10_phase{}_ep10.pth'.format(phase_idx + 1)))
            if epoch == e:  # end of phase
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'rhan_stl10_phase{}_final.pth'.format(phase_idx + 1)))
                print("Phase {} complete. Best so far: {:.2f}%".format(phase_idx + 1, best_acc))

    print("\nAll phases complete. Best test accuracy: {:.2f}%".format(best_acc))
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume from rolling checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    model = RHANSTL10(head_type='cosine').to(device)
    total = sum(p.numel() for p in model.parameters())
    print("Parameters: {:,}".format(total))

    train_loader, test_loader = get_stl10_loaders(batch_size=64)

    # CHANGE 6: Resume from rolling checkpoint
    start_epoch = 1
    if args.resume:
        rolling_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_rolling.pth')
        if os.path.exists(rolling_ckpt):
            ckpt = torch.load(rolling_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            start_epoch = ckpt['epoch'] + 1
            print("Resumed from epoch {}".format(ckpt['epoch']))
        else:
            print("No rolling checkpoint found. Starting from scratch.")

    # Phase 0: Clean pretraining (CHANGE 3)
    if start_epoch <= PHASE0_EPOCHS:
        model = train_phase0(model, train_loader, test_loader, device, ckpt_dir, epochs=PHASE0_EPOCHS)

    # Phases 1-8: TRADES curriculum
    if start_epoch <= TOTAL_EPOCHS:
        # Adjust start_epoch for TRADES phases
        trades_start = max(start_epoch, PHASE0_EPOCHS + 1)
        model = train_trades(model, train_loader, test_loader, device, ckpt_dir,
                             start_epoch=trades_start)


if __name__ == '__main__':
    main()
