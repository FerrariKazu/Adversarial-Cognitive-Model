#!/usr/bin/env python3
"""
RHAN-UNIFIED: Full training pipeline for STL-10 96x96.

Phase 0: Semantic initialization with unlabeled data (50 epochs)
  - Uses both labeled (5K) and unlabeled (100K) data
  - Pseudo-labeling on unlabeled for richer pretraining
  - Cosine head preserved throughout all phases

Phase 1-8: TRADES adversarial curriculum (20 epochs each = 160 total)
  - Cosine head (no replacement)
  - Beta=2.0 for phases 1-6, 2.5 for phase 7, 3.0 for phase 8
  - 3-epoch warmup at each phase transition (beta=0.5)
  - Lower LR: 0.002 (P1-4), 0.001 (P5-8)
  - Rolling checkpoint every epoch
  - Mid-phase checkpoint at epoch 10
  - Resume capability via --resume and --start-phase

Loss: 0.60*TRADES + 0.25*alignment + 0.15*freq_consistency

Curriculum:
  Phase 1: eps=0.016  beta=2.0
  Phase 2: eps=0.031  beta=2.0
  Phase 3: eps=0.047  beta=2.0
  Phase 4: eps=0.062  beta=2.0
  Phase 5: eps=0.094  beta=2.0
  Phase 6: eps=0.125  beta=2.5
  Phase 7: eps=0.150  beta=2.5
  Phase 8: eps=0.200  beta=3.0
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
from dataset_stl10 import get_stl10_loaders, get_stl10_unlabeled_loader, STL10_MIN, STL10_MAX


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


def save_rolling_checkpoint(args, model, phase, epoch, eps, best_acc):
    """FIX 4: Save rolling checkpoint every epoch for resume capability."""
    ckpt = {
        'model': model.state_dict(),
        'phase': phase,
        'epoch': epoch,
        'eps': eps,
        'best_acc': best_acc,
    }
    torch.save(ckpt, os.path.join(args.ckpt_dir, 'rhan_unified_rolling.pth'))


def load_rolling_checkpoint(args, model):
    """FIX 4: Load rolling checkpoint and return resume state."""
    ckpt_path = os.path.join(args.ckpt_dir, 'rhan_unified_rolling.pth')
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    print("Loaded rolling checkpoint: Phase {}, Epoch {}, eps={}".format(
        ckpt['phase'], ckpt['epoch'], ckpt['eps']))
    return ckpt


def train_phase0(device, args):
    """Phase 0: Semantic initialization with labeled + unlabeled data."""
    print("\n" + "=" * 60)
    print("PHASE 0: SEMANTIC INITIALIZATION (labeled + unlabeled)")
    print("=" * 60)

    model = RHANUnified(head_type='cosine').to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: {:,}".format(total_params))

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    train_loader, test_loader = get_stl10_loaders(batch_size=64)
    unlabeled_loader = get_stl10_unlabeled_loader(batch_size=128)
    unlabeled_iter = iter(unlabeled_loader)

    # IMPROVEMENT 3: CutMix augmentation
    from dataset_stl10 import cutmix_data

    best_acc = 0.0
    for epoch in range(1, 51):
        model.train()
        train_loss = 0; train_correct = 0; total_b = 0
        unl_loss_s = 0; unl_batches = 0
        t0 = time.time()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)

            # IMPROVEMENT 3: Apply CutMix 50% of the time
            use_cutmix = np.random.random() > 0.5
            if use_cutmix:
                imgs, y_a, y_b, lam = cutmix_data(imgs, lbls, alpha=1.0)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(imgs)
                if use_cutmix:
                    loss_sup = lam * ce_loss(logits, y_a) + (1 - lam) * ce_loss(logits, y_b)
                else:
                    loss_sup = ce_loss(logits, lbls)
                loss = loss_sup

                # Unlabeled pseudo-labeling step
                try:
                    unl_imgs, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unl_imgs, _ = next(unlabeled_iter)
                unl_imgs = unl_imgs.to(device)
                with torch.no_grad():
                    pseudo_logits = model(unl_imgs)
                    pseudo_labels = F.softmax(pseudo_logits / 2.0, dim=1)
                unl_logits = model(unl_imgs)
                loss_unl = (-pseudo_labels * F.log_softmax(unl_logits, dim=1)).sum(1).mean()
                loss = loss_sup + 0.3 * loss_unl
                unl_loss_s += loss_unl.item() * unl_imgs.size(0)
                unl_batches += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss_sup.item() * B
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
        unl_avg = unl_loss_s / (unl_batches * 128) if unl_batches else 0
        print("Phase 0 | Ep {:02d}/50 | Sup:{:.4f} Unl:{:.4f} | TrAcc:{:.1f}% TeAcc:{:.1f}% | {:.0f}s".format(
            epoch, train_loss/total_b, unl_avg,
            100.*train_correct/total_b, test_acc, time.time()-t0))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, 'rhan_unified_best.pth'))

    torch.save(model.state_dict(),
               os.path.join(args.ckpt_dir, 'rhan_unified_phase0_final.pth'))
    print("Phase 0 complete. Best test acc: {:.2f}%".format(best_acc))
    return model


def train_phases(device, args, model=None, start_phase=1):
    """Phases 1-8: Full TRADES adversarial curriculum."""
    print("\n" + "=" * 60)
    print("PHASES 1-8: TRADES ADVERSARIAL CURRICULUM")
    print("=" * 60)

    # FIX 4: Resume from rolling checkpoint if --resume
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_rolling_checkpoint(args, model)

    if model is None:
        model = RHANUnified(head_type='cosine').to(device)
        phase0_ckpt = os.path.join(args.ckpt_dir, 'rhan_unified_phase0_final.pth')
        if os.path.exists(phase0_ckpt) and resume_ckpt is None:
            model.load_state_dict(
                torch.load(phase0_ckpt, map_location=device, weights_only=False))
            print("Loaded Phase 0 checkpoint")
        elif resume_ckpt is None:
            print("WARNING: No Phase 0 checkpoint. Starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: {:,}".format(total_params))

    ce_loss = nn.CrossEntropyLoss()
    train_loader, test_loader = get_stl10_loaders(batch_size=64)

    clip_min_t = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
    clip_max_t = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)

    # FIX 1: Keep beta=2.0 through phase 6, only increase at 7-8
    curriculum = [
        (1, 0.016, 2.0, 10),
        (2, 0.031, 2.0, 10),
        (3, 0.047, 2.0, 10),
        (4, 0.062, 2.0, 10),
        (5, 0.094, 2.0, 10),
        (6, 0.125, 2.5, 10),
        (7, 0.150, 2.5, 10),
        (8, 0.200, 3.0, 10),
    ]

    best_acc = resume_ckpt['best_acc'] if resume_ckpt else 0.0

    for phase, eps, beta, steps in curriculum:
        # FIX 4: Skip phases before start_phase
        if phase < start_phase:
            print("Skipping Phase {} (start_phase={})".format(phase, start_phase))
            continue

        print("\nPhase {}: eps={:.3f}, beta={}, steps={}".format(phase, eps, beta, steps))

        # FIX 2: Lower LR for all phases
        if phase <= 4:
            optimizer = optim.SGD(model.parameters(), lr=0.002,
                                  momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.001,
                                  momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        scaler = GradScaler('cuda')
        step_size = eps / 4

        # FIX 4: Determine starting epoch for resume
        start_epoch = 1
        if resume_ckpt and resume_ckpt['phase'] == phase:
            start_epoch = resume_ckpt['epoch'] + 1
            print("Resuming Phase {} from epoch {}".format(phase, start_epoch))

        for epoch in range(start_epoch, 21):
            model.train()
            train_loss = 0; train_correct = 0; total_b = 0
            l_trades_s = 0; l_align_s = 0; l_freq_s = 0
            t0 = time.time()

            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                B = imgs.size(0)

                # IMPROVEMENT 3: Apply CutMix 50% of the time
                use_cutmix = np.random.random() > 0.5
                if use_cutmix:
                    imgs, y_a, y_b, lam = cutmix_data(imgs, lbls, alpha=1.0)
                    # Generate adversarial from mixed images
                    x_adv = generate_trades_adv(
                        model, imgs, step_size, eps, steps,
                        clip_min_t, clip_max_t)
                else:
                    x_adv = generate_trades_adv(
                        model, imgs, step_size, eps, steps,
                        clip_min_t, clip_max_t)

                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    feat_c = model.get_feature_vector(imgs)
                    logits_c = model.head(feat_c)

                    feat_a = model.get_feature_vector(x_adv)
                    logits_a = model.head(feat_a)

                    # FIX 3: 3-epoch warmup with reduced beta
                    effective_beta = 0.5 if epoch <= 3 else beta

                    if use_cutmix:
                        l_trades = (lam * ce_loss(logits_c, y_a) + (1-lam) * ce_loss(logits_c, y_b)) + \
                                   effective_beta * F.kl_div(
                                       F.log_softmax(logits_a, dim=1),
                                       F.softmax(logits_c, dim=1),
                                       reduction='batchmean')
                    else:
                        l_trades = ce_loss(logits_c, lbls) + effective_beta * F.kl_div(
                            F.log_softmax(logits_a, dim=1),
                            F.softmax(logits_c, dim=1),
                            reduction='batchmean')
                    l_align = 1.0 - F.cosine_similarity(
                        feat_c.detach(), feat_a, dim=-1).mean()

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

            # FIX 3: Show warmup status in output
            warmup_tag = " [WARMUP beta=0.5]" if epoch <= 3 else ""
            print("Phase {} (eps={:.3f}) | Ep {:02d}/20{} | L:{:.3f} | "
                  "T:{:.3f} A:{:.4f} F:{:.4f} | "
                  "TrAcc:{:.1f}% TeAcc:{:.1f}% | {:.0f}s".format(
                phase, eps, epoch, warmup_tag, train_loss/total_b,
                l_trades_s/total_b, l_align_s/total_b, l_freq_s/total_b,
                100.*train_correct/total_b, test_acc, time.time()-t0))

            # Health checks
            if epoch == 1 and l_trades_s/total_b > 5.0:
                print("WARNING: TRADES loss > 5.0 on epoch 1 — beta may still be too high")
            if epoch == 5 and test_acc < 60.0:
                print("WARNING: Test acc < 60% by epoch 5 — model may be collapsing")

            # FIX 4: Save rolling checkpoint every epoch
            save_rolling_checkpoint(args, model, phase, epoch, eps, best_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),
                           os.path.join(args.ckpt_dir, 'rhan_unified_best.pth'))

        # FIX 5: Save phase checkpoints regardless of clean accuracy
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
    parser.add_argument('--resume', action='store_true',
                        help='Resume from rolling checkpoint')
    parser.add_argument('--start-phase', type=int, default=1,
                        help='Start from this phase (skips earlier phases)')
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
        train_phases(device, args, model=model, start_phase=args.start_phase)


if __name__ == '__main__':
    main()
