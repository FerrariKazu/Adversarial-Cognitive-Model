#!/usr/bin/env python3
"""
RHAN-v7: Generative World-Model TRADES Curriculum Training
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v7 import RHANv7
from dataset import get_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps, clip_min, clip_max):
    x_natural = x_natural.detach()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()

    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

    with torch.no_grad():
        logits_clean, _, _, _ = model(x_natural)
        probs_clean = F.softmax(logits_clean, dim=1).detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv, _, _, _ = model(x_adv)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                probs_clean,
                reduction='batchmean'
            )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
        x_adv = (x_natural + delta).detach()
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

    for m in bn_modules:
        m.train()
    return x_adv


def train_phase0(device, args):
    print("\n" + "=" * 60)
    print("PHASE 0: DECODER WARMUP")
    print("=" * 60)
    model = RHANv7(head_type='cosine').to(device)
    start_ckpt = os.path.join(args.ckpt_dir, 'rhan_trades_phase_c_final.pth')
    if os.path.exists(start_ckpt):
        state = torch.load(start_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded backbone from {start_ckpt}")
    else:
        print(f"WARNING: Backbone {start_ckpt} not found! Starting from scratch.")

    # Freeze backbone
    for name, p in model.named_parameters():
        if any(x in name for x in ['vae_mu', 'vae_log_var', 'decoder', 'generative_classifier']):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainloader, testloader = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scaler = GradScaler('cuda')

    epochs = 10
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and not any(x in name for x in ['decoder']):
                m.eval()

        train_loss = 0; l_task_sum = 0; l_recon_sum = 0; l_kl_sum = 0
        train_correct = 0; total = 0

        t0 = time.time()
        for imgs, lbls in trainloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits, x_recon, mu, log_var = model(imgs)
                l_recon = mse_loss(x_recon, imgs)
                kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
                FREE_BITS = 0.1
                l_kl = torch.clamp(kl_per_dim, min=FREE_BITS).mean()
                l_task = ce_loss(logits, lbls)
                loss = 0.5 * l_task + 0.4 * l_recon + 0.1 * l_kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * B
            l_task_sum += l_task.item() * B
            l_recon_sum += l_recon.item() * B
            l_kl_sum += l_kl.item() * B
            train_correct += logits.argmax(1).eq(lbls).sum().item()
            total += B

        print(f"Phase 0 | Epoch {epoch:02d}/{epochs} | "
              f"Loss: {train_loss/total:.4f} (task:{l_task_sum/total:.4f} recon:{l_recon_sum/total:.4f} kl:{l_kl_sum/total:.4f}) | "
              f"Acc: {100.*train_correct/total:.2f}% | {time.time()-t0:.1f}s")

    save_path = os.path.join(args.ckpt_dir, 'rhan_v7_decoder_warm.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Saved Phase 0 checkpoint to {save_path}")


def train_phase1_4(device, args):
    print("\n" + "=" * 60)
    print("PHASES 1-4: GENERATIVE TRADES CURRICULUM")
    print("=" * 60)
    model = RHANv7(head_type='cosine').to(device)
    warm_ckpt = os.path.join(args.ckpt_dir, 'rhan_v7_decoder_warm.pth')
    if os.path.exists(warm_ckpt):
        missing, unexpected = model.load_state_dict(
            torch.load(warm_ckpt, map_location=device, weights_only=False), strict=False)
        print(f"Loaded warmup checkpoint from {warm_ckpt}")
    else:
        print(f"ERROR: {warm_ckpt} not found! Run --phase 0 first.")
        return

    # CORnet-S teacher removed — alignment loss was orthogonal (cos_sim ~ -0.05)
    # and contributed only noise. Freed teacher forward pass entirely.

    # Unfreeze all trainable params
    for n, p in model.named_parameters():
        p.requires_grad = True

    trainloader, testloader = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(x in n for x in ['decoder', 'vae_mu', 'vae_log_var', 'generative_classifier'])]
    decoder_params  = [p for n, p in model.named_parameters()
                       if any(x in n for x in ['decoder', 'vae_mu', 'vae_log_var', 'generative_classifier'])]

    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': 0.003},
        {'params': decoder_params,  'lr': 0.010},
    ], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=0.0001)
    scaler = GradScaler('cuda')

    ce_loss = nn.CrossEntropyLoss()

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1, 3, 1, 1).to(device)

    best_acc = 0.0

    model.eval()
    test_correct = 0; test_total = 0
    with torch.no_grad():
        for imgs, lbls in testloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits, _, _, _ = model(imgs)
            test_correct += logits.argmax(1).eq(lbls).sum().item()
            test_total += lbls.size(0)
    prev_test_acc = 100. * test_correct / test_total
    print(f"Initial test accuracy of loaded checkpoint: {prev_test_acc:.2f}%")

    for epoch in range(1, 81):
        if epoch <= 20:
            phase, eps, beta, steps = 1, 0.062, 6.0, 10
            step_size = eps / 4
        elif epoch <= 40:
            phase, eps, beta, steps = 2, 0.100, 6.0, 10
            step_size = eps / 4
        elif epoch <= 60:
            phase, eps, beta, steps = 3, 0.150, 5.0, 10
            step_size = eps / 4
        else:
            phase, eps, beta, steps = 4, 0.200, 4.5, 10
            step_size = eps / 4

        model.train()
        # Online FR uses model.stem_low directly — no separate critic needed
        train_loss = 0; train_correct = 0; total = 0
        l_trades_s = 0; l_fr_s = 0; l_kl_s = 0
        kl_raw_sum = 0.0

        t0 = time.time()
        for imgs, lbls in trainloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)

            saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            x_adv = generate_trades_adv(model, imgs, step_size, eps, steps, cifar_min, cifar_max)
            for p, g in zip(model.parameters(), saved_grads): p.grad = g

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c, _, mu_c, lv_c = model(imgs)
                logits_a, x_recon_a, mu_a, lv_a, features_a = model.forward_full(x_adv)

                # 1. TRADES loss (50%)
                l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                    F.log_softmax(logits_a, dim=1),
                    F.softmax(logits_c, dim=1),
                    reduction='batchmean'
                )

                # 2. Feature-level adversarial reconstruction (30%)
                # Online feature comparison using CURRENT model state.
                # Clean features: detached (no gradient) — this is our target.
                # Adversarial features: in gradient graph — this trains the backbone.
                # The loss asks: "are YOUR OWN adversarial features similar
                # to YOUR OWN clean features right now?"
                with torch.no_grad():
                    x_low_clean, _ = model.separate_frequencies(imgs)
                    feats_clean = model.stem_low(x_low_clean)
                    feats_clean_n = model.safe_normalize(feats_clean.flatten(1))
                x_low_adv_f, _ = model.separate_frequencies(x_adv)
                feats_adv = model.stem_low(x_low_adv_f)
                feats_adv_n = model.safe_normalize(feats_adv.flatten(1))
                l_feat_recon = (1.0 - (feats_clean_n * feats_adv_n).sum(dim=1)).mean()

                # 3. KL divergence with free bits (20%)
                kl_per_dim = -0.5 * (1 + lv_c - mu_c**2 - torch.exp(lv_c))
                FREE_BITS = 0.1
                l_kl = torch.clamp(kl_per_dim, min=FREE_BITS).mean()

                # Alignment removed: CORnet-S features orthogonal to RHAN latents

                loss = 0.50 * l_trades + 0.30 * l_feat_recon + 0.20 * l_kl

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * B
            train_correct += logits_c.argmax(1).eq(lbls).sum().item()
            total += B

            l_trades_s += l_trades.item() * B
            l_fr_s += l_feat_recon.item() * B
            l_kl_s += l_kl.item() * B
            kl_raw_sum += kl_per_dim.mean().item() * B

        scheduler.step()

        model.eval()
        test_correct = 0; test_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits, _, _, _ = model(imgs)
                test_correct += logits.argmax(1).eq(lbls).sum().item()
                test_total += lbls.size(0)

        test_acc = 100. * test_correct / test_total
        epoch_kl_mean = kl_raw_sum / total
        epoch_fr = l_fr_s / total

        print(f"Phase {phase} (ε={eps:.3f}) | Ep {epoch:02d}/80 | "
              f"L:{train_loss/total:.3f} | TrAcc:{100.*train_correct/total:.1f}% TeAcc:{test_acc:.1f}% | {time.time()-t0:.0f}s")
        print(f"  > T:{l_trades_s/total:.3f} FR:{epoch_fr:.4f} KL:{l_kl_s/total:.4f} (raw:{epoch_kl_mean:.4f})")

        # Health checks
        if epoch_kl_mean < 0.05:
            print("STOP: KL collapsing -- check free bits")
            sys.exit(1)
        if test_acc < 77.0:
            print("STOP: Accuracy below threshold")
            sys.exit(1)
        if epoch_fr > 0.7:
            print("STOP: Feature reconstruction degrading")
            sys.exit(1)

        prev_test_acc = test_acc

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'rhan_v7_best.pth'))

        if epoch == 20:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'rhan_v7_phase1_final.pth'))
        elif epoch == 40:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'rhan_v7_phase2_final.pth'))
        elif epoch == 60:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'rhan_v7_phase3_final.pth'))
        elif epoch == 80:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'rhan_v7_phase4_final.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='all', choices=['0', '1-4', 'all'])
    args = parser.parse_args()

    set_seed(42)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.phase in ['0', 'all']:
        train_phase0(device, args)
    if args.phase in ['1-4', 'all']:
        train_phase1_4(device, args)


if __name__ == '__main__':
    main()
