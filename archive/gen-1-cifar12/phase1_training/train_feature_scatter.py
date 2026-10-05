#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_pgd_adversarial(model, x, labels, epsilon, alpha, steps,
                              clip_min, clip_max, random_start=True):
    """PGD attack targeting the feature-space cosine distance."""
    x = x.detach()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta).detach()
    else:
        x_adv = x.clone().detach()

    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min)

    model.eval()
    
    with torch.no_grad():
        with autocast('cuda'):
            _, clean_feat = model.forward_with_features(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                _, adv_feat = model.forward_with_features(x_adv)
                # Feature scatter distance (to be maximized)
                loss = (1.0 - (F.normalize(adv_feat, dim=-1) * 
                               F.normalize(clean_feat.detach(), dim=-1)).sum(dim=-1)
                       ).mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, clip_min, clip_max).detach()

    model.train()
    return x_adv.detach()


def frequency_invariance_loss(model, x_clean, x_adv):
    x_low_c, _ = model.separate_frequencies(x_clean)
    x_low_a, _ = model.separate_frequencies(x_adv)
    f_low_c = model.stem_low(x_low_c)
    f_low_a = model.stem_low(x_low_a)
    return F.mse_loss(f_low_a, f_low_c.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── MODEL (RHANv5) ─────────────────────────────────────────────────────
    model = RHANv5().to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    start_ckpt = os.path.join(ckpt_dir, 'rhan_trades_phase_c_final.pth')

    if os.path.exists(start_ckpt):
        state = torch.load(start_ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
            
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"Skipping parameter {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                    continue
            filtered_state[k] = v
            
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded checkpoint: {start_ckpt}")
        if missing:
            print(f"  New parameters (random init): {missing}")
    else:
        print(f"Error: checkpoint {start_ckpt} not found!")
        return

    # ── HYPERPARAMETERS ────────────────────────────────────────────────────
    epochs = args.epochs
    batch_size = 64
    epsilon = 0.031
    alpha = 0.008  # ~ epsilon / 4
    steps = 7
    beta = 3.0
    lr = 0.0002
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    # ── DATA ───────────────────────────────────────────────────────────────
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=batch_size, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testloader = DataLoader(testloader_raw.dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # ── TRAINING LOOP ──────────────────────────────────────────────────────
    print("=" * 70)
    print("FEATURE SCATTER FINE-TUNING (RHAN-v5)")
    print(f"Epochs: {epochs}, LR: {lr}, Beta: {beta}, Epsilon: {epsilon}, Steps: {steps}")
    print("=" * 70)

    best_acc = 0.0
    best_ckpt_path = os.path.join(ckpt_dir, 'rhan_feature_scatter_best.pth')
    rolling_ckpt_path = os.path.join(ckpt_dir, 'rhan_feature_scatter_rolling.pth')
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        
        total_loss = n_total = 0
        l_trades_sum = l_self_align_sum = l_freq_sum = 0
        
        for imgs, lbls in trainloader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            B = imgs.size(0)

            # 1. Generate adversarial examples targeting feature scatter loss
            x_adv = generate_pgd_adversarial(
                model, imgs, lbls, epsilon, alpha, steps, cifar_min, cifar_max
            )

            # 2. Forward passes
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c, feat_c = model.forward_with_features(imgs)
                logits_a, feat_a = model.forward_with_features(x_adv)
                
                # 3. TRADES Loss with feature-space cosine distance replacing KL divergence
                l_trades = ce_loss(logits_c, lbls) + beta * (
                    1.0 - (F.normalize(feat_a, dim=-1) * F.normalize(feat_c.detach(), dim=-1)).sum(dim=-1)
                ).mean()
                
                # 4. Self-Alignment Loss
                l_self_align = 1.0 - F.cosine_similarity(
                    F.normalize(feat_a, dim=-1),
                    F.normalize(feat_c.detach(), dim=-1)
                ).mean()
                
                # 5. Frequency Invariance Loss
                l_freq = frequency_invariance_loss(model, imgs, x_adv)
                
                # 6. Combined Loss
                loss = 0.55 * l_trades + 0.35 * l_self_align + 0.10 * l_freq

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * B
            l_trades_sum += l_trades.item() * B
            l_self_align_sum += l_self_align.item() * B
            l_freq_sum += l_freq.item() * B
            n_total += B

        scheduler.step()

        # Evaluate clean accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits, _ = model.forward_with_features(imgs)
                correct += logits.argmax(1).eq(lbls).sum().item()
                total += lbls.size(0)
        
        acc = 100. * correct / total
        
        marker = ''
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_ckpt_path)
            marker = ' ★'
            
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, rolling_ckpt_path)
            
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {total_loss/n_total:.4f} | "
            f"Scatter-TRADES: {l_trades_sum/n_total:.4f} | "
            f"Align: {l_self_align_sum/n_total:.4f} | "
            f"Freq: {l_freq_sum/n_total:.4f} | "
            f"CleanAcc: {acc:.2f}% | {time.time()-t0:.0f}s{marker}"
        )

    print(f"\nTraining complete. Best clean acc: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    
    print("\nNext steps:")
    print("Run AutoAttack evaluation targeting Automobile (class 1) and Truck (class 9):")
    print("python3 phase2_attacks/eval_autoattack.py --model checkpoints/rhan_feature_scatter_best.pth")

if __name__ == '__main__':
    main()
