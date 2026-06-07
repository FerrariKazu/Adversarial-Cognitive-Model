#!/usr/bin/env python3
"""
RHAN-STL-10: Clean training script (no v7 complexity).
Based on proven v5 architecture adapted for 96x96 STL-10.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10 import RHANSTL10
from dataset_stl10 import get_stl10_loaders, STL10_MIN, STL10_MAX

def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps, clip_min, clip_max):
    x_natural = x_natural.detach()
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm2d))]
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    model = RHANSTL10(head_type='cosine').to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    train_loader, test_loader = get_stl10_loaders(batch_size=64)

    cifar_min = torch.tensor(STL10_MIN).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor(STL10_MAX).view(1, 3, 1, 1).to(device)

    best_acc = 0.0

    for epoch in range(1, 81):
        if epoch <= 20:
            eps, beta, steps = 0.062, 6.0, 10
        elif epoch <= 40:
            eps, beta, steps = 0.100, 6.0, 10
        elif epoch <= 60:
            eps, beta, steps = 0.150, 5.0, 10
        else:
            eps, beta, steps = 0.200, 4.5, 10
        step_size = eps / 4

        model.train()
        train_loss = 0; train_correct = 0; total_b = 0
        t0 = time.time()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            B = imgs.size(0)
            x_adv = generate_trades_adv(model, imgs, step_size, eps, steps, cifar_min, cifar_max)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c = model(imgs)
                logits_a = model(x_adv)
                l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                    F.log_softmax(logits_a, dim=1), F.softmax(logits_c, dim=1), reduction='batchmean')
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
        print(f"Ep {epoch:02d}/80 | Loss:{train_loss/total_b:.3f} | TrAcc:{100.*train_correct/total_b:.1f}% TeAcc:{test_acc:.1f}% | {time.time()-t0:.0f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'rhan_stl10_best.pth'))

    print(f"Best test acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
