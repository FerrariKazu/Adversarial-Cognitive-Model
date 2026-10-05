#!/usr/bin/env python3
"""
RHAN-v3 Adaptive Recurrence Fine-Tuning
======================================
Fine-tunes the Ventral/Dorsal Split architecture with Adaptive Computation Time (ACT)
using checkpoints/rhan_v3_best.pth as baseline.

Loss composition:
  L_total = L_CE + 0.01 * L_ponder
where:
  L_ponder = mean(steps_used)
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v3_adaptive import AdaptiveRHANSplit
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

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    v3_ckpt = os.path.join(ckpt_dir, 'rhan_v3_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v3_adaptive_best.pth')

    if not os.path.exists(v3_ckpt):
        print(f"ERROR: Base RHAN-v3 checkpoint not found at {v3_ckpt}")
        return

    # 1. Initialize AdaptiveRHANSplit model and load weights
    model = AdaptiveRHANSplit(max_steps=6, epsilon_halt=0.01, head_type='cosine').to(device)
    model.load_from_rhan_split(v3_ckpt, device=device)

    # 2. Setup DataLoaders
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=64, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=64, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2)

    # CIFAR normalisation bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # 3. Setup Optimizer & Scheduler
    epochs = 40
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_v3_adaptive'))

    # Compile model using torch.compile
    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0

    class LogitsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _, _ = self.m(x)
            return logits

    print(f"\n{'='*70}")
    print(f"RHAN-v3 Adaptive Recurrence Fine-Tuning")
    print(f"{'='*70}")
    print(f"  Base checkpoint:   {v3_ckpt}")
    print(f"  Optimizer:         AdamW (lr=5e-5, wd=0.05)")
    print(f"  Scheduler:         CosineAnnealingLR (T_max={epochs})")
    print(f"  Batch size:        64")
    print(f"  Epochs:            {epochs}")
    print(f"  Ponder Weight:     0.01")
    print(f"  Adv Training:      50% PGD-5 (eps=0.031) + 50% clean")
    print(f"  AMP & Compile:     Enabled")
    print(f"  Save to:           {output_ckpt}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()

        epoch_ce_loss = 0.0
        epoch_ponder_loss = 0.0
        epoch_total_loss = 0.0
        epoch_steps_used = 0.0
        train_correct = train_total = 0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed adversarial training: 50% PGD-5, 50% clean
            B = imgs.size(0)
            half = B // 2
            if half > 0:
                attack_wrapper = LogitsWrapper(compiled_model)
                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        attack_wrapper, imgs[:half], labels[:half],
                        epsilon=0.031, alpha=0.031/4, steps=5,
                        device=device, clip_min=cifar_min, clip_max=cifar_max,
                        random_start=True
                    )
                mixed_imgs = torch.cat([adv_imgs.detach(), imgs[half:]], dim=0)
                mixed_labels = torch.cat([labels[:half], labels[half:]], dim=0)
            else:
                mixed_imgs = imgs
                mixed_labels = labels

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                logits, steps_used, _ = compiled_model(mixed_imgs)
                loss_ce = F.cross_entropy(logits, mixed_labels)
                ponder_cost = steps_used.mean()
                total_loss = loss_ce + 0.01 * ponder_cost

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_ce_loss += loss_ce.item() * B
            epoch_ponder_loss += ponder_cost.item() * B
            epoch_total_loss += total_loss.item() * B
            epoch_steps_used += steps_used.mean().item() * B

            _, predicted = logits.max(1)
            train_total += mixed_labels.size(0)
            train_correct += predicted.eq(mixed_labels).sum().item()

        scheduler.step()

        N = len(trainloader.dataset)
        epoch_ce_loss /= N
        epoch_ponder_loss /= N
        epoch_total_loss /= N
        epoch_steps_used /= N
        train_acc = 100.0 * train_correct / train_total

        # Real-time evaluation on clean validation set
        compiled_model.eval()
        test_correct = test_total = 0
        test_steps = 0.0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs, steps, _ = compiled_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                test_steps += steps.mean().item() * targets.size(0)

        test_acc = 100.0 * test_correct / test_total
        test_steps /= len(testloader.dataset)

        tb_writer.add_scalar('Loss/CrossEntropy', epoch_ce_loss, epoch)
        tb_writer.add_scalar('Loss/PonderCost', epoch_ponder_loss, epoch)
        tb_writer.add_scalar('Loss/Total', epoch_total_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)
        tb_writer.add_scalar('Steps/Train', epoch_steps_used, epoch)
        tb_writer.add_scalar('Steps/Test', test_steps, epoch)

        if test_acc >= best_test_acc:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), output_ckpt)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Total Loss: {epoch_total_loss:.4f} | CE: {epoch_ce_loss:.4f} | Ponder: {epoch_ponder_loss:.4f} | "
              f"Steps: {epoch_steps_used:.2f} | Train: {train_acc:.1f}% | Test: {test_acc:.2f}% (Steps: {test_steps:.2f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.7f} | {time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\nFine-tuning complete. Best checkpoint saved to {output_ckpt}")
    tb_writer.close()

    # =========================================================================
    # POST-TRAINING EVALUATION PIPELINE
    # =========================================================================
    print("\nLoading best checkpoint for post-training evaluation...")
    eval_model = AdaptiveRHANSplit(max_steps=6, epsilon_halt=0.01, head_type='cosine').to(device)
    eval_model.load_state_dict(torch.load(output_ckpt, map_location=device))
    eval_model.eval()
    for p in eval_model.parameters():
        p.requires_grad = False

    class EvalWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _, _ = self.m(x)
            return logits
    wrapper = EvalWrapper(eval_model)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    # ── Step 1: Gradient masking check ──
    print(f"\n{'='*70}\nGradient Masking Check\n{'='*70}")

    # Random noise at ε=0.05
    rn_correct = rn_total = 0
    with torch.no_grad():
        for images, lbls in testloader:
            if rn_total >= max_samples: break
            images, lbls = images.to(device), lbls.to(device)
            noise = torch.empty_like(images).uniform_(-0.05, 0.05)
            noisy = torch.max(torch.min(images + noise, cifar_max), cifar_min)
            logits, _, _ = eval_model(noisy)
            _, preds = logits.max(1)
            rn_correct += preds.eq(lbls).sum().item(); rn_total += lbls.size(0)
    rn_acc_05 = 100.0 * rn_correct / max(rn_total, 1)

    # PGD-20 at ε=0.05
    p20_correct = p20_total = 0
    for images, lbls in testloader:
        if p20_total >= max_samples: break
        images, lbls = images.to(device), lbls.to(device)
        adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=0.05, alpha=0.005,
            steps=20, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
        with torch.no_grad():
            logits, _, _ = eval_model(adv_images)
            _, preds = logits.max(1)
            p20_correct += preds.eq(lbls).sum().item(); p20_total += lbls.size(0)
    p20_acc_05 = 100.0 * p20_correct / max(p20_total, 1)

    print(f"  Random noise ε=0.05: {rn_acc_05:.2f}%")
    print(f"  PGD-20      ε=0.05: {p20_acc_05:.2f}%")
    rn_gap = rn_acc_05 - p20_acc_05
    print(f"  Gap (Random - PGD-20): {rn_gap:.2f}%")

    masking_detected = False
    if rn_gap < 20.0:
        print(f"  ⚠ WARNING: Random-PGD gap ({rn_gap:.2f}%) < 20% — possible gradient masking!")
        masking_detected = True
    else:
        print(f"  ✓ Random-PGD gap ({rn_gap:.2f}%) >= 20% — genuine robustness")

    # ── Step 2: Full PGD-100 evaluation ──
    print(f"\n{'='*70}\nPGD-100 Evaluation & Adaptive Steps\n{'='*70}")
    v3_adapt_accs = []
    mean_steps_per_eps = []

    for eps in epsilons:
        t0 = time.time()
        print(f"Evaluating ε={eps:.2f}...", end=' ', flush=True)
        correct = total = 0
        total_steps = 0.0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, lbls in testloader:
            if total >= max_samples: break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv_images = images
            with torch.no_grad():
                logits, steps, _ = eval_model(adv_images)
                _, preds = logits.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)
                total_steps += steps.sum().item()
        acc = 100.0 * correct / max(total, 1); v3_adapt_accs.append(acc)
        m_step = total_steps / max(total, 1); mean_steps_per_eps.append(m_step)
        print(f"Acc:{acc:.2f}% | Steps: {m_step:.2f} | {time.time()-t0:.1f}s")

    # ── Step 3: PGD-20 vs PGD-100 gap check ──
    p100_05 = v3_adapt_accs[2]  # ε=0.05
    pgd_gap = p20_acc_05 - p100_05
    print(f"\nPGD-20 vs PGD-100 gap at ε=0.05: {pgd_gap:.2f}%")
    if pgd_gap < 8.0:
        print(f"  ✓ Gap ({pgd_gap:.2f}%) < 8% — no gradient masking")
    else:
        print(f"  ⚠ Gap ({pgd_gap:.2f}%) >= 8% — potential gradient masking")
        masking_detected = True

    # ── Step 4: SDT d-prime & εthresh ──
    import scipy.stats as stats
    v3_adapt_dprimes = []
    for acc_pct in v3_adapt_accs:
        acc_val = acc_pct / 100.0
        hr = np.clip(acc_val, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc_val) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        v3_adapt_dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(v3_adapt_dprimes) - 1):
        d1, d2 = v3_adapt_dprimes[i], v3_adapt_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(v3_adapt_dprimes) > 0 and v3_adapt_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]

    # ── Step 5: Final comparative verdict table ──
    rhan_v3 = {0.00: 91.41, 0.01: 85.35, 0.05: 60.74, 0.10: 26.17, 0.20: 1.17, 0.30: 0.00}
    rhan_adv = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    resnet = {0.00: 95.82, 0.01: 75.57, 0.05: 2.84, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit = {0.00: 97.80, 0.01: 55.18, 0.05: 8.80, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}

    print(f"\n{'='*85}\nRHAN-v3 Adaptive Recurrence final comparative table\n{'='*85}")
    print(f"{'ε':<8} | {'Human':>8} | {'V3-Adapt':>8} | {'V3-Fixed':>8} | {'RHAN-adv':>8} | {'ResNet':>8} | {'ViT':>8}")
    print("-" * 85)
    human_accs = {0.00: 73.33, 0.01: 'N/A', 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}
    for i, eps in enumerate(epsilons):
        h = human_accs[eps]
        h_str = f"{h:.2f}%" if isinstance(h, float) else h
        v3_a = v3_adapt_accs[i]
        v3_f = rhan_v3[eps]
        ra = rhan_adv[eps]
        rn = resnet[eps]
        vt = vit[eps]
        print(f"{eps:<8.2f} | {h_str:>8} | {v3_a:>7.2f}% | {v3_f:>7.2f}% | {ra:>7.2f}% | {rn:>7.2f}% | {vt:>7.2f}%")
    print("=" * 85)

    print(f"\n--- Mean Recurrent Steps Per Epsilon ---")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f}: {mean_steps_per_eps[i]:.3f} steps")

    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    # ── Step 6: Robustness threshold ranking ──
    print(f"\n{'='*70}\nROBUSTNESS RANKING (SDT ε_thresh)\n{'='*70}")
    print(f"  {'System':<20} | {'ε_thresh':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Human':<20} | {'> 0.3000':>10}")
    print(f"  {'RHAN-v3-Adapt':<20} | {thresh_str:>10}")
    print(f"  {'RHAN-v3':<20} | {'0.0900':>10}")
    print(f"  {'RHAN-adv':<20} | {'0.0764':>10}")
    print(f"  {'ResNet-18':<20} | {'0.0295':>10}")
    print(f"  {'ViT-Small':<20} | {'0.0264':>10}")
    print(f"{'='*70}\n")

    total_elapsed = time.time() - total_start
    print(f"Total time elapsed: {total_elapsed/60:.1f} minutes\n")


if __name__ == '__main__':
    main()
