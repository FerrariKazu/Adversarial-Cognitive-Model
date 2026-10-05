#!/usr/bin/env python3
"""
RHAN-v6: Dynamic Adaptive Frequency Gating Joint Curriculum Training (Phase 2)
================================================================================
Loads CLIP-pretrained weights from checkpoints/rhan_v6_clip_init.pth.
Loads pretrained noise estimator backbone from checkpoints/noise_estimator_pretrained.pth.

6-Phase Curriculum schedule:
  Phase A (1-20):    PGD-5,  ε=0.031
  Phase B (21-45):   PGD-7,  ε=0.062
  Phase C (46-70):   PGD-10, ε=0.100
  Phase D (71-95):   PGD-10, ε=0.150
  Phase E (96-120):  PGD-10, ε=0.200
  Phase F (121-140): PGD-10, ε=0.250 (overrides loss weights for robustness)

Loss (5 components):
  l_adv (CrossEntropy) + l_clean (CrossEntropy) + l_align (CORnet cosine distance) +
  l_freq (low-freq stem MSE consistency) + l_ponder (ACT computational step cost)
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

from model_rhan_v6 import RHANv6
from model_cornets import CIFARCORnet
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing, with optional phase F constant LR."""
    def __init__(self, optimizer, warmup_epochs, cosine_end_epoch, total_epochs, base_lr, phase_f_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine_end_epoch = cosine_end_epoch
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.phase_f_lr = phase_f_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = 0.0001 + (self.base_lr - 0.0001) * (epoch + 1) / self.warmup_epochs
        elif epoch < self.cosine_end_epoch:
            progress = (epoch - self.warmup_epochs) / (self.cosine_end_epoch - self.warmup_epochs)
            lr = 5e-5 + (self.base_lr - 5e-5) * 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            lr = self.phase_f_lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_it_features(teacher_model, x):
    """Extract 512-dim CORnet-S IT features."""
    x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    out = teacher_model.model.module.V1(x_224)
    out = teacher_model.model.module.V2(out)
    out = teacher_model.model.module.V4(out)
    out = teacher_model.model.module.IT(out)
    out = teacher_model.model.module.decoder.avgpool(out)
    out = teacher_model.model.module.decoder.flatten(out)
    return out


def get_curriculum(epoch):
    """Returns (pgd_steps, epsilon, alpha, phase_name, is_phase_f)."""
    if epoch < 20:
        return 5, 0.031, 0.031 / 4, 'A', False
    elif epoch < 45:
        return 7, 0.062, 0.062 / 4, 'B', False
    elif epoch < 70:
        return 10, 0.100, 0.100 / 4, 'C', False
    elif epoch < 95:
        return 10, 0.150, 0.150 / 4, 'D', False
    elif epoch < 120:
        return 10, 0.200, 0.200 / 4, 'E', False
    else:
        return 10, 0.250, 0.250 / 4, 'F', True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RHAN-v6 Curriculum Training")
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoints/rhan_v6_checkpoint.pth')
    args = parser.parse_args()

    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    clip_init_ckpt = os.path.join(ckpt_dir, 'rhan_v6_clip_init.pth')
    noise_est_ckpt = os.path.join(ckpt_dir, 'noise_estimator_pretrained.pth')
    cornet_ckpt = os.path.join(script_dir, 'checkpoints', 'cornets_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v6_best.pth')
    checkpoint_path = os.path.join(ckpt_dir, 'rhan_v6_checkpoint.pth')

    if not os.path.exists(cornet_ckpt):
        print(f"ERROR: CORnet-S checkpoint not found at {cornet_ckpt}")
        return

    # ── Model: RHANv6 ──
    model = RHANv6(head_type='cosine').to(device)

    start_epoch = 0
    best_test_acc = 0.0
    loss_history = {
        'l_adv': [],
        'l_clean': [],
        'l_align': [],
        'l_freq': [],
        'l_ponder': [],
        'l_total': [],
        'train_acc': [],
        'test_acc': []
    }

    # Load weights conditionally
    is_resumed = False
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_acc = checkpoint.get('best_test_acc', 0.0)
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
        is_resumed = True
        print(f"Successfully resumed at Epoch {start_epoch} (next to run: {start_epoch+1})")
    else:
        if args.resume:
            print(f"WARNING: Checkpoint {checkpoint_path} not found. Starting from CLIP init.")
        if not os.path.exists(clip_init_ckpt):
            print(f"ERROR: Phase 1 CLIP init checkpoint not found at {clip_init_ckpt}")
            print("Run pretrain_rhan_v6_clip.py first!")
            return
        if not os.path.exists(noise_est_ckpt):
            print(f"ERROR: Phase 0 Noise Estimator checkpoint not found at {noise_est_ckpt}")
            print("Run pretrain_noise_estimator.py first!")
            return
        
        model.load_state_dict(torch.load(clip_init_ckpt, map_location=device, weights_only=False))
        print(f"RHANv6 loaded from CLIP initialization checkpoint: {clip_init_ckpt}")
        model.load_noise_estimator_weights(noise_est_ckpt, device=device)

    # ── CORnet-S teacher (frozen) ──
    teacher = CIFARCORnet().to(device)
    teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device, weights_only=False))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── DataLoaders ──
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=64, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=64, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # ── Training config ──
    epochs = 140; warmup_epochs = 15; accum_steps = 2
    base_lr = 0.001; phase_f_lr = 0.00005

    optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, 120, epochs, base_lr, phase_f_lr)
    scaler = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_v6'))

    if is_resumed:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Base loss weights (A-E)
    W_ADV = 0.40; W_CLEAN = 0.15; W_ALIGN = 0.25; W_FREQ = 0.10; W_PONDER = 0.10

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    print(f"\n{'='*70}")
    print("RHAN-v6 · Joint Curriculum Training")
    print(f"{'='*70}")
    print(f"  Curriculum:      6-Phase (A-F) up to ε=0.250")
    print(f"  Warmup:          {warmup_epochs} epochs")
    print(f"  Batch size:      64 × {accum_steps} accum = 128 effective")
    print(f"  Save to:         {output_ckpt}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)
        pgd_steps, pgd_eps, pgd_alpha, phase_name, is_phase_f = get_curriculum(epoch)

        # Set loss weights based on Phase
        if is_phase_f:
            w_adv, w_clean, w_align, w_freq, w_ponder = 0.55, 0.10, 0.10, 0.15, 0.10
        else:
            w_adv, w_clean, w_align, w_freq, w_ponder = W_ADV, W_CLEAN, W_ALIGN, W_FREQ, W_PONDER

        compiled_model.train()
        s_adv = s_clean = s_align = s_freq = s_ponder = s_total = 0.0
        train_correct = train_total = 0
        optimizer.zero_grad(set_to_none=True)

        # Log dynamic gating outputs for training tracking
        batch_alpha_lows = []
        batch_alpha_highs = []
        batch_steps_used = []

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0); half = B // 2

            if half > 0:
                saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                with torch.enable_grad():
                    # wrap model to handle tuple returns in pgd
                    class PGDWrapper(nn.Module):
                        def __init__(self, m): super().__init__(); self.m = m
                        def forward(self, x): return self.m(x)[0]
                    pgd_wrapper = PGDWrapper(compiled_model)

                    adv_imgs, _ = pgd_attack(
                        pgd_wrapper, imgs[:half], labels[:half],
                        epsilon=pgd_eps, alpha=pgd_alpha, steps=pgd_steps,
                        device=device, clip_min=cifar_min, clip_max=cifar_max,
                        random_start=True)
                adv_imgs = adv_imgs.detach()
                for p, g in zip(model.parameters(), saved_grads):
                    p.grad = g

                clean_imgs = imgs[half:]
                adv_labels = labels[:half]; cln_labels = labels[half:]
            else:
                adv_imgs = clean_imgs = imgs
                adv_labels = cln_labels = labels

            with autocast('cuda'):
                # 1. Adversarial Task Loss & Steps
                adv_logits, adv_feats, steps_adv = compiled_model.forward_with_features(adv_imgs)
                loss_adv = F.cross_entropy(adv_logits, adv_labels)

                # 2. Clean Task Loss
                cln_logits, _ = compiled_model(clean_imgs)
                loss_clean = F.cross_entropy(cln_logits, cln_labels)

                # 3. Neural Alignment on Adversarial Features
                with torch.no_grad():
                    it_feats = get_it_features(teacher, adv_imgs)
                loss_align = 1.0 - (F.normalize(adv_feats, dim=-1) * F.normalize(it_feats, dim=-1)).sum(dim=-1).mean()

                # 4. Frequency Consistency Loss (low-freq stem)
                if half > 0:
                    x_low_clean, _ = compiled_model.separate_frequencies(imgs[:half])
                    x_low_adv, _ = compiled_model.separate_frequencies(adv_imgs)
                    f_low_clean = compiled_model.stem_low(x_low_clean)
                    f_low_adv = compiled_model.stem_low(x_low_adv)
                    loss_freq = F.mse_loss(f_low_adv, f_low_clean.detach())
                else:
                    loss_freq = torch.tensor(0.0, device=device)

                # 5. Ponder Cost (ACT step penalty)
                loss_ponder = steps_adv / model.max_ponder_steps

                total_loss = (w_adv * loss_adv + w_clean * loss_clean +
                              w_align * loss_align + w_freq * loss_freq +
                              w_ponder * loss_ponder) / accum_steps

            scaler.scale(total_loss).backward()
            s_adv += loss_adv.item() * B; s_clean += loss_clean.item() * B
            s_align += loss_align.item() * B; s_freq += loss_freq.item() * B
            s_ponder += loss_ponder.item() * B; s_total += total_loss.item() * accum_steps * B

            _, pred = adv_logits.max(1)
            train_total += adv_labels.size(0); train_correct += pred.eq(adv_labels).sum().item()

            # Record dynamic weights in evaluation style (sigmoid outputs)
            with torch.no_grad():
                _, x_high = model.separate_frequencies(adv_imgs)
                n_feats = model.noise_estimator_backbone(x_high)
                gates = model.gate_head(n_feats)
                batch_alpha_lows.append(gates[:, 0].mean().item())
                batch_alpha_highs.append(gates[:, 1].mean().item())
                batch_steps_used.append(steps_adv.item())

            if (step + 1) % accum_steps == 0 or (step + 1) == len(trainloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

        N = len(trainloader.dataset)
        l_adv = s_adv / N; l_clean = s_clean / N; l_align = s_align / N
        l_freq_val = s_freq / N; l_ponder_val = s_ponder / N; l_total = s_total / N
        train_acc = 100.0 * train_correct / max(train_total, 1)

        # Average dynamic gate values
        mean_low = np.mean(batch_alpha_lows)
        mean_high = np.mean(batch_alpha_highs)
        mean_steps = np.mean(batch_steps_used)

        # ── Test accuracy ──
        compiled_model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs, _ = compiled_model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0); test_correct += pred.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        # ── TensorBoard Logging ──
        tb_writer.add_scalar('Loss/Adv_CE', l_adv, epoch)
        tb_writer.add_scalar('Loss/Clean_CE', l_clean, epoch)
        tb_writer.add_scalar('Loss/Align', l_align, epoch)
        tb_writer.add_scalar('Loss/FreqConsist', l_freq_val, epoch)
        tb_writer.add_scalar('Loss/Ponder', l_ponder_val, epoch)
        tb_writer.add_scalar('Loss/Total', l_total, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)
        tb_writer.add_scalar('Curriculum/Epsilon', pgd_eps, epoch)
        tb_writer.add_scalar('Weights/alpha_low', mean_low, epoch)
        tb_writer.add_scalar('Weights/alpha_high', mean_high, epoch)
        tb_writer.add_scalar('ACT/Steps', mean_steps, epoch)

        if test_acc >= best_test_acc:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw.state_dict(), output_ckpt)
            best_test_acc = test_acc; marker = ' ★ BEST'
        else:
            marker = ''

        # Record loss history
        loss_history['l_adv'].append(l_adv)
        loss_history['l_clean'].append(l_clean)
        loss_history['l_align'].append(l_align)
        loss_history['l_freq'].append(l_freq_val)
        loss_history['l_ponder'].append(l_ponder_val)
        loss_history['l_total'].append(l_total)
        loss_history['train_acc'].append(train_acc)
        loss_history['test_acc'].append(test_acc)

        # Save rolling checkpoint
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': {},
            'phase_name': phase_name,
            'epsilon': pgd_eps,
            'best_test_acc': best_test_acc,
            'loss_history': loss_history
        }, checkpoint_path)
        print(f"Saved rolling checkpoint to: {checkpoint_path}")

        print(f"Epoch {epoch+1:03d}/{epochs} [{phase_name}] ε={pgd_eps:.3f} | "
              f"Adv:{l_adv:.4f} Cln:{l_clean:.4f} Alg:{l_align:.4f} Frq:{l_freq_val:.4f} Pnd:{l_ponder_val:.4f} | "
              f"Train:{train_acc:.1f}% Test:{test_acc:.2f}% | "
              f"wL:{mean_low:.3f} wH:{mean_high:.3f} Steps:{mean_steps:.2f} | LR:{current_lr:.6f} | "
              f"{time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\n{'='*70}")
    print(f"Phase 2 training complete. Best checkpoint: {output_ckpt}")
    print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}\n")
    tb_writer.close()

    # =====================================================================
    # POST-TRAINING EVALUATION & DIAGNOSTICS PIPELINE
    # =====================================================================
    print("Loading best checkpoint for post-training diagnostics...")
    eval_model = RHANv6(head_type='cosine').to(device)
    eval_model.load_state_dict(torch.load(output_ckpt, map_location=device, weights_only=False))
    eval_model.eval()
    for p in eval_model.parameters():
        p.requires_grad = False

    class EvalWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)[0]
    wrapper = EvalWrapper(eval_model)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    # ── Step 1: Gradient Masking Check ──
    print(f"\n{'='*70}\nGradient Masking Check\n{'='*70}")
    gaps_ok = True
    for check_eps in [0.05, 0.10, 0.20]:
        rn_correct = rn_total = 0
        with torch.no_grad():
            for images, lbls in testloader:
                if rn_total >= max_samples: break
                images, lbls = images.to(device), lbls.to(device)
                noise = torch.empty_like(images).uniform_(-check_eps, check_eps)
                noisy = torch.max(torch.min(images + noise, cifar_max), cifar_min)
                preds, _ = eval_model(noisy)
                preds = preds.max(1)[1]
                rn_correct += preds.eq(lbls).sum().item(); rn_total += lbls.size(0)
        rn_acc = 100.0 * rn_correct / max(rn_total, 1)
        print(f"  Random noise ε={check_eps:.2f}: {rn_acc:.2f}%")

    # PGD-20 at ε=0.05 & 0.10
    p20_accs = {}
    for check_eps in [0.05, 0.10]:
        p20_correct = p20_total = 0
        alpha = check_eps / 4
        for images, lbls in testloader:
            if p20_total >= max_samples: break
            images, lbls = images.to(device), lbls.to(device)
            adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=check_eps, alpha=alpha,
                steps=20, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            with torch.no_grad():
                preds, _ = eval_model(adv_images)
                preds = preds.max(1)[1]
                p20_correct += preds.eq(lbls).sum().item(); p20_total += lbls.size(0)
        p20_accs[check_eps] = 100.0 * p20_correct / max(p20_total, 1)
        print(f"  PGD-20 ε={check_eps:.2f}: {p20_accs[check_eps]:.2f}%")

    # ── Step 2: Full PGD-100 evaluation & Gating/Predictive/ACT logs ──
    print(f"\n{'='*70}\nRHAN-v6 PGD-100 Evaluation & Diagnostics\n{'='*70}")
    v6_accs = []
    dynamic_alpha_lows = []
    dynamic_alpha_highs = []
    avg_ponder_steps = []
    avg_pred_errors = []

    for eps in epsilons:
        t0 = time.time()
        print(f"Evaluating ε={eps:.2f}...", end=' ', flush=True)
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0

        eps_lows = []
        eps_highs = []
        eps_steps = []
        eps_errors = []

        for images, lbls in testloader:
            if total >= max_samples: break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv_images = images

            with torch.no_grad():
                # Extract dynamic outputs
                B = adv_images.size(0)
                x_low, x_high = eval_model.separate_frequencies(adv_images)
                
                # Dynamic Gating
                noise_feats = eval_model.noise_estimator_backbone(x_high)
                gates = eval_model.gate_head(noise_feats)
                eps_lows.append(gates[:, 0].mean().item())
                eps_highs.append(gates[:, 1].mean().item())

                # Run forward pass, tracking steps and computing predictive coding error
                logits, feats, mean_step = eval_model.forward_with_features(adv_images)
                eps_steps.append(mean_step.item())

                # Predictive Coding Error Magnitude
                f_low = eval_model.stem_low(x_low)
                f_high = eval_model.stem_high(x_high)
                f = gates[:, 0].view(B, 1, 1, 1) * f_low + gates[:, 1].view(B, 1, 1, 1) * f_high
                
                predicted_f = eval_model.prediction_decoder(feats)
                predicted_f = predicted_f.view(B, 512, 1, 1).expand_as(f)
                pred_error = torch.norm(f - predicted_f, p=2, dim=1).mean().item()
                eps_errors.append(pred_error)

                preds = logits.max(1)[1]
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)

        acc = 100.0 * correct / max(total, 1); v6_accs.append(acc)
        dynamic_alpha_lows.append(np.mean(eps_lows))
        dynamic_alpha_highs.append(np.mean(eps_highs))
        avg_ponder_steps.append(np.mean(eps_steps))
        avg_pred_errors.append(np.mean(eps_errors))
        print(f"Acc:{acc:.2f}% | Steps:{avg_ponder_steps[-1]:.2f} | Error:{avg_pred_errors[-1]:.4f} | {time.time()-t0:.1f}s")

    # Check PGD-20 vs PGD-100 gaps for gradient masking
    for check_eps in [0.05, 0.10]:
        idx = epsilons.index(check_eps)
        p100_acc = v6_accs[idx]
        gap = p20_accs[check_eps] - p100_acc
        print(f"  PGD-20 vs PGD-100 gap at ε={check_eps:.2f}: {gap:.2f}%")
        if gap >= 8.0:
            print("  ⚠ Potential gradient masking detected!")
            gaps_ok = False
    if gaps_ok:
        print("  ✓ Gradient masking checks PASSED.")

    # ── Step 3: SDT d-prime & εthresh ──
    import scipy.stats as stats
    v6_dprimes = []
    for acc_pct in v6_accs:
        acc = acc_pct / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        v6_dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(v6_dprimes) - 1):
        d1, d2 = v6_dprimes[i], v6_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(v6_dprimes) > 0 and v6_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"

    # ── Step 4 & 5 & 6: Diagnostics Verification ──
    print(f"\n{'='*70}\nDynamic Gating & Predictive Coding Verification\n{'='*70}")
    print(f"{'ε':<8} | {'mean(alpha_low)':>16} | {'mean(alpha_high)':>17} | {'mean(steps)':>11} | {'mean(pred_error)':>16}")
    print("-" * 75)
    for i, eps in enumerate(epsilons):
        print(f"{eps:<8.2f} | {dynamic_alpha_lows[i]:>16.4f} | {dynamic_alpha_highs[i]:>17.4f} | {avg_ponder_steps[i]:>11.2f} | {avg_pred_errors[i]:>16.4f}")
    print("-" * 75)

    if dynamic_alpha_highs[2] < dynamic_alpha_highs[0]:
        print("  ✓ Biological hypothesis CONFIRMED: High-frequency gate alpha_high scales down dynamically with epsilon noise.")
    else:
        print("  ⚠ High-frequency gate did not show dynamic suppression.")

    if avg_ponder_steps[2] > avg_ponder_steps[0]:
        print("  ✓ Reaction time correlate CONFIRMED: Ponder steps increase dynamically under perceptual difficulty.")
    else:
        print("  ⚠ Ponder steps did not scale with perturbation strength.")

    if avg_pred_errors[2] > avg_pred_errors[0]:
        print("  ✓ Predictive coding hypothesis CONFIRMED: Prediction error magnitude increases with noise levels.")
    else:
        print("  ⚠ Prediction error did not show sensitivity to noise.")

    # ── Step 7: Dynamic Inference-Time Ablation Suite ──
    print(f"\n{'='*70}\nInference-Time Ablation Suite\n{'='*70}")
    
    # ── Ablation A: Remove ACT (fixed 2 steps) ──
    eval_model.max_ponder_steps = 2
    eval_model.epsilon_halt = -1.0  # bypasses early halting check
    ab_a_accs = []
    for eps in epsilons:
        correct = total = 0
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
                logits, _ = eval_model(adv_images)
                preds = logits.max(1)[1]
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        ab_a_accs.append(100.0 * correct / max(total, 1))

    # Restore ACT
    eval_model.max_ponder_steps = 6
    eval_model.epsilon_halt = 0.05

    # ── Ablation B: Remove HF Suppressor (force multiplier to 1.0) ──
    # Mock self.hf_suppressor outputs in forward pass. 
    # To do this cleanly, we can temporarily monkeypatch hf_suppressor forward pass:
    class DummySuppressor(nn.Module):
        def forward(self, x): return torch.zeros(x.size(0), 1, device=x.device) # sigmoid(dummy) = 0.0 -> hf_suppress = hf_suppress * (1 - 0) = hf_suppress
    saved_suppressor = eval_model.hf_suppressor
    eval_model.hf_suppressor = DummySuppressor()

    ab_b_accs = []
    for eps in epsilons:
        correct = total = 0
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
                logits, _ = eval_model(adv_images)
                preds = logits.max(1)[1]
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        ab_b_accs.append(100.0 * correct / max(total, 1))
    
    # Restore hf_suppressor
    eval_model.hf_suppressor = saved_suppressor

    # ── Ablation C: Remove Dynamic Gating (use static weights from RHAN-v5) ──
    class StaticGatingHead(nn.Module):
        def forward(self, x):
            # Sigmoid outputs corresponding to static weights (v5)
            # freq_weight_low = 1.2099 -> sigmoid(1.2099) = 0.7703
            # freq_weight_high = 0.5551 -> sigmoid(0.5551) = 0.6353
            B = x.size(0)
            outs = torch.tensor([0.7703, 0.6353], device=x.device).view(1, 2).repeat(B, 1)
            return outs
    saved_gating = eval_model.gate_head
    eval_model.gate_head = StaticGatingHead()

    ab_c_accs = []
    for eps in epsilons:
        correct = total = 0
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
                logits, _ = eval_model(adv_images)
                preds = logits.max(1)[1]
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        ab_c_accs.append(100.0 * correct / max(total, 1))

    # Restore gate head
    eval_model.gate_head = saved_gating

    # Print Ablation Table
    print(f"{'ε':<8} | {'Full Model':>10} | {'Ablate-ACT':>10} | {'Ablate-HFS':>10} | {'Ablate-Gate':>10}")
    print("-" * 60)
    for i, eps in enumerate(epsilons):
        print(f"{eps:<8.2f} | {v6_accs[i]:>9.2f}% | {ab_a_accs[i]:>9.2f}% | {ab_b_accs[i]:>9.2f}% | {ab_c_accs[i]:>9.2f}%")
    print("-" * 60)

    # ── Step 8: Master Comparison Table ──
    rhan_v5 = {0.00: 84.57, 0.01: 80.66, 0.05: 61.13, 0.10: 34.38, 0.20: 2.73, 0.30: 0.20}
    rhan_v3 = {0.00: 91.41, 0.01: 85.35, 0.05: 60.74, 0.10: 26.17, 0.20: 1.17, 0.30: 0.00}
    rhan_adv = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    resnet = {0.00: 95.82, 0.01: 75.57, 0.05: 2.84, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit = {0.00: 97.80, 0.01: 55.18, 0.05: 8.80, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}
    human = {0.00: 73.33, 0.01: 'N/A', 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}

    print(f"\n{'='*95}\nRHAN-v6 FINAL MASTER VERDICT\n{'='*95}")
    print(f"{'ε':<8} | {'Human':>8} | {'RHAN-v6':>8} | {'RHAN-v5':>8} | {'RHAN-v3':>8} | {'RHAN-adv':>8} | {'ResNet':>8} | {'ViT':>8}")
    print("-" * 95)
    for i, eps in enumerate(epsilons):
        h = human[eps]
        h_str = f"{h:.2f}%" if isinstance(h, float) else h
        print(f"{eps:<8.2f} | {h_str:>8} | {v6_accs[i]:>7.2f}% | {rhan_v5[eps]:>7.2f}% | {rhan_v3[eps]:>7.2f}% | {rhan_adv[eps]:>7.2f}% | {resnet[eps]:>7.2f}% | {vit[eps]:>7.2f}%")
    print("=" * 95)

    print(f"\n--- SDT d-prime ---")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f}: d'={v6_dprimes[i]:.4f}")
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    print(f"\n{'='*70}")
    print("ROBUSTNESS RANKING (SDT ε_thresh)")
    print(f"{'='*70}")
    print(f"  {'System':<20} | {'ε_thresh':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Human':<20} | {'> 0.3000':>10}")
    print(f"  {'RHAN-v6':<20} | {thresh_str:>10}")
    print(f"  {'RHAN-v5':<20} | {'0.1030':>10}")
    print(f"  {'RHAN-v3':<20} | {'0.0900':>10}")
    print(f"  {'RHAN-adv':<20} | {'0.0764':>10}")
    print(f"  {'ResNet-18':<20} | {'0.0295':>10}")
    print(f"  {'ViT-Small':<20} | {'0.0264':>10}")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    if eps_thresh is not None and eps_thresh > 0.180:
        print(f"🏆 LANDMARK SUCCESS: ε_thresh = {thresh_str} > 0.180")
    elif eps_thresh is not None and eps_thresh > 0.150:
        print(f"✅ Strong improvement: ε_thresh = {thresh_str} > 0.150")
    else:
        print(f"⚠  ε_thresh = {thresh_str} — did not reach 0.150 success floor")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
