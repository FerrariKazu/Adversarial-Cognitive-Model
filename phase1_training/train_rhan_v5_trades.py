#!/usr/bin/env python3
"""
RHAN-v5: TRADES Adversarial Training with Biological Neural Alignment
===================================================================
Loads CLIP-pretrained weights from checkpoints/rhan_v5_clip_init.pth.
Reuses the exact model_rhan_v5.py architecture.

Loss formulation:
  total_loss = L_trades + 0.2 * L_align
  where L_trades = CE(f(x_clean), y) + beta * KL(f(x_clean) || f(x_adv))
  and L_align is computed on the representations of x_adv vs CORnet-S IT.

Training settings:
  Epochs: 120
  Batch size: 128
  Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
  Scheduler: MultiStepLR (milestones=[75, 100], gamma=0.1)
  AMP: Yes (mixed precision)
  TRADES beta: 6.0
  TRADES attack: L_inf epsilon=0.031, step_size=0.007, perturb_steps=10
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import scipy.stats as stats

# Set float32 matrix multiplication precision
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v5 import RHANv5
from model_cornets import CIFARCORnet
from dataset import get_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def generate_trades_adv(model, x_natural, step_size, epsilon, perturb_steps, clip_min, clip_max):
    """
    Generate adversarial examples for TRADES using KL divergence in FP32.
    """
    model.eval()
    x_natural = x_natural.detach()
    
    # Initialize x_adv with random noise inside L_inf epsilon ball
    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    
    # Compute clean predictions once to avoid redundant model passes
    with torch.no_grad():
        logits_clean = model(x_natural)
        logits_clean = logits_clean[0] if isinstance(logits_clean, tuple) else logits_clean
        probs_clean = F.softmax(logits_clean, dim=1).detach()
        
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = model(x_adv)
            logits_adv = logits_adv[0] if isinstance(logits_adv, tuple) else logits_adv
            
            # KL divergence from clean predictions
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                probs_clean,
                reduction='batchmean'
            )
        # Compute gradient of KL divergence w.r.t. x_adv
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        # Gradient step
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        # Projection back onto epsilon ball
        delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
        x_adv = (x_natural + delta).detach()
        # Clip to valid normalized image bounds
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
        
    model.train()
    return x_adv


def run_autoattack(model, loader, epsilon, device, max_samples=500):
    try:
        from autoattack import AutoAttack
        print(f"\nRunning AutoAttack (standard) at ε={epsilon:.4f} on {max_samples} samples...")
        class AAWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                return out[0] if isinstance(out, tuple) else out
        
        wrapper = AAWrapper(model)
        adversary = AutoAttack(wrapper, norm='Linf', eps=epsilon, version='standard', device=device, verbose=False)
        
        correct = 0
        total = 0
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            x_adv = adversary.run_standard_evaluation(images, labels, bs=images.size(0))
            with torch.no_grad():
                outputs = wrapper(x_adv)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        aa_acc = 100.0 * correct / max(total, 1)
        print(f"AutoAttack standard ε={epsilon:.3f} Accuracy: {aa_acc:.2f}%")
        return aa_acc
    except ImportError:
        print("\nAutoAttack is not installed. Attempting to install autoattack...")
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "autoattack"], check=True)
            from autoattack import AutoAttack
            print("Successfully installed autoattack! Running evaluation...")
            return run_autoattack(model, loader, epsilon, device, max_samples)
        except Exception as e:
            print(f"Failed to install/run AutoAttack: {e}")
            return None
    except Exception as e:
        print(f"Error running AutoAttack: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='RHAN-v5 TRADES Adversarial Training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    clip_init_ckpt = os.path.join(ckpt_dir, 'rhan_v5_clip_init.pth')
    cornet_ckpt = os.path.join(script_dir, 'checkpoints', 'cornets_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_adv_trades_best.pth')
    checkpoint_path = os.path.join(ckpt_dir, 'rhan_adv_trades_checkpoint.pth')

    if not os.path.exists(clip_init_ckpt):
        print(f"ERROR: Phase 0 CLIP init checkpoint not found at {clip_init_ckpt}")
        return
    if not os.path.exists(cornet_ckpt):
        print(f"ERROR: CORnet-S checkpoint not found at {cornet_ckpt}")
        return

    # ── Model: RHANv5 ──
    model = RHANv5(head_type='cosine').to(device)

    # ── Training configuration ──
    epochs = 120
    batch_size = 128
    beta = 6.0
    epsilon = 0.031
    step_size = 0.007
    perturb_steps = 10
    align_weight = 0.2

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma=0.1)
    scaler = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_v5_trades'))

    start_epoch = 0
    best_test_acc = 0.0
    loss_history = {
        'clean_loss': [],
        'robust_loss': [],
        'align_loss': [],
        'total_loss': [],
        'train_acc': [],
        'test_acc': [],
        'w_low': [],
        'w_high': []
    }

    # Load weights conditionally
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_acc = checkpoint.get('best_test_acc', 0.0)
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
        print(f"Successfully resumed at Epoch {start_epoch} (next to run: {start_epoch+1})")
    else:
        if args.resume:
            print(f"WARNING: Checkpoint {checkpoint_path} not found. Starting from CLIP init.")
        model.load_state_dict(torch.load(clip_init_ckpt, map_location=device, weights_only=False))
        print(f"RHANv5 loaded from CLIP initialization checkpoint: {clip_init_ckpt}")

    # ── CORnet-S teacher (frozen) ──
    teacher = CIFARCORnet().to(device)
    teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device, weights_only=False))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── DataLoaders ──
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=batch_size, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    print(f"\n{'='*70}")
    print("RHAN-v5 · TRADES Adversarial Training")
    print(f"{'='*70}")
    print(f"  Architecture:    RHANv5 (freq-separated, ventral/dorsal)")
    print(f"  Initialization:  Phase 0 CLIP checkpoint")
    print(f"  Optimizer:       SGD (lr=0.1, momentum=0.9, wd=5e-4)")
    print(f"  Scheduler:       MultiStepLR (milestones=[75, 100], gamma=0.1)")
    print(f"  Batch size:      {batch_size}")
    print(f"  Epochs:          {epochs}")
    print(f"  TRADES parameters: beta={beta}, eps={epsilon}, steps={perturb_steps}")
    print(f"  Neural alignment weight: {align_weight}")
    print(f"  Save to:         {output_ckpt}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        compiled_model.train()
        s_clean = s_robust = s_align = s_total = 0.0
        train_correct = train_total = 0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0)

            # 1. Generate adversarial examples (x_adv) in FP32
            # Save parameter gradients to protect them from the attack backward passes
            saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            x_adv = generate_trades_adv(
                compiled_model, imgs, 
                step_size=step_size, epsilon=epsilon, perturb_steps=perturb_steps,
                clip_min=cifar_min, clip_max=cifar_max
            )
            for p, g in zip(model.parameters(), saved_grads):
                p.grad = g

            # 2. Compute TRADES and alignment losses in autocast (AMP)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                # forward passes in train mode
                logits_clean, clean_features = compiled_model.forward_with_features(imgs)
                logits_adv, adv_features = compiled_model.forward_with_features(x_adv)

                # TRADES components
                loss_natural = F.cross_entropy(logits_clean, labels)
                loss_robust = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_clean, dim=1),
                    reduction='batchmean'
                )
                loss_trades = loss_natural + beta * loss_robust

                # IT alignment loss on adversarial images
                with torch.no_grad():
                    cornet_it = get_it_features(teacher, x_adv)
                loss_align = 1.0 - F.cosine_similarity(adv_features, cornet_it, dim=-1).mean()

                # Joint Loss
                total_loss = loss_trades + align_weight * loss_align

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Stats aggregation
            s_clean += loss_natural.item() * B
            s_robust += loss_robust.item() * B
            s_align += loss_align.item() * B
            s_total += total_loss.item() * B

            # Track clean train accuracy
            _, pred = logits_clean.max(1)
            train_total += B
            train_correct += pred.eq(labels).sum().item()

        # Step the learning rate scheduler
        scheduler.step()

        # Epoch metrics
        N = len(trainloader.dataset)
        l_clean_epoch = s_clean / N
        l_robust_epoch = s_robust / N
        l_align_epoch = s_align / N
        l_total_epoch = s_total / N
        train_acc = 100.0 * train_correct / train_total

        # Evaluate clean test accuracy
        compiled_model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = compiled_model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0)
                test_correct += pred.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        # Save frequency weights details
        w_lo = torch.sigmoid(model.freq_weight_low).item()
        w_hi = torch.sigmoid(model.freq_weight_high).item()

        # Save history
        loss_history['clean_loss'].append(l_clean_epoch)
        loss_history['robust_loss'].append(l_robust_epoch)
        loss_history['align_loss'].append(l_align_epoch)
        loss_history['total_loss'].append(l_total_epoch)
        loss_history['train_acc'].append(train_acc)
        loss_history['test_acc'].append(test_acc)
        loss_history['w_low'].append(w_lo)
        loss_history['w_high'].append(w_hi)

        # TensorBoard writing
        tb_writer.add_scalar('Loss/Clean_CE', l_clean_epoch, epoch)
        tb_writer.add_scalar('Loss/Robust_KL', l_robust_epoch, epoch)
        tb_writer.add_scalar('Loss/Align', l_align_epoch, epoch)
        tb_writer.add_scalar('Loss/Total', l_total_epoch, epoch)
        tb_writer.add_scalar('Accuracy/Train_Clean', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test_Clean', test_acc, epoch)
        tb_writer.add_scalar('Weights/freq_low', w_lo, epoch)
        tb_writer.add_scalar('Weights/freq_high', w_hi, epoch)

        # Save best checkpoint
        if test_acc >= best_test_acc:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw.state_dict(), output_ckpt)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        # Save rolling checkpoint
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': raw.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_test_acc': best_test_acc,
            'loss_history': loss_history
        }, checkpoint_path)

        print(f"Epoch {epoch+1:03d}/{epochs} | ClnLoss:{l_clean_epoch:.4f} RobLoss:{l_robust_epoch:.4f} Align:{l_align_epoch:.4f} | "
              f"TrainClean:{train_acc:.1f}% TestClean:{test_acc:.2f}% | "
              f"wL:{w_lo:.3f} wH:{w_hi:.3f} | LR:{current_lr:.6f} | "
              f"{time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\n{'='*70}")
    print(f"Training complete. Best checkpoint: {output_ckpt}")
    print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}\n")
    tb_writer.close()

    # =====================================================================
    # POST-TRAINING EVALUATION SUITE
    # =====================================================================
    print("Loading best checkpoint for evaluation...")
    eval_model = RHANv5(head_type='cosine').to(device)
    eval_model.load_state_dict(torch.load(output_ckpt, map_location=device, weights_only=False))
    eval_model.eval()
    for p in eval_model.parameters():
        p.requires_grad = False

    class EvalWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out
    wrapper = EvalWrapper(eval_model)

    from phase2_attacks.pgd import pgd_attack
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    # ── Step 1: Gradient masking check ──
    print(f"\n{'='*70}\nGradient Masking Check\n{'='*70}")
    # Random noise vs PGD-100 at ε=0.05
    rn_correct = rn_total = 0
    with torch.no_grad():
        for images, lbls in testloader:
            if rn_total >= max_samples:
                break
            images, lbls = images.to(device), lbls.to(device)
            noise = torch.empty_like(images).uniform_(-0.05, 0.05)
            noisy = torch.max(torch.min(images + noise, cifar_max), cifar_min)
            outputs = wrapper(noisy)
            _, preds = outputs.max(1)
            rn_correct += preds.eq(lbls).sum().item()
            rn_total += lbls.size(0)
    rn_acc = 100.0 * rn_correct / max(rn_total, 1)
    print(f"  Random noise ε=0.05: {rn_acc:.2f}%")

    # PGD-20 at ε=0.05
    p20_correct = p20_total = 0
    for images, lbls in testloader:
        if p20_total >= max_samples:
            break
        images, lbls = images.to(device), lbls.to(device)
        adv_images, _ = pgd_attack(
            wrapper, images, lbls, epsilon=0.05, alpha=0.005,
            steps=20, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True
        )
        with torch.no_grad():
            outputs = wrapper(adv_images)
            _, preds = outputs.max(1)
            p20_correct += preds.eq(lbls).sum().item()
            p20_total += lbls.size(0)
    p20_acc_05 = 100.0 * p20_correct / max(p20_total, 1)
    print(f"  PGD-20 ε=0.05: {p20_acc_05:.2f}%")

    # ── Step 2: Full PGD-100 evaluation ──
    print(f"\n{'='*70}\nPGD-100 Robustness Evaluation\n{'='*70}")
    trades_accs = []
    for eps in epsilons:
        t0 = time.time()
        print(f"Evaluating ε={eps:.2f}...", end=' ', flush=True)
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, lbls in testloader:
            if total >= max_samples:
                break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(
                    wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True
                )
            else:
                adv_images = images
            with torch.no_grad():
                outputs = wrapper(adv_images)
                _, preds = outputs.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)
        acc = 100.0 * correct / max(total, 1)
        trades_accs.append(acc)
        print(f"Acc: {acc:.2f}% | {time.time()-t0:.1f}s")

    # PGD-20 vs PGD-100 gap
    p100_05 = trades_accs[2]
    pgd_gap = p20_acc_05 - p100_05
    print(f"\nPGD-20 vs PGD-100 gap at ε=0.05: {pgd_gap:.2f}%")
    if pgd_gap >= 8.0:
        print("  ⚠ Potential gradient masking!")
    else:
        print(f"  ✓ No gradient masking (gap {pgd_gap:.2f}% < 8%)")

    random_noise_gap = rn_acc - p100_05
    print(f"Random noise vs PGD-100 gap at ε=0.05: {random_noise_gap:.2f}%")
    if random_noise_gap > 20.0:
        print(f"  ✓ Strong decision boundary difference (gap {random_noise_gap:.2f}% > 20%)")
    else:
        print(f"  ⚠ Narrow gap between random noise and optimization attack.")

    # ── Step 3: SDT d-prime & εthresh ──
    trades_dprimes = []
    for acc_pct in trades_accs:
        acc = acc_pct / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        trades_dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(trades_dprimes) - 1):
        d1, d2 = trades_dprimes[i], trades_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(trades_dprimes) > 0 and trades_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"

    # ── Step 4: Final comparison table ──
    human = {0.00: 73.33, 0.01: 'N/A', 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}
    rhan_v5 = {0.00: 84.57, 0.01: 80.66, 0.05: 61.13, 0.10: 34.38, 0.20: 2.73, 0.30: 0.20}
    rhan_v3 = {0.00: 91.41, 0.01: 85.35, 0.05: 60.74, 0.10: 26.17, 0.20: 1.17, 0.30: 0.00}
    rhan_adv = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    resnet = {0.00: 95.82, 0.01: 75.57, 0.05: 2.84, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit = {0.00: 97.80, 0.01: 55.18, 0.05: 8.80, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}

    print(f"\n{'='*95}\nRHAN-TRADES FINAL COMPARISON\n{'='*95}")
    print(f"{'ε':<8} | {'Human':>8} | {'RHAN-TRADES':>11} | {'RHAN-v5':>8} | {'RHAN-v3':>8} | {'RHAN-adv':>8} | {'ResNet':>8} | {'ViT':>8}")
    print("-" * 95)
    for i, eps in enumerate(epsilons):
        h = human[eps]
        h_str = f"{h:.2f}%" if isinstance(h, float) else h
        print(f"{eps:<8.2f} | {h_str:>8} | {trades_accs[i]:>10.2f}% | {rhan_v5[eps]:>7.2f}% | {rhan_v3[eps]:>7.2f}% | {rhan_adv[eps]:>7.2f}% | {resnet[eps]:>7.2f}% | {vit[eps]:>7.2f}%")
    print("=" * 95)

    print(f"\n--- SDT d-prime ---")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f}: d'={trades_dprimes[i]:.4f}")
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    print(f"\n{'='*70}")
    print("ROBUSTNESS RANKING (SDT ε_thresh)")
    print(f"{'='*70}")
    print(f"  {'System':<20} | {'ε_thresh':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Human':<20} | {'> 0.3000':>10}")
    print(f"  {'RHAN-TRADES':<20} | {thresh_str:>10}")
    print(f"  {'RHAN-v5':<20} | {'0.1030':>10}")
    print(f"  {'RHAN-v3':<20} | {'0.0900':>10}")
    print(f"  {'RHAN-adv':<20} | {'0.0764':>10}")
    print(f"  {'ResNet-18':<20} | {'0.0295':>10}")
    print(f"  {'ViT-Small':<20} | {'0.0264':>10}")
    print(f"{'='*70}\n")

    # ── Step 5: AutoAttack Sweep ──
    run_autoattack(eval_model, testloader, epsilon=0.031, device=device, max_samples=500)

    # ── Step 6: M-Pathway Dominance Check ──
    w_lo_final = torch.sigmoid(eval_model.freq_weight_low).item()
    w_hi_final = torch.sigmoid(eval_model.freq_weight_high).item()
    print(f"\n{'='*70}\nFrequency Gating Weight Analysis\n{'='*70}")
    print(f"  freq_weight_low (sigmoid):  {w_lo_final:.4f}")
    print(f"  freq_weight_high (sigmoid): {w_hi_final:.4f}")
    if w_lo_final > w_hi_final:
        print("  ✓ M-pathway shape dominance preserved (low > high)")
    else:
        print("  ⚠ Low frequency dominance not preserved.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
