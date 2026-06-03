#!/usr/bin/env python3
"""
RHAN-v5: Class-Hardened TRADES Fine-Tuning
===========================================
Loads trained weights from checkpoints/rhan_adv_trades_best.pth.
Reuses the exact model_rhan_v5.py architecture.

Key Innovation: Class-hardened TRADES applies stronger attacks to the
vulnerable class pairs identified by AutoAttack:
  vulnerable_pairs = {
      'automobile': 1,
      'truck': 9,
      'horse': 7,
      'dog': 5,
      'cat': 3,
  }
Base epsilon: 0.031
Hard epsilon: 0.055

Loss formulation:
  total_loss = L_trades + 0.15 * L_align + 0.20 * L_margin
  where L_trades = CE(f(x_clean), y) + beta * KL(f(x_clean) || f(x_adv))
  L_align is computed on representations of x_adv vs CORnet-S IT
  L_margin is the inter-class margin loss between centroids of vulnerable pairs:
    - Automobile (1) vs Truck (9)
    - Horse (7) vs Dog (5)
    - Dog (5) vs Cat (3)

Training settings:
  Epochs: 30
  Batch size: 128
  Optimizer: SGD(lr=0.005, momentum=0.9, weight_decay=5e-4)
  Scheduler: CosineAnnealingLR (lr: 0.005 -> 5e-5)
  AMP: Yes (mixed precision)
  TRADES beta: 6.0
  APGD-style adaptive step: steps=20, milestones=[5, 10, 15]
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


def apgd_attack(model, x_natural, labels, eps, steps=20):
    """
    APGD-style adaptive step attack for TRADES.
    Uses KL divergence to find adversarial directions.
    """
    x_natural = x_natural.detach()
    device = x_natural.device
    B = x_natural.size(0)
    eps = eps.view(B, 1, 1, 1)

    # Freeze BN stats for adversarial generation without full eval mode
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()

    # CIFAR bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # Initialize x_adv with random noise inside L_inf epsilon ball
    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.max(torch.min(x_adv, cifar_max), cifar_min).detach()

    # Precompute clean predictions once
    with torch.no_grad():
        logits_clean = model(x_natural)
        logits_clean = logits_clean[0] if isinstance(logits_clean, tuple) else logits_clean
        probs_clean = F.softmax(logits_clean, dim=1).detach()

    # APGD parameters
    step_size = 2.0 * eps
    x_best = x_adv.clone()

    # Initialize best and previous loss
    with torch.no_grad():
        logits_adv = model(x_adv)
        logits_adv = logits_adv[0] if isinstance(logits_adv, tuple) else logits_adv
        kl_div = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            probs_clean,
            reduction='none'
        ).sum(dim=1)

    loss_best = kl_div.clone()
    loss_best_milestone = kl_div.clone()
    loss_prev = kl_div.clone()

    milestones = [5, 10, 15]
    n_succ = torch.zeros(B, device=device)

    for step in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = model(x_adv)
            logits_adv = logits_adv[0] if isinstance(logits_adv, tuple) else logits_adv
            kl_div = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                probs_clean,
                reduction='none'
            ).sum(dim=1)

        grad = torch.autograd.grad(kl_div.sum(), [x_adv])[0]

        with torch.no_grad():
            # Update best-performing adversarial examples
            improved = kl_div > loss_best
            loss_best[improved] = kl_div[improved]
            x_best[improved] = x_adv[improved]

            # Count success: step is successful if kl_div > loss_prev
            if step > 0:
                n_succ += (kl_div > loss_prev).float()
            else:
                n_succ += 1.0
            loss_prev = kl_div.clone()

            # APGD step (sign of gradient)
            x_next = x_adv.detach() + step_size * torch.sign(grad.detach())

            # Projection onto L_inf ball with per-image epsilon
            delta = torch.clamp(x_next - x_natural, min=-eps, max=eps)
            x_adv = torch.max(torch.min(x_natural + delta, cifar_max), cifar_min).detach()

            # Milestone step-size adaptation
            if step + 1 in milestones:
                prev_milestone = 0 if step + 1 == 5 else milestones[milestones.index(step + 1) - 1]
                interval_len = (step + 1) - prev_milestone

                # Check condition for each image:
                # (1) best loss didn't improve compared to the beginning of the interval, OR
                # (2) successful updates count is less than 0.75 * interval_len
                condition = (loss_best <= loss_best_milestone) | (n_succ < 0.75 * interval_len)

                # Halve step size for those that met the condition
                step_size[condition] /= 2.0
                # Fallback to the best x for those that met the condition
                x_adv[condition] = x_best[condition]

                # Reset milestone statistics
                loss_best_milestone = loss_best.clone()
                n_succ.zero_()

    # Restore BN modules to train mode
    for m in bn_modules:
        m.train()

    return x_best


def inter_class_margin_loss(features, labels, margin=0.5):
    """
    Minimize similarity between centroids of vulnerable class pairs.
    Pairs:
      - (1, 9): Automobile vs Truck
      - (7, 5): Horse vs Dog
      - (5, 3): Dog vs Cat
    """
    loss = 0
    pairs = [(1, 9), (7, 5), (5, 3)]  # auto/truck, horse/dog, dog/cat
    for cls_a, cls_b in pairs:
        feat_a = features[labels == cls_a]
        feat_b = features[labels == cls_b]
        if len(feat_a) == 0 or len(feat_b) == 0:
            continue
        # Minimize similarity between class centroids
        centroid_a = feat_a.mean(0)
        centroid_b = feat_b.mean(0)
        similarity = F.cosine_similarity(
            centroid_a.unsqueeze(0),
            centroid_b.unsqueeze(0)
        )
        loss += F.relu(similarity - margin)
    return loss


def run_autoattack(model, loader, epsilon, device, max_samples=1000):
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
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_correct = {i: 0 for i in range(10)}
        class_total = {i: 0 for i in range(10)}
        
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
                
                # Per-class stats
                for i in range(labels.size(0)):
                    lbl = labels[i].item()
                    pred = preds[i].item()
                    class_total[lbl] += 1
                    if pred == lbl:
                        class_correct[lbl] += 1
                        
        aa_acc = 100.0 * correct / max(total, 1)
        print(f"\nAutoAttack standard ε={epsilon:.3f} Overall Accuracy: {aa_acc:.2f}%")
        
        print("\nPer-class Robust Accuracy under AutoAttack:")
        for i in range(10):
            acc = 100.0 * class_correct[i] / max(class_total[i], 1)
            print(f"  {class_names[i]:>10}: {acc:.2f}% (n={class_total[i]})")
            
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
    parser = argparse.ArgumentParser(description='RHAN-v5 Class-Hardened TRADES Fine-Tuning')
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

    start_ckpt = os.path.join(ckpt_dir, 'rhan_adv_trades_best.pth')
    cornet_ckpt = os.path.join(script_dir, 'checkpoints', 'cornets_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_trades_hardened_best.pth')
    checkpoint_path = os.path.join(ckpt_dir, 'rhan_trades_hardened_checkpoint.pth')

    if not os.path.exists(start_ckpt):
        print(f"ERROR: Starting checkpoint not found at {start_ckpt}")
        return
    if not os.path.exists(cornet_ckpt):
        print(f"ERROR: CORnet-S checkpoint not found at {cornet_ckpt}")
        return

    # ── Model: RHANv5 ──
    model = RHANv5(head_type='cosine').to(device)

    # ── Fine-tuning configuration ──
    epochs = 30
    batch_size = 128
    beta = 6.0
    
    # Vulnerable class epsilon scaling configuration
    vulnerable_pairs = {
        'automobile': 1,
        'truck': 9,
        'horse': 7,
        'dog': 5,
        'cat': 3,
    }
    vulnerable_indices = list(vulnerable_pairs.values())
    base_eps = 0.031
    hard_eps = 0.055
    
    align_weight = 0.15
    margin_weight = 0.20

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    # Cosine annealing decay from 0.005 to 5e-5 for stable fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    scaler = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_trades_hardened'))

    start_epoch = 0
    best_test_acc = 0.0
    loss_history = {
        'clean_loss': [],
        'robust_loss': [],
        'align_loss': [],
        'margin_loss': [],
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
            print(f"WARNING: Checkpoint {checkpoint_path} not found. Starting from best TRADES checkpoint.")
        model.load_state_dict(torch.load(start_ckpt, map_location=device, weights_only=False))
        print(f"RHANv5 loaded from starting checkpoint: {start_ckpt}")

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

    compiled_model = model

    print(f"\n{'='*70}")
    print("RHAN-v5 · Class-Hardened TRADES Training")
    print(f"{'='*70}")
    print(f"  Architecture:     RHANv5 (freq-separated, ventral/dorsal)")
    print(f"  Initialization:   checkpoints/rhan_adv_trades_best.pth")
    print(f"  Optimizer:        SGD (lr=0.005, momentum=0.9, wd=5e-4)")
    print(f"  Scheduler:        CosineAnnealingLR (T_max={epochs}, eta_min=5e-5)")
    print(f"  Batch size:       {batch_size}")
    print(f"  Epochs:           {epochs}")
    print(f"  TRADES beta:      {beta}")
    print(f"  Epsilon scaling:  Base={base_eps:.4f}, Hardened={hard_eps:.4f}")
    print(f"  Vulnerable cls:   {list(vulnerable_pairs.keys())}")
    print(f"  Neural alignment: weight={align_weight}")
    print(f"  Margin loss:      weight={margin_weight}, margin=0.5")
    print(f"  Save to:          {output_ckpt}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        compiled_model.train()
        s_clean = s_robust = s_align = s_margin = s_total = 0.0
        train_correct = train_total = 0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0)

            # Assign epsilon per-image based on class
            eps_per_image = torch.full((B,), base_eps, device=device)
            for cls_idx in vulnerable_indices:
                mask = (labels == cls_idx)
                eps_per_image[mask] = hard_eps

            # 1. Generate adversarial examples (x_adv) using APGD-style adaptive attack
            saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            x_adv = apgd_attack(
                compiled_model, imgs, labels,
                eps=eps_per_image, steps=20
            )
            for p, g in zip(model.parameters(), saved_grads):
                p.grad = g

            # 2. Compute losses under AMP (mixed precision)
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

                # Inter-class margin loss on adversarial features
                loss_margin = inter_class_margin_loss(adv_features, labels, margin=0.5)

                # Joint Loss
                total_loss = loss_trades + align_weight * loss_align + margin_weight * loss_margin

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Stats aggregation
            s_clean += loss_natural.item() * B
            s_robust += loss_robust.item() * B
            s_align += loss_align.item() * B
            s_margin += loss_margin.item() * B
            s_total += total_loss.item() * B

            # Track clean train accuracy
            _, pred = logits_clean.max(1)
            train_total += B
            train_correct += pred.eq(labels).sum().item()

        # Step learning rate
        scheduler.step()

        # Epoch metrics
        N = len(trainloader.dataset)
        l_clean_epoch = s_clean / N
        l_robust_epoch = s_robust / N
        l_align_epoch = s_align / N
        l_margin_epoch = s_margin / N
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
        loss_history['margin_loss'].append(l_margin_epoch)
        loss_history['total_loss'].append(l_total_epoch)
        loss_history['train_acc'].append(train_acc)
        loss_history['test_acc'].append(test_acc)
        loss_history['w_low'].append(w_lo)
        loss_history['w_high'].append(w_hi)

        # TensorBoard writing
        tb_writer.add_scalar('Loss/Clean_CE', l_clean_epoch, epoch)
        tb_writer.add_scalar('Loss/Robust_KL', l_robust_epoch, epoch)
        tb_writer.add_scalar('Loss/Align', l_align_epoch, epoch)
        tb_writer.add_scalar('Loss/Margin', l_margin_epoch, epoch)
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

        print(f"Epoch {epoch+1:02d}/{epochs} | ClnLoss:{l_clean_epoch:.4f} RobLoss:{l_robust_epoch:.4f} Margin:{l_margin_epoch:.4f} | "
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

    print(f"\n--- SDT d-prime ---")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f}: d'={trades_dprimes[i]:.4f}")
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    # ── Step 4: AutoAttack Sweep on 1000 images with Class Breakdown ──
    print(f"\n{'='*70}\nAutoAttack Evaluation\n{'='*70}")
    run_autoattack(eval_model, testloader, epsilon=0.031, device=device, max_samples=1000)

    # ── Step 5: M-Pathway Dominance Check ──
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
