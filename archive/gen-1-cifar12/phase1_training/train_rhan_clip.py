#!/usr/bin/env python3
"""
RHAN + CLIP Semantic Grounding Fine-Tuning & Evaluation (Trial 1)
==================================================================

SCIENTIFIC ARGUMENTS & MECHANISMS:

1. Why InfoNCE loss encourages semantic clustering:
   - Traditional cross-entropy classification treats classes as mutually exclusive categories.
     Under attack, the model's logits can be easily perturbed to shift across boundaries.
   - InfoNCE / NT-Xent loss operates on the unit hypersphere, maximizing the alignment of
     positive pairs (image_i, correct_text_i) while minimizing similarity with all negative text
     prompts within the batch.
   - Geometrically, this forces the network to project visual representations of a class into a
     highly cohesive "semantic cluster" centered around the linguistic concept anchor. It makes the
     learned representations continuous and clustered, preventing isolated texture-correlated shortcuts.

2. Why MSE on features maintains adversarial robustness:
   - Adversarial attacks work by adding high-frequency noise that shifts representations in the
     hidden layer's latent space, which easily tricks standard classification heads.
   - The adversarial consistency loss penalizes deviations in the recurrently-modulated visual
     features between clean and adversarial inputs using Mean Squared Error (MSE):
     L_adv_consistency = MSE(features(x_clean), features(x_adv))
   - This explicitly regularizes the visual backbone's latent space, forcing the representation of
     an adversarial image to lie extremely close to its clean counterpart. It prevents the semantic
     grounding projection head from breaking the robustness properties inherited from RHAN-adv.

3. What "aligned in CLIP space" means geometrically:
   - It means that both visual feature vectors and text prompt embeddings are normalized to L2 norm = 1.0.
   - Visual and text vectors reside on the same unit hypersphere. The angle between them represents
     semantic similarity.
   - By alignment, we are rotating the visual representation space such that the visual direction of
     "cat" points directly at the CLIP text embedding direction of "a photo of a cat".
   - Because L2 normalization is applied, magnitude-based adversarial inflation has zero effect on
     the final classification decision.

4. The Prediction:
   - If semantic grounding successfully replaces fragile texture classification with linguistic
     concept alignment, the adversarial robust threshold (ε where accuracy is 50%) is predicted to
     increase from the current ε ≈ 0.053 (with d' = 1.0 threshold at ε ≈ 0.076) toward ε ≈ 0.12+.
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_clip import RHANWithCLIP
from model_rhan import RHAN
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def info_nce_loss(image_features, text_features, scale, device):
    """
    Computes symmetric InfoNCE / NT-Xent loss on the unit hypersphere.
    
    image_features: (B, 512) normalized
    text_features: (B, 512) normalized (correct class text embedding for each image)
    scale: scalar learnable contrastive temperature scale
    """
    B = image_features.size(0)
    logits = scale * (image_features @ text_features.T)
    labels = torch.arange(B, device=device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss


def generate_pgd_3(model, imgs, labels, eps, alpha, device, cifar_min, cifar_max):
    """
    Fast 3-step PGD attack for adversarial consistency loss during training.
    Freezes model parameter gradients to avoid double-backward and version mismatches.
    """
    was_training = model.training
    model.eval()
    
    # Temporarily disable requires_grad on model parameters to avoid backward graph tracking issues
    orig_grad_states = []
    for param in model.parameters():
        orig_grad_states.append(param.requires_grad)
        param.requires_grad = False
    
    delta = torch.zeros_like(imgs).uniform_(-eps, eps).to(device)
    delta = torch.clamp(delta, -(imgs - cifar_min), (cifar_max - imgs))
    delta.requires_grad_(True)
    
    for _ in range(3):
        with autocast():
            logits = model(imgs + delta)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = delta.data + alpha * grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(imgs + delta.data, cifar_min, cifar_max) - imgs
        delta.grad.zero_()
        
    # Restore requires_grad states
    for param, state in zip(model.parameters(), orig_grad_states):
        param.requires_grad = state
        
    if was_training:
        model.train()
        
    return (imgs + delta).detach()


def main():
    set_seed(42)
    total_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimize GPU operations
    torch.backends.cudnn.benchmark = True

    # Checkpoint paths
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    rhan_ckpt = os.path.join(ckpt_dir, 'rhan_adv_best.pth')

    if not os.path.exists(rhan_ckpt):
        print(f"ERROR: Base RHAN-adv checkpoint not found at {rhan_ckpt}")
        return

    # 1. Initialize RHANWithCLIP model and load weights
    model = RHANWithCLIP(rhan_checkpoint_path=rhan_ckpt, device=device).to(device)

    # 2. Setup DataLoaders
    # Batch size 128 as requested
    trainloader_raw, testloader_raw = get_dataloaders(
        batch_size=128, num_workers=4, model_name='resnet'
    )
    trainloader = DataLoader(
        trainloader_raw.dataset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
        prefetch_factor=2,
    )

    # CIFAR bounds for attacks
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # 3. Setup Optimizer & Scheduler
    epochs = 30
    # Optimize: projection head + logit_scale + RHAN backbone unfrozen
    # (CLIP parameters have requires_grad=False and are automatically excluded)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=0.00003, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # TensorBoard setup
    tb_writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), '..', 'runs', 'rhan_clip'))

    # Compile model using torch.compile
    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    best_test_acc = 0.0
    output_ckpt_path = os.path.join(ckpt_dir, 'rhan_clip_best.pth')

    print(f"\n{'='*70}")
    print(f"RHAN + CLIP Semantic Grounding Fine-Tuning (Trial 1)")
    print(f"{'='*70}")
    print(f"  Base checkpoint:   {rhan_ckpt}")
    print(f"  Optimizer:         AdamW (lr=3e-5, wd=0.05)")
    print(f"  Scheduler:         CosineAnnealingLR (T_max={epochs})")
    print(f"  Batch size:        128")
    print(f"  Epochs:            {epochs}")
    print(f"  Loss Weights:      CE = 1.0, InfoNCE = 0.5, Adv Consistency = 0.1")
    print(f"  AMP & Compile:     Enabled")
    print(f"{'='*70}\n")

    # Fixed training attack parameters for PGD-3
    train_eps = 0.05
    train_alpha = 0.01

    for epoch in range(epochs):
        epoch_start = time.time()
        compiled_model.train()
        
        epoch_ce_loss = 0.0
        epoch_nce_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_total_loss = 0.0
        
        train_correct = 0
        train_total = 0

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # A. Generate PGD-3 adversarial examples
            adv_imgs = generate_pgd_3(
                compiled_model, imgs, labels,
                eps=train_eps, alpha=train_alpha,
                device=device, cifar_min=cifar_min, cifar_max=cifar_max
            )

            # B. Forward passes & loss computation
            with autocast():
                # Get clean features (only runs backbone ONCE)
                features_clean = compiled_model.get_feature_vector(imgs)
                
                # Reconstruct clean logits from the features without running backbone again
                scale = compiled_model.logit_scale.exp().clamp(max=100)
                logits_clean = scale * features_clean @ compiled_model.text_features.T
                
                # Get adversarial features (runs backbone once)
                features_adv = compiled_model.get_feature_vector(adv_imgs)
                
                # 1. Primary Cross Entropy Loss on contrastive logits
                loss_ce = F.cross_entropy(logits_clean, labels)
                
                # 2. Contrastive alignment loss (InfoNCE)
                correct_text_features = compiled_model.text_features[labels]
                loss_nce = info_nce_loss(features_clean, correct_text_features, scale, device)
                
                # 3. Adversarial consistency loss
                loss_adv_consistency = F.mse_loss(features_clean, features_adv)
                
                # Total dynamic loss composition
                total_loss = loss_ce + 0.5 * loss_nce + 0.1 * loss_adv_consistency

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Stats gathering
            epoch_ce_loss += loss_ce.item() * imgs.size(0)
            epoch_nce_loss += loss_nce.item() * imgs.size(0)
            epoch_adv_loss += loss_adv_consistency.item() * imgs.size(0)
            epoch_total_loss += total_loss.item() * imgs.size(0)

            _, predicted = logits_clean.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Normalize metrics
        N = len(trainloader.dataset)
        epoch_ce_loss /= N
        epoch_nce_loss /= N
        epoch_adv_loss /= N
        epoch_total_loss /= N
        train_acc = 100.0 * train_correct / train_total

        # Real-time evaluation on clean validation set
        compiled_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs = compiled_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total

        # Logging to TensorBoard
        tb_writer.add_scalar('Loss/CrossEntropy', epoch_ce_loss, epoch)
        tb_writer.add_scalar('Loss/InfoNCE', epoch_nce_loss, epoch)
        tb_writer.add_scalar('Loss/AdvConsistency', epoch_adv_loss, epoch)
        tb_writer.add_scalar('Loss/Total', epoch_total_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)
        tb_writer.add_scalar('Params/LogitScale', model.logit_scale.item(), epoch)

        # Checkpoint saving
        if test_acc > best_test_acc:
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), output_ckpt_path)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"CE: {epoch_ce_loss:.4f} | NCE: {epoch_nce_loss:.4f} | Adv: {epoch_adv_loss:.4f} | "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.2f}% | "
              f"LR: {current_lr:.7f} | Temp: {1.0/scale.item():.4f} | Time: {elapsed:.1f}s{marker}", flush=True)

    # Save final completed checkpoint
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(raw_model.state_dict(), output_ckpt_path)
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"Fine-tuning complete. Model saved successfully to: {output_ckpt_path}")
    print(f"Total training time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")

    # 4. Immediate Evaluation Post Training
    print(f"\n{'='*70}")
    print(f"EVALUATING TRIAL 1 VERDICT — PGD-100 SPECTRUM")
    print(f"{'='*70}")

    # Load both models for clean evaluation
    # Set compile=False for eval loader compatibility or re-compile
    rhan_adv = RHAN(num_classes=10, head_type='linear').to(device)
    rhan_adv.load_state_dict(torch.load(rhan_ckpt, map_location=device))
    rhan_adv.eval()
    rhan_adv = torch.compile(rhan_adv)

    rhan_clip = RHANWithCLIP(rhan_checkpoint_path=rhan_ckpt, device=device).to(device)
    rhan_clip.load_state_dict(torch.load(output_ckpt_path, map_location=device))
    rhan_clip.eval()
    rhan_clip = torch.compile(rhan_clip)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]

    def run_eval_pgd(eval_model, loader, eps_val, steps=100, max_samples=512):
        if eps_val == 0:
            correct = total = 0
            with torch.no_grad():
                for images, labels in loader:
                    if total >= max_samples:
                        break
                    images, labels = images.to(device), labels.to(device)
                    outputs = eval_model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            return 100. * correct / total

        a = max(eps_val / 10, 0.001)
        correct = total = 0
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            _, predicted = pgd_attack(
                eval_model, images, labels,
                epsilon=eps_val, alpha=a, steps=steps,
                device=device, clip_min=cifar_min, clip_max=cifar_max,
                random_start=True
            )
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    print("Evaluating PGD-100 for RHAN-adv...")
    adv_results = {}
    for eps in epsilons:
        acc = run_eval_pgd(rhan_adv, testloader, eps_val=eps, steps=100)
        adv_results[eps] = acc
        print(f"  ε={eps:.2f} → {acc:.2f}%")

    print("\nEvaluating PGD-100 for RHAN-CLIP...")
    clip_results = {}
    for eps in epsilons:
        acc = run_eval_pgd(rhan_clip, testloader, eps_val=eps, steps=100)
        clip_results[eps] = acc
        print(f"  ε={eps:.2f} → {acc:.2f}%")

    # Print Trial 1 Verdict Table
    print(f"\n{'='*70}")
    print(f"TRIAL 1 VERDICT: PGD-100 ROBUSTNESS SPECTRUM")
    print(f"{'='*70}")
    print(f"{'ε':<8} | {'RHAN-adv':>12} | {'RHAN-CLIP':>12} | {'Delta':>12}")
    print("-" * 52)
    for eps in epsilons:
        adv_acc = adv_results[eps]
        clip_acc = clip_results[eps]
        delta = clip_acc - adv_acc
        print(f"{eps:<8.2f} | {adv_acc:>11.2f}% | {clip_acc:>11.2f}% | {delta:>+11.2f}%")
    print(f"{'='*70}\n")

    # 5. Gradient Masking Check at ε=0.05
    print(f"\n{'='*70}")
    print(f"GRADIENT MASKING DIAGNOSTIC (ε=0.05)")
    print(f"{'='*70}")
    print("Evaluating PGD-20 for RHAN-CLIP...")
    pgd_20_acc = run_eval_pgd(rhan_clip, testloader, eps_val=0.05, steps=20)
    pgd_100_acc = clip_results[0.05]
    gap = pgd_20_acc - pgd_100_acc
    print(f"  PGD-20 Accuracy:  {pgd_20_acc:.2f}%")
    print(f"  PGD-100 Accuracy: {pgd_100_acc:.2f}%")
    print(f"  Robustness Gap:   {gap:.2f}%")
    if gap < 5.0:
        print("  ✓ PASS: Gap is < 5%. No significant gradient masking detected.")
    else:
        print("  ❌ WARNING: Gap is >= 5%. The model may be exhibiting gradient masking / artificial robustness.")
    print(f"{'='*70}\n")

    tb_writer.close()


if __name__ == '__main__':
    main()
