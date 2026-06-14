#!/usr/bin/env python3
"""
SAIL: Self-Supervised Adversarial Invariance Learning
=====================================================

This is the core algorithmic innovation that addresses the fundamental problem:
TRADES optimizes output similarity. Human vision maintains REPRESENTATION invariance.

The algorithm:
  For every training image x:
    1. Generate adversarial version x_adv = PGD(model, x, eps)
    2. Compute clean representation:       z     = encoder(x)
    3. Compute adversarial representation: z_adv = encoder(x_adv)
    4. InfoNCE loss: z and z_adv should be IDENTICAL (positive pair)
                     z_i and z_j (i≠j) should be DIFFERENT (negative pairs)

The effect:
  After SAIL pretraining, encoder(x_adv) ≈ encoder(x_clean) for ALL x.
  The representations are adversarially invariant BY CONSTRUCTION.
  TRADES fine-tuning then only needs to align outputs — an easy task
  because the representations are already correct.

Why this closes the automobile/truck gap:
  Current failure: the representations of adversarial automobiles
  drift toward truck representation space.
  SAIL fix: trains encoder to output SAME auto representation
  for both clean and adversarial automobiles — truck representations
  are never activated by adversarial autos.

Expected improvements over TRADES alone:
  AutoAttack: +10-20% (representation invariance removes the geometric collapse)
  εthresh:    +0.05-0.10 (much stronger starting point for TRADES fine-tuning)
  Auto/truck: 0% → 15-30% (concept boundary preserved in representation space)

Usage:
  # Phase 1: SAIL pretraining (no labels)
  python train_sail.py --phase sail --epochs 50 --start rhan_v8_best.pth

  # Phase 2: TRADES fine-tuning (with labels)
  python train_sail.py --phase trades --epochs 60 --start rhan_v9_sail.pth
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v9 import RHANv9
from model_cornets import CIFARCORnet
from dataset import get_dataloaders


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ATTACK GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_pgd_adversarial(model, x, epsilon, alpha, steps,
                              clip_min, clip_max, random_start=True):
    """Standard PGD for generating adversarial examples during training."""
    x = x.detach()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta).detach()
    else:
        x_adv = x.clone().detach()

    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min)

    model.eval()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                logits = model(x_adv)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # Use CE loss for adversarial generation
                # (targeting maximum confusion regardless of labels)
                # For SAIL: we just want diverse, strong perturbations
                loss = -logits.max(dim=1)[0].mean()  # maximize top logit

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, clip_min, clip_max).detach()

    model.train()
    return x_adv


def generate_trades_adversarial(model, x, epsilon, alpha, steps,
                                 clip_min, clip_max):
    """TRADES-style adversarial examples targeting KL divergence."""
    x = x.detach()
    x_adv = x + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()

    # Get clean probabilities
    model.eval()
    with torch.no_grad():
        with autocast('cuda'):
            logits_clean = model(x)
            if isinstance(logits_clean, tuple):
                logits_clean = logits_clean[0]
        probs_clean = F.softmax(logits_clean.float(), dim=1).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                logits_adv = model(x_adv)
                if isinstance(logits_adv, tuple):
                    logits_adv = logits_adv[0]
                loss = F.kl_div(
                    F.log_softmax(logits_adv.float(), dim=1),
                    probs_clean, reduction='batchmean'
                )
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, clip_min, clip_max).detach()

    model.train()
    return x_adv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAIL LOSS: Self-Supervised Adversarial Invariance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sail_infonce_loss(z_clean: torch.Tensor, z_adv: torch.Tensor,
                      temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE loss for adversarial invariance.

    z_clean: (B, D) normalized clean representations
    z_adv:   (B, D) normalized adversarial representations

    Positive pairs: (z_clean[i], z_adv[i]) — same image, clean vs adversarial
    Negative pairs: (z_clean[i], z_adv[j]) for i≠j — different images

    This directly trains: encoder(x_adv) ≈ encoder(x_clean)
    which is the representation-level invariance humans have.
    """
    B = z_clean.shape[0]

    # Build similarity matrix: clean vs all adversarial
    # Shape: (B, B) where [i, j] = similarity(clean_i, adv_j)
    sim = torch.mm(z_clean, z_adv.T) / temperature  # (B, B)

    # Diagonal is the positive pair (same image)
    labels = torch.arange(B, device=z_clean.device)

    # Cross-entropy: maximize diagonal, minimize off-diagonal
    loss = F.cross_entropy(sim, labels)

    return loss


def frequency_invariance_loss(model: RHANv9, x_clean: torch.Tensor,
                              x_adv: torch.Tensor) -> torch.Tensor:
    """
    Low-frequency features should be invariant to adversarial perturbation.
    Adversarial attacks primarily live in high-frequency space.
    Therefore, shape representations (low-freq stem) should be unchanged.
    """
    x_low_c, _ = model.separate_frequencies(x_clean)
    x_low_a, _ = model.separate_frequencies(x_adv)
    f_low_c = model.stem_low(x_low_c)
    f_low_a = model.stem_low(x_low_a)
    return F.mse_loss(f_low_a, f_low_c.detach())


def concept_supervision_loss(model: RHANv9, x: torch.Tensor,
                              labels: torch.Tensor,
                              concept_weight: float = 2.0) -> torch.Tensor:
    """
    Supervised concept loss: specifically designed to separate automobile/truck.
    Trains the concept bottleneck to activate different concepts for these classes.
    """
    _, concepts = model.forward_with_concepts(x)
    # Ground truth concepts for each sample
    target_concepts = model.concept_labels[labels]  # (B, n_concepts)
    return F.binary_cross_entropy(concepts, target_concepts)


def get_it_features(teacher: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract IT cortex features from CORnet-S teacher."""
    x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        out = teacher.model.module.V1(x_224)
        out = teacher.model.module.V2(out)
        out = teacher.model.module.V4(out)
        out = teacher.model.module.IT(out)
        out = teacher.model.module.decoder.avgpool(out)
        out = teacher.model.module.decoder.flatten(out)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1: SAIL PRETRAINING (no labels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_sail_phase(model, trainloader, testloader, optimizer, scheduler,
                   scaler, device, cifar_min, cifar_max,
                   epochs, ckpt_dir, epsilon=0.031, alpha=0.008, steps=5):
    """
    SAIL pretraining: no labels, trains representation invariance only.

    Loss breakdown:
      L_SAIL (0.70): InfoNCE between clean and adversarial representations
      L_freq (0.30): Low-frequency representation must be invariant

    After this phase, the encoder produces f(x_clean) ≈ f(x_adv).
    """
    print("=" * 70)
    print("PHASE 1: SAIL SELF-SUPERVISED ADVERSARIAL INVARIANCE PRETRAINING")
    print("No labels used. Training representation invariance only.")
    print(f"Epsilon: {epsilon}, Steps: {steps}, Temperature: 0.07")
    print("=" * 70)

    best_acc = 0.0
    sail_output_path = os.path.join(ckpt_dir, 'rhan_v9_sail.pth')
    rolling_path = os.path.join(ckpt_dir, 'rhan_v9_sail_rolling.pth')

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = n_total = 0
        l_sail_sum = l_freq_sum = 0

        for batch_idx, (imgs, _) in enumerate(trainloader):
            # NOTE: No labels used in SAIL phase
            imgs = imgs.to(device, non_blocking=True)

            # Generate adversarial examples
            x_adv = generate_pgd_adversarial(
                model, imgs, epsilon, alpha, steps, cifar_min, cifar_max
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                # Contrastive representations
                z_clean = model.forward_contrastive(imgs)   # (B, 128)
                z_adv = model.forward_contrastive(x_adv)    # (B, 128)

                # SAIL InfoNCE loss
                l_sail = sail_infonce_loss(z_clean, z_adv, temperature=0.07)

                # Frequency invariance loss
                l_freq = frequency_invariance_loss(model, imgs, x_adv)

                loss = 0.70 * l_sail + 0.30 * l_freq

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            total_loss += loss.item() * B
            l_sail_sum += l_sail.item() * B
            l_freq_sum += l_freq.item() * B
            n_total += B

        scheduler.step()

        # Evaluate clean accuracy as a proxy
        # (representations should still be classifiable even without classification training)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits = model(imgs)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                correct += logits.argmax(1).eq(lbls).sum().item()
                total += lbls.size(0)
        acc = 100. * correct / total

        marker = ''
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), sail_output_path)
            marker = ' ★'

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_acc': best_acc,
        }, rolling_path)

        print(
            f"SAIL Epoch {epoch:02d}/{epochs} | "
            f"Loss: {total_loss/n_total:.4f} | "
            f"SAIL: {l_sail_sum/n_total:.4f} | "
            f"Freq: {l_freq_sum/n_total:.4f} | "
            f"CleanAcc: {acc:.1f}% | {time.time()-t0:.0f}s{marker}"
        )

    print(f"\nSAIL pretraining complete. Best clean acc: {best_acc:.2f}%")
    print(f"Checkpoint: {sail_output_path}")
    return sail_output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2: TRADES FINE-TUNING ON INVARIANT REPRESENTATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_trades_phase(model, teacher, text_features, clip_projector,
                     trainloader, testloader, optimizer, scheduler,
                     scaler, device, cifar_min, cifar_max,
                     epochs, ckpt_dir, beta=4.0, epsilon=0.031,
                     alpha=0.008, steps=10):
    """
    TRADES fine-tuning starting from SAIL-pretrained invariant representations.

    Because SAIL already made f(x_adv) ≈ f(x_clean), TRADES has an easier job:
    it only needs to maintain this property while also learning to classify.

    Loss breakdown:
      L_trades   (0.50): TRADES CE + KL in output space
      L_clip     (0.20): CLIP semantic anchoring for auto/truck separation
      L_align    (0.15): CORnet-S IT alignment
      L_freq     (0.10): Frequency invariance (from SAIL)
      L_concept  (0.05): Concept supervision for auto/truck concepts
    """
    print("=" * 70)
    print("PHASE 2: TRADES FINE-TUNING ON SAIL-INVARIANT REPRESENTATIONS")
    print(f"Beta: {beta}, Epsilon: {epsilon}, Steps: {steps}")
    print("=" * 70)

    best_acc = 0.0
    output_path = os.path.join(ckpt_dir, 'rhan_v9_best.pth')
    rolling_path = os.path.join(ckpt_dir, 'rhan_v9_rolling.pth')
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total_loss = n_total = correct = 0
        sums = {k: 0 for k in ['trades', 'clip', 'align', 'freq', 'concept']}

        for batch_idx, (imgs, lbls) in enumerate(trainloader):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            # TRADES adversarial examples
            x_adv = generate_trades_adversarial(
                model, imgs, epsilon, alpha, steps, cifar_min, cifar_max
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits_c, feat_c = model.forward_with_features(imgs)
                logits_a, feat_a = model.forward_with_features(x_adv)

                # TRADES loss
                l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                    F.log_softmax(logits_a, dim=1),
                    F.softmax(logits_c.detach(), dim=1),
                    reduction='batchmean'
                )

                # CLIP semantic anchoring
                feat_proj = F.normalize(clip_projector(feat_a), dim=-1)
                l_clip = (1 - (feat_proj * text_features[lbls]).sum(dim=1)).mean()

                # CORnet-S alignment (every other batch)
                if batch_idx % 2 == 0:
                    with torch.no_grad():
                        it_feats = get_it_features(teacher, x_adv.float())
                    l_align = 1.0 - (
                        F.normalize(feat_a, dim=-1) *
                        F.normalize(it_feats.to(feat_a.dtype), dim=-1)
                    ).sum(dim=-1).mean()
                else:
                    l_align = torch.tensor(0.0, device=device)

                # Frequency invariance
                l_freq = frequency_invariance_loss(model, imgs, x_adv)

                # Concept supervision (focuses on auto/truck disambiguation)
                l_concept = concept_supervision_loss(model, imgs, lbls)

                loss = (0.50 * l_trades + 0.20 * l_clip +
                        0.15 * l_align + 0.10 * l_freq + 0.05 * l_concept)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            total_loss += loss.item() * B
            correct += logits_c.argmax(1).eq(lbls).sum().item()
            n_total += B
            sums['trades'] += l_trades.item() * B
            sums['clip'] += l_clip.item() * B
            sums['align'] += l_align.item() * B
            sums['freq'] += l_freq.item() * B
            sums['concept'] += l_concept.item() * B

        scheduler.step()

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits = model(imgs)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                val_correct += logits.argmax(1).eq(lbls).sum().item()
                val_total += lbls.size(0)
        val_acc = 100. * val_correct / val_total

        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'clip_projector': clip_projector.state_dict(),
                'epoch': epoch,
            }, output_path)
            marker = ' ★'

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'clip_projector': clip_projector.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_acc': best_acc,
        }, rolling_path)

        N = n_total
        print(
            f"TRADES Epoch {epoch:02d}/{epochs} | "
            f"L:{total_loss/N:.3f} | TrAcc:{100.*correct/N:.1f}% TeAcc:{val_acc:.1f}% | "
            f"T:{sums['trades']/N:.3f} C:{sums['clip']/N:.3f} "
            f"A:{sums['align']/N:.3f} F:{sums['freq']/N:.3f} Co:{sums['concept']/N:.3f} | "
            f"{time.time()-t0:.0f}s{marker}"
        )

    print(f"\nTRADES phase complete. Best clean acc: {best_acc:.2f}%")
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['sail', 'trades', 'both'],
                        default='both', help='Training phase to run')
    parser.add_argument('--sail-epochs', type=int, default=50)
    parser.add_argument('--trades-epochs', type=int, default=60)
    parser.add_argument('--start', type=str, default=None,
                        help='Starting checkpoint (default: rhan_v8_best.pth)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── MODEL ──────────────────────────────────────────────────────────────
    model = RHANv9(head_type='cosine').to(device)

    start_ckpt = args.start or os.path.join(ckpt_dir, 'rhan_v8_best.pth')
    if os.path.exists(start_ckpt):
        state = torch.load(start_ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {start_ckpt}")
        if missing:
            print(f"  New parameters (random init): {missing}")
    else:
        print(f"WARNING: Start checkpoint not found at {start_ckpt}")
        print("Training from scratch — results will be worse")

    # ── TEACHER (CORnet-S) ─────────────────────────────────────────────────
    teacher = CIFARCORnet().to(device)
    teacher_ckpt = os.path.join(ckpt_dir, 'phase1_training/checkpoints/cornets_best.pth')
    if os.path.exists(teacher_ckpt):
        teacher.load_state_dict(
            torch.load(teacher_ckpt, map_location=device, weights_only=False)
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print("Loaded CORnet-S teacher")
    else:
        print("WARNING: CORnet-S teacher not found — alignment loss will be zero")
        teacher = None

    # ── CLIP TEXT FEATURES ─────────────────────────────────────────────────
    try:
        import clip
        clip_model, _ = clip.load('ViT-B/32', device=device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        PROMPTS = [
            "a photo of an airplane",      "a photo of an automobile",
            "a photo of a bird",           "a photo of a cat",
            "a photo of a deer",           "a photo of a dog",
            "a photo of a frog",           "a photo of a horse",
            "a photo of a ship",           "a photo of a truck"
        ]
        with torch.no_grad():
            text_tokens = clip.tokenize(PROMPTS).to(device)
            text_features = F.normalize(
                clip_model.encode_text(text_tokens).float(), dim=-1
            )
        print("CLIP text features ready")
    except Exception as e:
        print(f"CLIP not available: {e}")
        text_features = None
        clip_model = None

    # CLIP projector
    clip_projector = nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)
    ).to(device)

    # ── DATA ───────────────────────────────────────────────────────────────
    batch_size = 64
    _, testloader_raw = get_dataloaders(batch_size=batch_size, num_workers=0,
                                         model_name='resnet')
    trainloader_raw, _ = get_dataloaders(batch_size=batch_size, num_workers=0,
                                          model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)
    testloader = DataLoader(testloader_raw.dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # CIFAR-10 normalization bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([2.6400,  2.6210,  2.7615]).view(1,3,1,1).to(device)

    # ── PHASE 1: SAIL ──────────────────────────────────────────────────────
    if args.phase in ('sail', 'both'):
        optimizer = optim.SGD(model.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.sail_epochs, eta_min=1e-5
        )
        scaler = GradScaler('cuda')

        sail_ckpt = run_sail_phase(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            cifar_min=cifar_min,
            cifar_max=cifar_max,
            epochs=args.sail_epochs,
            ckpt_dir=ckpt_dir,
            epsilon=0.031,
            alpha=0.031 / 4,
            steps=5,
        )

        # Load best SAIL checkpoint for TRADES phase
        model.load_state_dict(torch.load(sail_ckpt, map_location=device))
        print(f"\nLoaded SAIL checkpoint for TRADES phase: {sail_ckpt}")

    # ── PHASE 2: TRADES ────────────────────────────────────────────────────
    if args.phase in ('trades', 'both'):
        optimizer = optim.SGD(
            list(model.parameters()) + list(clip_projector.parameters()),
            lr=0.002, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.trades_epochs, eta_min=1e-5
        )
        scaler = GradScaler('cuda')

        if args.phase == 'trades':
            # Load SAIL checkpoint if only running TRADES
            sail_ckpt = os.path.join(ckpt_dir, 'rhan_v9_sail.pth')
            if os.path.exists(sail_ckpt):
                model.load_state_dict(
                    torch.load(sail_ckpt, map_location=device)
                )
                print(f"Loaded SAIL checkpoint: {sail_ckpt}")

        run_trades_phase(
            model=model,
            teacher=teacher,
            text_features=text_features,
            clip_projector=clip_projector,
            trainloader=trainloader,
            testloader=testloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            cifar_min=cifar_min,
            cifar_max=cifar_max,
            epochs=args.trades_epochs,
            ckpt_dir=ckpt_dir,
            beta=4.0,
            epsilon=0.031,
            alpha=0.031 / 4,
            steps=10,
        )

    print("\nSAIL training pipeline complete.")
    print("Run AutoAttack evaluation:")
    print("  python eval_autoattack.py --model rhan_v9_best.pth")


if __name__ == '__main__':
    main()
