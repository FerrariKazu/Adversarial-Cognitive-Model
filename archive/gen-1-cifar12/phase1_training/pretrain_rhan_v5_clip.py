#!/usr/bin/env python3
"""
RHAN-v5 Phase 0: CLIP Semantic Initialization
==============================================
Trains the RHANv5 model from random initialization with ONLY clean CLIP alignment
for 30 epochs. No adversarial examples. No neural alignment loss.

Loss = CrossEntropy(logits, labels) + 0.5 * (1 - cosine_sim(features, clip_text[labels]))

Saves to: checkpoints/rhan_v5_clip_init.pth
This checkpoint has CLIP semantic priors baked into the weights.
CLIP will NOT be used again in Phase 1.

Expected: ~85% clean accuracy, high semantic organization.
"""

import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders

import clip


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v5_clip_init.pth')

    # ── Model: RHANv5 — randomly initialized ──
    model = RHANv5(head_type='cosine').to(device)
    print("RHANv5 initialized randomly (from scratch).")

    # ── CLIP text embeddings (frozen) ──
    print("Loading CLIP ViT-B/32 for semantic grounding...")
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    PROMPTS = [
        "a photo of an airplane", "a photo of an automobile",
        "a photo of a bird", "a photo of a cat", "a photo of a deer",
        "a photo of a dog", "a photo of a frog", "a photo of a horse",
        "a photo of a ship", "a photo of a truck",
    ]
    with torch.no_grad():
        text_tokens = clip.tokenize(PROMPTS).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=1)  # (10, 512)
    print(f"  CLIP text embeddings: {text_features.shape}")

    # ── DataLoaders ──
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=128, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader = DataLoader(testloader_raw.dataset, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False)

    # ── Training config ──
    epochs = 30
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    print(f"\n{'='*70}")
    print("RHAN-v5 Phase 0: CLIP Semantic Initialization")
    print(f"{'='*70}")
    print(f"  Architecture:  RHANv5 (frequency-separated, ventral/dorsal)")
    print(f"  Initialization:Random (Scratch)")
    print(f"  Objective:     CE + 0.5 * CLIP semantic alignment")
    print(f"  Adversarial:   NONE (clean images only)")
    print(f"  Optimizer:     AdamW (lr=0.001, wd=0.05)")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    128")
    print(f"  Save to:       {output_ckpt}")
    print(f"{'='*70}\n")

    best_test_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        s_ce = s_clip = s_total = 0.0
        train_correct = train_total = 0

        for imgs, labels in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                logits, features = model.forward_with_features(imgs)

                # 1. Classification CE
                loss_ce = F.cross_entropy(logits, labels)

                # 2. CLIP semantic alignment
                rhan_proj = F.normalize(features, dim=1)
                target_text = text_features[labels]
                loss_clip = (1.0 - F.cosine_similarity(rhan_proj, target_text)).mean()

                total_loss = loss_ce + 0.5 * loss_clip

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            s_ce += loss_ce.item() * B
            s_clip += loss_clip.item() * B
            s_total += total_loss.item() * B
            _, pred = logits.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        scheduler.step()

        N = len(trainloader.dataset)
        l_ce = s_ce / N; l_clip_val = s_clip / N; l_total = s_total / N
        train_acc = 100.0 * train_correct / max(train_total, 1)

        # ── Test accuracy ──
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0)
                test_correct += pred.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        if test_acc >= best_test_acc:
            torch.save(model.state_dict(), output_ckpt)
            best_test_acc = test_acc
            marker = ' ★ BEST'
        else:
            marker = ''

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"CE:{l_ce:.4f} CLIP:{l_clip_val:.4f} Tot:{l_total:.4f} | "
              f"Train:{train_acc:.1f}% Test:{test_acc:.2f}% | "
              f"LR:{optimizer.param_groups[0]['lr']:.6f} | "
              f"{time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\n{'='*70}")
    print(f"Phase 0 complete. Best test accuracy: {best_test_acc:.2f}%")
    print(f"Checkpoint saved to: {output_ckpt}")
    print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}\n")
    print("Next step: run train_rhan_v5.py (Phase 1 curriculum training)")


if __name__ == '__main__':
    main()
