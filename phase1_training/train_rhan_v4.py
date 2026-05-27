#!/usr/bin/env python3
"""
RHAN-v4: Multi-Scale Feedback + CLIP Semantic + Contrastive Adversarial Training
=================================================================================
Architecture: RHANv4 (ventral/dorsal + multi-scale feedback + CLIP projection)
Initialization: Random (from scratch)
Loss (5 components from epoch 1):
  0.40 * adv_CE (PGD-5) +
  0.15 * clean_CE +
  0.20 * align_on_adv (CORnet-S IT) +
  0.15 * clip_semantic (CLIP text embeddings) +
  0.10 * infonce_contrastive
"""

import os, sys, time, random
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

from model_rhan_v4 import RHANv4
from model_cornets import CIFARCORnet
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

import clip


# =============================================================================
# Helpers
# =============================================================================

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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


def info_nce_loss(clean_feat, adv_feat, temperature=0.07):
    """
    InfoNCE contrastive loss.
    Treats (clean_i, adv_i) as positive pairs; all cross-sample as negatives.
    """
    B = clean_feat.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=clean_feat.device)

    clean_norm = F.normalize(clean_feat, dim=1)
    adv_norm = F.normalize(adv_feat, dim=1)

    # Similarity matrix (2B × 2B)
    feats = torch.cat([clean_norm, adv_norm], dim=0)  # (2B, 512)
    sim = feats @ feats.T / temperature

    # Labels: positive pair for sample i is at index i+B (and vice versa)
    labels = torch.arange(B, device=clean_feat.device)
    labels = torch.cat([labels + B, labels])  # (2B,)

    # Mask self-similarity
    mask = torch.eye(2 * B, device=clean_feat.device).bool()
    sim.masked_fill_(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    cornet_ckpt = os.path.join(script_dir, 'checkpoints', 'cornets_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v4_best.pth')

    if not os.path.exists(cornet_ckpt):
        print(f"ERROR: CORnet-S checkpoint not found at {cornet_ckpt}"); return

    # ── Model: RHANv4 — randomly initialized ──
    model = RHANv4(head_type='cosine').to(device)
    print("RHANv4 initialized randomly (from scratch).")

    # ── CORnet-S teacher (frozen) ──
    teacher = CIFARCORnet().to(device)
    teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=64, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=64, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # ── Training config ──
    epochs = 100; warmup_epochs = 15; accum_steps = 2
    base_lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lr)
    scaler = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_v4'))

    # Loss weights
    W_ADV = 0.40; W_CLEAN = 0.15; W_ALIGN = 0.20; W_CLIP = 0.15; W_NCE = 0.10

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    print(f"\n{'='*70}")
    print("RHAN-v4 · Joint Training from Scratch")
    print(f"{'='*70}")
    print(f"  Architecture:  RHANv4 (ventral/dorsal + multi-scale feedback)")
    print(f"  Initialization:Random (Scratch)")
    print(f"  Teachers:      CORnet-S (IT alignment) + CLIP ViT-B/32 (semantic)")
    print(f"  Optimizer:     AdamW (base_lr={base_lr}, wd=0.05)")
    print(f"  Warmup:        {warmup_epochs} epochs")
    print(f"  Batch:         64 × {accum_steps} accum = 128 effective")
    print(f"  Epochs:        {epochs}")
    print(f"  Loss weights:  adv={W_ADV} clean={W_CLEAN} align={W_ALIGN} clip={W_CLIP} nce={W_NCE}")
    print(f"  Alignment:     ON ADVERSARIAL IMAGES")
    print(f"  Save to:       {output_ckpt}")
    print(f"{'='*70}\n")

    best_test_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)
        compiled_model.train()
        s_adv = s_clean = s_align = s_clip = s_nce = s_total = 0.0
        train_correct = train_total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0); half = B // 2

            if half > 0:
                # Save gradients before PGD
                saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        compiled_model, imgs[:half], labels[:half],
                        epsilon=0.031, alpha=0.006, steps=5,
                        device=device, clip_min=cifar_min, clip_max=cifar_max,
                        random_start=True)
                adv_imgs = adv_imgs.detach()

                # Restore gradients
                for p, g in zip(model.parameters(), saved_grads):
                    p.grad = g

                clean_imgs = imgs[half:]
                adv_labels = labels[:half]; cln_labels = labels[half:]
            else:
                adv_imgs = clean_imgs = imgs; adv_labels = cln_labels = labels

            with autocast('cuda'):
                # 1. Adversarial CE (40%)
                adv_logits, adv_feats = compiled_model.forward_with_features(adv_imgs)
                loss_adv = F.cross_entropy(adv_logits, adv_labels)

                # 2. Clean CE (15%)
                cln_logits, cln_feats = compiled_model.forward_with_features(clean_imgs)
                loss_clean = F.cross_entropy(cln_logits, cln_labels)

                # 3. Neural alignment ON ADVERSARIAL IMAGES (20%)
                with torch.no_grad():
                    it_feats = get_it_features(teacher, adv_imgs)
                loss_align = 1.0 - (F.normalize(adv_feats, dim=-1) * F.normalize(it_feats, dim=-1)).sum(dim=-1).mean()

                # 4. CLIP semantic contrastive (15%)
                # Project clean features into CLIP space
                rhan_proj = F.normalize(compiled_model.clip_projection(cln_feats), dim=1)
                target_text = text_features[cln_labels]  # (B_clean, 512)
                loss_clip = (1.0 - F.cosine_similarity(rhan_proj, target_text)).mean()

                # 5. InfoNCE contrastive adversarial consistency (10%)
                if half > 0:
                    with torch.no_grad():
                        clean_feats_ref = compiled_model.get_feature_vector(imgs[:half])
                    loss_nce = info_nce_loss(clean_feats_ref, adv_feats)
                else:
                    loss_nce = torch.tensor(0.0, device=device)

                # Combined loss
                total_loss = (W_ADV * loss_adv + W_CLEAN * loss_clean +
                              W_ALIGN * loss_align + W_CLIP * loss_clip +
                              W_NCE * loss_nce) / accum_steps

            scaler.scale(total_loss).backward()
            s_adv += loss_adv.item() * B; s_clean += loss_clean.item() * B
            s_align += loss_align.item() * B; s_clip += loss_clip.item() * B
            s_nce += loss_nce.item() * B; s_total += total_loss.item() * accum_steps * B
            _, pred = adv_logits.max(1)
            train_total += adv_labels.size(0); train_correct += pred.eq(adv_labels).sum().item()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(trainloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

        N = len(trainloader.dataset)
        l_adv = s_adv / N; l_clean = s_clean / N; l_align = s_align / N
        l_clip_val = s_clip / N; l_nce = s_nce / N; l_total = s_total / N
        train_acc = 100.0 * train_correct / max(train_total, 1)

        # ── Test accuracy ──
        compiled_model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = compiled_model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0); test_correct += pred.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        # ── TensorBoard logging ──
        tb_writer.add_scalar('Loss/Adv_CE', l_adv, epoch)
        tb_writer.add_scalar('Loss/Clean_CE', l_clean, epoch)
        tb_writer.add_scalar('Loss/Align', l_align, epoch)
        tb_writer.add_scalar('Loss/CLIP', l_clip_val, epoch)
        tb_writer.add_scalar('Loss/InfoNCE', l_nce, epoch)
        tb_writer.add_scalar('Loss/Total', l_total, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)

        if test_acc >= best_test_acc:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw.state_dict(), output_ckpt)
            best_test_acc = test_acc; marker = ' ★ BEST'
        else:
            marker = ''

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Adv:{l_adv:.4f} Cln:{l_clean:.4f} Alg:{l_align:.4f} "
              f"Clip:{l_clip_val:.4f} NCE:{l_nce:.4f} Tot:{l_total:.4f} | "
              f"Train:{train_acc:.1f}% Test:{test_acc:.2f}% | "
              f"LR:{current_lr:.7f} | {time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\n{'='*70}")
    print(f"Training complete. Saved to: {output_ckpt}")
    print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}\n")
    tb_writer.close()

    # =====================================================================
    # POST-TRAINING EVALUATION PIPELINE
    # =====================================================================
    print("Loading best checkpoint for evaluation...")
    eval_model = RHANv4(head_type='cosine').to(device)
    eval_model.load_state_dict(torch.load(output_ckpt, map_location=device))
    eval_model.eval()
    for p in eval_model.parameters():
        p.requires_grad = False

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    wrapper = W(eval_model)

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
            _, preds = eval_model(noisy).max(1)
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
            _, preds = eval_model(adv_images).max(1)
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
    print(f"\n{'='*70}\nRHAN-v4  PGD-100 Evaluation\n{'='*70}")
    v4_accs = []
    for eps in epsilons:
        t0 = time.time()
        print(f"Evaluating ε={eps:.2f}...", end=' ', flush=True)
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
                _, preds = eval_model(adv_images).max(1)
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        acc = 100.0 * correct / max(total, 1); v4_accs.append(acc)
        print(f"Acc:{acc:.2f}% | {time.time()-t0:.1f}s")

    # ── Step 3: PGD-20 vs PGD-100 gap check ──
    p100_05 = v4_accs[2]  # ε=0.05
    pgd_gap = p20_acc_05 - p100_05
    print(f"\nPGD-20 vs PGD-100 gap at ε=0.05: {pgd_gap:.2f}%")
    if pgd_gap < 8.0:
        print(f"  ✓ Gap ({pgd_gap:.2f}%) < 8% — no gradient masking")
    else:
        print(f"  ⚠ Gap ({pgd_gap:.2f}%) >= 8% — potential gradient masking")
        masking_detected = True

    if masking_detected:
        print("\n⚠  GRADIENT MASKING DETECTED — results may be unreliable!")

    # ── Step 4: SDT d-prime & εthresh ──
    import scipy.stats as stats

    v4_dprimes = []
    for acc_pct in v4_accs:
        acc = acc_pct / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        v4_dprimes.append(float(dp))

    # Find εthresh where d' crosses 1.0
    eps_thresh = None
    for i in range(len(v4_dprimes) - 1):
        d1, d2 = v4_dprimes[i], v4_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(v4_dprimes) > 0 and v4_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]

    # ── Step 5: Final verdict table ──
    rhan_adv = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    rhan_v3 = {0.00: 91.41, 0.01: 85.35, 0.05: 60.74, 0.10: 26.17, 0.20: 1.17, 0.30: 0.00}

    print(f"\n{'='*70}\nRHAN-v4  FINAL VERDICT\n{'='*70}")
    print(f"{'ε':<8} | {'Human':>8} | {'RHAN-v4':>8} | {'RHAN-v3':>8} | {'RHAN-adv':>8} | {'ResNet':>8} | {'ViT':>8}")
    print("-" * 75)
    human_accs = {0.00: 73.33, 0.01: 'N/A', 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}
    resnet_accs = {0.00: 95.82, 0.01: 75.57, 0.05: 2.84, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit_accs = {0.00: 97.80, 0.01: 55.18, 0.05: 8.80, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}

    for i, eps in enumerate(epsilons):
        h = human_accs[eps]
        h_str = f"{h:.2f}%" if isinstance(h, float) else h
        v4 = v4_accs[i]
        v3 = rhan_v3[eps]
        ra = rhan_adv[eps]
        rn = resnet_accs[eps]
        vt = vit_accs[eps]
        print(f"{eps:<8.2f} | {h_str:>8} | {v4:>7.2f}% | {v3:>7.2f}% | {ra:>7.2f}% | {rn:>7.2f}% | {vt:>7.2f}%")
    print("=" * 75)

    print(f"\n--- SDT d-prime ---")
    print(f"{'ε':<8} | {'d-prime':>8}")
    print("-" * 20)
    for i, eps in enumerate(epsilons):
        print(f"{eps:<8.2f} | {v4_dprimes[i]:>8.4f}")
    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    # ── Step 6: εthresh ranking ──
    print(f"\n{'='*70}")
    print("ROBUSTNESS RANKING (SDT ε_thresh where d' crosses 1.0)")
    print(f"{'='*70}")
    print(f"  {'System':<20} | {'ε_thresh':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Human':<20} | {'> 0.3000':>10}")
    print(f"  {'RHAN-v4':<20} | {thresh_str:>10}")
    print(f"  {'RHAN-v3':<20} | {'0.0900':>10}")
    print(f"  {'RHAN-adv':<20} | {'0.0764':>10}")
    print(f"  {'ResNet-18':<20} | {'0.0295':>10}")
    print(f"  {'ViT-Small':<20} | {'0.0264':>10}")
    print(f"{'='*70}")

    # ── Step 7: Alignment metrics ──
    print(f"\n--- RHAN-v4 ↔ CORnet-IT Cosine Similarity per Epsilon ---")
    for i, eps in enumerate(epsilons):
        # Use PGD images from eval if available, else clean
        cos_correct = cos_total = 0; cos_sum = 0.0
        for images, lbls in testloader:
            if cos_total >= 256: break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                alpha_e = max(eps / 10, 0.001)
                adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha_e,
                    steps=20, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv_images = images
            with torch.no_grad():
                rhan_feats = eval_model.get_feature_vector(adv_images)
                it_feats = get_it_features(teacher, adv_images)
                cos_sim = F.cosine_similarity(F.normalize(rhan_feats, dim=-1),
                                               F.normalize(it_feats, dim=-1)).mean().item()
                cos_sum += cos_sim * lbls.size(0); cos_total += lbls.size(0)
        avg_cos = cos_sum / max(cos_total, 1)
        print(f"  ε={eps:.2f}: cosine_sim = {avg_cos:.4f}")

    print(f"\n--- RHAN-v4 ↔ CLIP Text Cosine Similarity per Class ---")
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    class_cos = {c: [] for c in range(10)}
    with torch.no_grad():
        for images, lbls in testloader:
            images, lbls = images.to(device), lbls.to(device)
            feats = eval_model.get_feature_vector(images)
            proj = F.normalize(eval_model.clip_projection(feats), dim=1)
            for c in range(10):
                mask = lbls == c
                if mask.sum() > 0:
                    cos = F.cosine_similarity(proj[mask], text_features[c].unsqueeze(0)).mean().item()
                    class_cos[c].append(cos)
    for c in range(10):
        if class_cos[c]:
            avg = np.mean(class_cos[c])
            print(f"  {CLASSES[c]:<12}: {avg:.4f}")

    # ── Final summary ──
    v4_clean = v4_accs[0]; v4_005 = v4_accs[2]
    v3_clean = rhan_v3[0.00]; v3_005 = rhan_v3[0.05]
    print(f"\n{'='*70}")
    if eps_thresh is not None and eps_thresh > 0.120:
        print(f"🏆 TARGET ACHIEVED: ε_thresh = {thresh_str} > 0.120")
        if eps_thresh > 0.200:
            print(f"   → Submit to NeurIPS/ICLR")
        elif eps_thresh > 0.150:
            print(f"   → Submit to arXiv this week")
        else:
            print(f"   → Write the paper immediately")
    elif v4_005 > v3_005 and v4_clean > v3_clean:
        print(f"✅ RHAN-v4 improves over v3: clean {v4_clean:.2f}% vs {v3_clean:.2f}%, "
              f"ε=0.05 {v4_005:.2f}% vs {v3_005:.2f}%")
    else:
        print(f"⚠  RHAN-v4 did not uniformly improve over v3.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
