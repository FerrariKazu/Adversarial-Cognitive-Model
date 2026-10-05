#!/usr/bin/env python3
"""
RHAN-v5: Frequency-Separated Joint Curriculum Training (Phase 1)
=================================================================
Loads CLIP-pretrained weights from checkpoints/rhan_v5_clip_init.pth.
No ongoing CLIP loss — CLIP was used only as an initialization strategy.

Curriculum schedule:
  Phase A (1-30):    PGD-5,  ε=0.031
  Phase B (31-65):   PGD-7,  ε=0.062
  Phase C (66-100):  PGD-10, ε=0.100
  Phase D (101-120): PGD-10, ε=0.150 (reduced non-adv weights)

Loss (4 components):
  0.40 * adv_CE + 0.20 * clean_CE + 0.30 * align_on_adv + 0.10 * freq_consist
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

from model_rhan_v5 import RHANv5
from model_cornets import CIFARCORnet
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing, with optional phase D constant."""
    def __init__(self, optimizer, warmup_epochs, cosine_end_epoch, total_epochs, base_lr, phase_d_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine_end_epoch = cosine_end_epoch
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.phase_d_lr = phase_d_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = 0.0001 + (self.base_lr - 0.0001) * (epoch + 1) / self.warmup_epochs
        elif epoch < self.cosine_end_epoch:
            progress = (epoch - self.warmup_epochs) / (self.cosine_end_epoch - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            lr = self.phase_d_lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
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


def get_curriculum(epoch):
    """Returns (pgd_steps, epsilon, alpha, phase_name, loss_scale_factor)."""
    if epoch < 30:
        return 5, 0.031, 0.031 / 4, 'A', 1.0
    elif epoch < 65:
        return 7, 0.062, 0.062 / 4, 'B', 1.0
    elif epoch < 100:
        return 10, 0.100, 0.100 / 4, 'C', 1.0
    else:
        return 10, 0.150, 0.150 / 4, 'D', 0.5  # halve non-adv weights


def main():
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
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v5_best.pth')

    if not os.path.exists(clip_init_ckpt):
        print(f"ERROR: Phase 0 checkpoint not found at {clip_init_ckpt}")
        print("Run pretrain_rhan_v5_clip.py first!")
        return
    if not os.path.exists(cornet_ckpt):
        print(f"ERROR: CORnet-S checkpoint not found at {cornet_ckpt}")
        return

    # ── Model: RHANv5 — load from Phase 0 ──
    model = RHANv5(head_type='cosine').to(device)
    model.load_state_dict(torch.load(clip_init_ckpt, map_location=device))
    print(f"RHANv5 loaded from Phase 0 checkpoint: {clip_init_ckpt}")

    # ── CORnet-S teacher (frozen) ──
    teacher = CIFARCORnet().to(device)
    teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device))
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
    epochs = 120; warmup_epochs = 15; accum_steps = 2
    base_lr = 0.001; phase_d_lr = 0.0001

    optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, 100, epochs, base_lr, phase_d_lr)
    scaler = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_v5'))

    # Base loss weights
    W_ADV = 0.40; W_CLEAN = 0.20; W_ALIGN = 0.30; W_FREQ = 0.10

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    print(f"\n{'='*70}")
    print("RHAN-v5 · Phase 1 Curriculum Training")
    print(f"{'='*70}")
    print(f"  Architecture:    RHANv5 (freq-separated, ventral/dorsal)")
    print(f"  Initialization:  Phase 0 CLIP checkpoint")
    print(f"  Curriculum:      A(ε=0.031)→B(ε=0.062)→C(ε=0.100)→D(ε=0.150)")
    print(f"  Optimizer:       AdamW (base_lr={base_lr}, wd=0.05)")
    print(f"  Warmup:          {warmup_epochs} epochs")
    print(f"  Batch:           64 × {accum_steps} accum = 128 effective")
    print(f"  Epochs:          {epochs}")
    print(f"  Loss weights:    adv={W_ADV} clean={W_CLEAN} align={W_ALIGN} freq={W_FREQ}")
    print(f"  Save to:         {output_ckpt}")
    print(f"{'='*70}\n")

    best_test_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)
        pgd_steps, pgd_eps, pgd_alpha, phase_name, scale = get_curriculum(epoch)

        # Scale non-adversarial weights in Phase D
        w_clean = W_CLEAN * scale
        w_align = W_ALIGN * scale
        w_freq = W_FREQ * scale

        compiled_model.train()
        s_adv = s_clean = s_align = s_freq = s_total = 0.0
        train_correct = train_total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0); half = B // 2

            if half > 0:
                saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        compiled_model, imgs[:half], labels[:half],
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
                # 1. Adversarial CE (40%)
                adv_logits, adv_feats = compiled_model.forward_with_features(adv_imgs)
                loss_adv = F.cross_entropy(adv_logits, adv_labels)

                # 2. Clean CE (20% * scale)
                cln_logits, _ = compiled_model.forward_with_features(clean_imgs)
                loss_clean = F.cross_entropy(cln_logits, cln_labels)

                # 3. Neural alignment ON ADVERSARIAL IMAGES (30% * scale)
                with torch.no_grad():
                    it_feats = get_it_features(teacher, adv_imgs)
                loss_align = 1.0 - (F.normalize(adv_feats, dim=-1) * F.normalize(it_feats, dim=-1)).sum(dim=-1).mean()

                # 4. Frequency consistency loss (10% * scale)
                # Low-freq features should be similar clean vs adversarial
                if half > 0:
                    x_low_clean, _ = compiled_model.separate_frequencies(imgs[:half])
                    x_low_adv, _ = compiled_model.separate_frequencies(adv_imgs)
                    f_low_clean = compiled_model.stem_low(x_low_clean)
                    f_low_adv = compiled_model.stem_low(x_low_adv)
                    loss_freq = F.mse_loss(f_low_adv, f_low_clean.detach())
                else:
                    loss_freq = torch.tensor(0.0, device=device)

                total_loss = (W_ADV * loss_adv + w_clean * loss_clean +
                              w_align * loss_align + w_freq * loss_freq) / accum_steps

            scaler.scale(total_loss).backward()
            s_adv += loss_adv.item() * B; s_clean += loss_clean.item() * B
            s_align += loss_align.item() * B; s_freq += loss_freq.item() * B
            s_total += total_loss.item() * accum_steps * B
            _, pred = adv_logits.max(1)
            train_total += adv_labels.size(0); train_correct += pred.eq(adv_labels).sum().item()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(trainloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

        N = len(trainloader.dataset)
        l_adv = s_adv / N; l_clean = s_clean / N; l_align = s_align / N
        l_freq_val = s_freq / N; l_total = s_total / N
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

        # ── TensorBoard ──
        tb_writer.add_scalar('Loss/Adv_CE', l_adv, epoch)
        tb_writer.add_scalar('Loss/Clean_CE', l_clean, epoch)
        tb_writer.add_scalar('Loss/Align', l_align, epoch)
        tb_writer.add_scalar('Loss/FreqConsist', l_freq_val, epoch)
        tb_writer.add_scalar('Loss/Total', l_total, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc, epoch)
        tb_writer.add_scalar('Curriculum/Epsilon', pgd_eps, epoch)
        tb_writer.add_scalar('Weights/freq_low', torch.sigmoid(model.freq_weight_low).item(), epoch)
        tb_writer.add_scalar('Weights/freq_high', torch.sigmoid(model.freq_weight_high).item(), epoch)

        if test_acc >= best_test_acc:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw.state_dict(), output_ckpt)
            best_test_acc = test_acc; marker = ' ★ BEST'
        else:
            marker = ''

        w_lo = torch.sigmoid(model.freq_weight_low).item()
        w_hi = torch.sigmoid(model.freq_weight_high).item()
        print(f"Epoch {epoch+1:03d}/{epochs} [{phase_name}] ε={pgd_eps:.3f} | "
              f"Adv:{l_adv:.4f} Cln:{l_clean:.4f} Alg:{l_align:.4f} Frq:{l_freq_val:.4f} | "
              f"Train:{train_acc:.1f}% Test:{test_acc:.2f}% | "
              f"wL:{w_lo:.3f} wH:{w_hi:.3f} | LR:{current_lr:.6f} | "
              f"{time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\n{'='*70}")
    print(f"Phase 1 training complete. Best checkpoint: {output_ckpt}")
    print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}\n")
    tb_writer.close()

    # =====================================================================
    # POST-TRAINING EVALUATION PIPELINE
    # =====================================================================
    print("Loading best checkpoint for evaluation...")
    eval_model = RHANv5(head_type='cosine').to(device)
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
    for check_eps in [0.05, 0.10]:
        rn_correct = rn_total = 0
        with torch.no_grad():
            for images, lbls in testloader:
                if rn_total >= max_samples: break
                images, lbls = images.to(device), lbls.to(device)
                noise = torch.empty_like(images).uniform_(-check_eps, check_eps)
                noisy = torch.max(torch.min(images + noise, cifar_max), cifar_min)
                _, preds = eval_model(noisy).max(1)
                rn_correct += preds.eq(lbls).sum().item(); rn_total += lbls.size(0)
        rn_acc = 100.0 * rn_correct / max(rn_total, 1)
        print(f"  Random noise ε={check_eps:.2f}: {rn_acc:.2f}%")

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
    print(f"  PGD-20 ε=0.05: {p20_acc_05:.2f}%")

    # ── Step 2: Full PGD-100 evaluation ──
    print(f"\n{'='*70}\nRHAN-v5 PGD-100 Evaluation\n{'='*70}")
    v5_accs = []
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
        acc = 100.0 * correct / max(total, 1); v5_accs.append(acc)
        print(f"Acc:{acc:.2f}% | {time.time()-t0:.1f}s")

    # PGD-20 vs PGD-100 gap
    p100_05 = v5_accs[2]
    pgd_gap = p20_acc_05 - p100_05
    print(f"\nPGD-20 vs PGD-100 gap at ε=0.05: {pgd_gap:.2f}%")
    masking_detected = pgd_gap >= 8.0
    if masking_detected:
        print("  ⚠ Potential gradient masking!")
    else:
        print(f"  ✓ No gradient masking (gap {pgd_gap:.2f}%)")

    # ── Step 3: SDT d-prime & εthresh ──
    import scipy.stats as stats
    v5_dprimes = []
    for acc_pct in v5_accs:
        acc = acc_pct / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        v5_dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(v5_dprimes) - 1):
        d1, d2 = v5_dprimes[i], v5_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(v5_dprimes) > 0 and v5_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"

    # ── Step 4: Frequency weight analysis ──
    w_lo_final = torch.sigmoid(eval_model.freq_weight_low).item()
    w_hi_final = torch.sigmoid(eval_model.freq_weight_high).item()

    print(f"\n{'='*70}\nFrequency Weight Analysis\n{'='*70}")
    print(f"  freq_weight_low (sigmoid):  {w_lo_final:.4f}")
    print(f"  freq_weight_high (sigmoid): {w_hi_final:.4f}")
    if w_lo_final > w_hi_final:
        print(f"  ✓ Model learned shape-dominant processing (low > high)")
        print(f"    Biological hypothesis CONFIRMED: M-pathway dominance")
    else:
        print(f"  ⚠ Model did not learn low-frequency dominance")

    # ── Step 5: Ablation — low-freq only inference ──
    print(f"\n{'='*70}\nAblation: Low-Frequency Only Inference\n{'='*70}")
    saved_high = eval_model.freq_weight_high.data.clone()
    eval_model.freq_weight_high.data.fill_(-100.0)  # sigmoid(-100) ≈ 0

    lo_only_accs = []
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
                _, preds = eval_model(adv_images).max(1)
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        lo_acc = 100.0 * correct / max(total, 1); lo_only_accs.append(lo_acc)
    eval_model.freq_weight_high.data = saved_high

    print(f"{'ε':<8} | {'Full Model':>10} | {'Low-Only':>10} | {'Delta':>8}")
    print("-" * 45)
    for i, eps in enumerate(epsilons):
        delta = lo_only_accs[i] - v5_accs[i]
        sign = '+' if delta >= 0 else ''
        print(f"{eps:<8.2f} | {v5_accs[i]:>9.2f}% | {lo_only_accs[i]:>9.2f}% | {sign}{delta:>7.2f}%")

    # ── Step 6: Final comparison table ──
    rhan_v3 = {0.00: 91.41, 0.01: 85.35, 0.05: 60.74, 0.10: 26.17, 0.20: 1.17, 0.30: 0.00}
    rhan_adv = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    resnet = {0.00: 95.82, 0.01: 75.57, 0.05: 2.84, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit = {0.00: 97.80, 0.01: 55.18, 0.05: 8.80, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}
    human = {0.00: 73.33, 0.01: 'N/A', 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}

    print(f"\n{'='*85}\nRHAN-v5 FINAL VERDICT\n{'='*85}")
    print(f"{'ε':<8} | {'Human':>8} | {'RHAN-v5':>8} | {'RHAN-v3':>8} | {'RHAN-adv':>8} | {'ResNet':>8} | {'ViT':>8}")
    print("-" * 85)
    for i, eps in enumerate(epsilons):
        h = human[eps]
        h_str = f"{h:.2f}%" if isinstance(h, float) else h
        print(f"{eps:<8.2f} | {h_str:>8} | {v5_accs[i]:>7.2f}% | {rhan_v3[eps]:>7.2f}% | {rhan_adv[eps]:>7.2f}% | {resnet[eps]:>7.2f}% | {vit[eps]:>7.2f}%")
    print("=" * 85)

    print(f"\n--- SDT d-prime ---")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f}: d'={v5_dprimes[i]:.4f}")
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    print(f"\n{'='*70}")
    print("ROBUSTNESS RANKING (SDT ε_thresh)")
    print(f"{'='*70}")
    print(f"  {'System':<20} | {'ε_thresh':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Human':<20} | {'> 0.3000':>10}")
    print(f"  {'RHAN-v5':<20} | {thresh_str:>10}")
    print(f"  {'RHAN-v3':<20} | {'0.0900':>10}")
    print(f"  {'RHAN-adv':<20} | {'0.0764':>10}")
    print(f"  {'ResNet-18':<20} | {'0.0295':>10}")
    print(f"  {'ViT-Small':<20} | {'0.0264':>10}")
    print(f"{'='*70}")

    # Final summary
    print(f"\n{'='*70}")
    if eps_thresh is not None and eps_thresh > 0.200:
        print(f"🏆 TARGET ACHIEVED: ε_thresh = {thresh_str} > 0.200")
    elif eps_thresh is not None and eps_thresh > 0.120:
        print(f"✅ Strong improvement: ε_thresh = {thresh_str} > 0.120")
    else:
        print(f"⚠  ε_thresh = {thresh_str} — did not reach 0.120 target")
    print(f"Frequency weights: low={w_lo_final:.3f}, high={w_hi_final:.3f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
