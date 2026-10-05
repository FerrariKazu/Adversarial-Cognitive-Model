#!/usr/bin/env python3
"""
RHAN-v3: Joint Training from Scratch
====================================
Architecture: RHANSplit (ventral/dorsal)
Initialization: Randomly initialized (from scratch)
Loss from epoch 1: 0.5*adv_CE + 0.2*clean_CE + 0.2*align_on_adv + 0.1*consistency
KEY FIX: alignment computed on adversarial images, not clean
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

from model_rhan_split import RHANSplit
from model_cornets import CIFARCORnet
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → base_lr over warmup_epochs
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing: base_lr → 0 over remaining epochs
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_it_features(teacher_model, x):
    """Extract 512-dim CORnet-S IT features. Resizes x to 224x224 internally."""
    x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    out = teacher_model.model.module.V1(x_224)
    out = teacher_model.model.module.V2(out)
    out = teacher_model.model.module.V4(out)
    out = teacher_model.model.module.IT(out)
    out = teacher_model.model.module.decoder.avgpool(out)
    out = teacher_model.model.module.decoder.flatten(out)
    return out


def main():
    set_seed(42)
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    script_dir  = os.path.dirname(__file__)
    ckpt_dir    = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    cornet_ckpt = os.path.join(script_dir, 'checkpoints', 'cornets_best.pth')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v3_best.pth')

    if not os.path.exists(cornet_ckpt):
        print(f"ERROR: Teacher checkpoint not found at {cornet_ckpt}"); return

    # Model — randomly initialized
    model = RHANSplit(head_type='cosine').to(device)
    print("Model initialized randomly (from scratch).")

    # CORnet-S teacher (frozen)
    teacher = CIFARCORnet().to(device)
    teacher.load_state_dict(torch.load(cornet_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # DataLoaders
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=64, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=64, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    testloader  = DataLoader(testloader_raw.dataset,  batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1,3,1,1).to(device)

    epochs = 100; warmup_epochs = 15; accum_steps = 2
    base_lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, base_lr=base_lr)
    scaler    = GradScaler('cuda')
    tb_writer = SummaryWriter(log_dir=os.path.join(script_dir, '..', 'runs', 'rhan_v3'))

    W_ADV_CE=0.50; W_CLEAN_CE=0.20; W_ALIGN=0.20; W_CONSIST=0.10

    print("Compiling model via torch.compile...")
    compiled_model = torch.compile(model)

    print(f"\n{'='*70}")
    print("RHAN-v3 · Joint Training from Scratch")
    print(f"{'='*70}")
    print(f"  Architecture:  RHANSplit (ventral/dorsal)")
    print(f"  Initialization:Random (Scratch)")
    print(f"  Teacher:       CORnet-S ({cornet_ckpt})")
    print(f"  Optimizer:     AdamW (base_lr={base_lr}, wd=0.05)")
    print(f"  Warmup:        {warmup_epochs} epochs")
    print(f"  Batch:         64 × {accum_steps} accum = 128 effective")
    print(f"  Epochs:        {epochs}")
    print(f"  Loss weights:  adv_CE={W_ADV_CE} clean_CE={W_CLEAN_CE} align={W_ALIGN} consist={W_CONSIST}")
    print(f"  Alignment:     ON ADVERSARIAL IMAGES")
    print(f"  Save to:       {output_ckpt}")
    print(f"{'='*70}\n")

    best_test_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)
        compiled_model.train()
        s_adv_ce=s_clean_ce=s_align=s_consist=s_total=0.0
        train_correct=train_total=0

        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(trainloader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.size(0); half = B // 2

            if half > 0:
                # Save gradients to prevent pgd_attack from wiping accumulated gradients
                saved_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                
                with torch.enable_grad():
                    adv_imgs, _ = pgd_attack(
                        compiled_model, imgs[:half], labels[:half],
                        epsilon=0.031, alpha=0.031/4, steps=5,
                        device=device, clip_min=cifar_min, clip_max=cifar_max,
                        random_start=True)
                adv_imgs   = adv_imgs.detach()
                
                # Restore gradients
                for p, g in zip(model.parameters(), saved_grads):
                    p.grad = g
                
                clean_imgs = imgs[half:]
                adv_labels = labels[:half]; cln_labels = labels[half:]
            else:
                adv_imgs=clean_imgs=imgs; adv_labels=cln_labels=labels

            with autocast('cuda'):
                # 1. Adversarial CE
                adv_logits    = compiled_model(adv_imgs)
                loss_adv_ce   = F.cross_entropy(adv_logits, adv_labels)
                # 2. Clean CE
                cln_logits    = compiled_model(clean_imgs)
                loss_clean_ce = F.cross_entropy(cln_logits, cln_labels)
                # 3. Adversarial alignment (align ON adv images)
                adv_feats = compiled_model.get_feature_vector(adv_imgs)
                with torch.no_grad():
                    it_feats  = get_it_features(teacher, adv_imgs)
                loss_align = 1.0 - (F.normalize(adv_feats, dim=-1) * F.normalize(it_feats, dim=-1)).sum(dim=-1).mean()
                # 4. Consistency: force adv features ≈ clean features
                if half > 0:
                    with torch.no_grad():
                        cln_feats_ref = compiled_model.get_feature_vector(imgs[:half])
                    adv_feats_cmp = compiled_model.get_feature_vector(adv_imgs)
                    loss_consist  = F.mse_loss(F.normalize(adv_feats_cmp, dim=-1),
                                               F.normalize(cln_feats_ref, dim=-1))
                else:
                    loss_consist  = torch.tensor(0.0, device=device)
                
                # Combined
                total_loss = (W_ADV_CE*loss_adv_ce + W_CLEAN_CE*loss_clean_ce +
                              W_ALIGN*loss_align + W_CONSIST*loss_consist) / accum_steps

            scaler.scale(total_loss).backward()
            s_adv_ce   += loss_adv_ce.item()*B;   s_clean_ce += loss_clean_ce.item()*B
            s_align    += loss_align.item()*B;     s_consist  += loss_consist.item()*B
            s_total    += total_loss.item()*accum_steps*B
            _, pred = adv_logits.max(1)
            train_total += adv_labels.size(0); train_correct += pred.eq(adv_labels).sum().item()

            if (step+1) % accum_steps == 0 or (step+1) == len(trainloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

        N = len(trainloader.dataset)
        l_adv_ce=s_adv_ce/N; l_clean_ce=s_clean_ce/N
        l_align=s_align/N;   l_consist=s_consist/N; l_total=s_total/N
        train_acc = 100.0 * train_correct / max(train_total, 1)

        compiled_model.eval()
        test_correct=test_total=0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs=inputs.to(device, non_blocking=True)
                targets=targets.to(device, non_blocking=True)
                with autocast('cuda'):
                    outputs = compiled_model(inputs)
                _, pred = outputs.max(1)
                test_total += targets.size(0); test_correct += pred.eq(targets).sum().item()
        test_acc = 100.0 * test_correct / test_total

        tb_writer.add_scalar('Loss/Adv_CE',   l_adv_ce,   epoch)
        tb_writer.add_scalar('Loss/Clean_CE', l_clean_ce, epoch)
        tb_writer.add_scalar('Loss/Align',    l_align,    epoch)
        tb_writer.add_scalar('Loss/Consist',  l_consist,  epoch)
        tb_writer.add_scalar('Loss/Total',    l_total,    epoch)
        tb_writer.add_scalar('Accuracy/Train',train_acc,  epoch)
        tb_writer.add_scalar('Accuracy/Test', test_acc,   epoch)

        if test_acc >= best_test_acc:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw.state_dict(), output_ckpt)
            best_test_acc = test_acc; marker = ' ★ BEST'
        else:
            marker = ''

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"AdvCE:{l_adv_ce:.4f} ClnCE:{l_clean_ce:.4f} "
              f"Align:{l_align:.4f} Consist:{l_consist:.4f} Total:{l_total:.4f} | "
              f"Train:{train_acc:.1f}% Test:{test_acc:.2f}% | "
              f"LR:{current_lr:.7f} | {time.time()-epoch_start:.1f}s{marker}", flush=True)

    print(f"\n{'='*70}")
    print(f"Training complete. Saved to: {output_ckpt}")
    print(f"Total time: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}\n")
    tb_writer.close()

    # ── Inline PGD-100 evaluation ──────────────────────────────────────────
    print("Loading best checkpoint for inline PGD-100 evaluation...")
    eval_model = RHANSplit(head_type='cosine').to(device)
    eval_model.load_state_dict(torch.load(output_ckpt, map_location=device))
    eval_model.eval()
    for p in eval_model.parameters(): p.requires_grad = False

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    wrapper = W(eval_model)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    rhan_adv_pgd = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    max_samples = 500; v2_accs = []

    print(f"\n{'='*70}\nRHAN-v3  Inline PGD-100 Evaluation\n{'='*70}")
    for eps in epsilons:
        t0=time.time(); print(f"Evaluating ε={eps:.2f}...", end=' ', flush=True)
        correct=total=0; alpha=max(eps/10, 0.001) if eps>0 else 0
        for images, lbls in testloader:
            if total >= max_samples: break
            images=images.to(device); lbls=lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv_images = images
            with torch.no_grad():
                _, preds = eval_model(adv_images).max(dim=1)
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        acc = 100.0*correct/max(total,1); v2_accs.append(acc)
        print(f"Acc:{acc:.2f}% | {time.time()-t0:.1f}s")

    print(f"\n{'='*70}\nRHAN-v3  FINAL VERDICT\n{'='*70}")
    print(f"{'ε':<8} | {'RHAN-adv':<12} | {'RHAN-v3':<12} | {'Delta':<10} | Verdict")
    print("-"*65)
    for i, eps in enumerate(epsilons):
        base=rhan_adv_pgd[eps]; v2=v2_accs[i]; delta=v2-base
        sign='+' if delta>=0 else ''
        verdict='✅ WIN' if delta>0 else ('🟰 TIE' if delta==0 else '❌ LOSS')
        print(f"{eps:<8.2f} | {base:<11.2f}% | {v2:<11.2f}% | {sign}{delta:<8.2f}% | {verdict}")
    print("="*65)

    v2_clean=v2_accs[0]; v2_005=v2_accs[2]
    b_clean=rhan_adv_pgd[0.00]; b_005=rhan_adv_pgd[0.05]
    print()
    if v2_005>b_005 and v2_clean>b_clean:
        print(f"🏆 HEADLINE RESULT CONFIRMED:")
        print(f"   RHAN-v3 improves BOTH clean ({v2_clean:.2f}% vs {b_clean:.2f}%)")
        print(f"   AND ε=0.05 robustness ({v2_005:.2f}% vs {b_005:.2f}%)")
        print(f"   — First model to achieve simultaneous improvement.")
    elif v2_005>b_005:
        print(f"✅ ε=0.05 improved: {v2_005:.2f}% vs {b_005:.2f}%  (clean: {v2_clean:.2f}% vs {b_clean:.2f}%)")
    elif v2_clean>b_clean:
        print(f"✅ Clean improved: {v2_clean:.2f}% vs {b_clean:.2f}%  (ε=0.05: {v2_005:.2f}% vs {b_005:.2f}%)")
    else:
        print(f"⚠  Neither headline metric improved.")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
