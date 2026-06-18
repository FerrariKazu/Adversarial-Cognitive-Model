#!/usr/bin/env python3
"""
RHAN-Unified STL-10 Training with Pretrained ResNet-50 Stem (No SAIL / Invariance Losses)
========================================================================================

Usage:
  # Phase 0: Warmup training (10 epochs, frozen stem)
  python phase1_training/train_rhan_stl10_pretrained.py --phase 0

  # Phases 1-8: TRADES curriculum (15 epochs each, unfrozen stem)
  python phase1_training/train_rhan_stl10_pretrained.py --phase 1 --resume

  # Evaluate a trained checkpoint under AutoAttack
  python phase1_training/train_rhan_stl10_pretrained.py --eval-only
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
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DEFINITION (RHAN-Unified STL-10)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from model_rhan_stl10_pretrained import PredictiveCodingLayerSTL, RHANUnifiedSTL10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_stl10_dataloaders(data_root='./data/stl10', batch_size=64):
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)

    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.STL10(
        data_root, split='train', transform=train_transform, download=True
    )
    testset = torchvision.datasets.STL10(
        data_root, split='test', transform=test_transform, download=True
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Normalization bounds in input space
    stl_min = torch.tensor([-(m/s) for m, s in zip(mean, std)]).view(1,3,1,1)
    stl_max = torch.tensor([(1-m)/s for m, s in zip(mean, std)]).view(1,3,1,1)

    return trainloader, testloader, stl_min, stl_max


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CURRICULUM DEFINITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CURRICULUM = {
    0: {'eps': 0.000, 'beta': 0.0,  'steps': 0,  'lr': 0.001, 'epochs': 15,  'desc': 'Clean pretraining (stem frozen)'},
    1: {'eps': 0.016, 'beta': 2.0,  'steps': 5,  'lr': 0.005, 'epochs': 15,  'desc': 'TRADES ε=0.016 (stem unfrozen)'},
    2: {'eps': 0.031, 'beta': 2.0,  'steps': 7,  'lr': 0.005, 'epochs': 15,  'desc': 'TRADES ε=0.031 (standard threat model)'},
    3: {'eps': 0.047, 'beta': 2.0,  'steps': 8,  'lr': 0.003, 'epochs': 15,  'desc': 'TRADES ε=0.047'},
    4: {'eps': 0.062, 'beta': 2.0,  'steps': 10, 'lr': 0.002, 'epochs': 15,  'desc': 'TRADES ε=0.062'},
    5: {'eps': 0.078, 'beta': 2.0,  'steps': 10, 'lr': 0.001, 'epochs': 15,  'desc': 'TRADES ε=0.078'},
    6: {'eps': 0.094, 'beta': 2.0,  'steps': 10, 'lr': 0.001, 'epochs': 15,  'desc': 'TRADES ε=0.094'},
    7: {'eps': 0.110, 'beta': 2.0,  'steps': 12, 'lr': 0.0005,'epochs': 15,  'desc': 'TRADES ε=0.110'},
    8: {'eps': 0.130, 'beta': 2.0,  'steps': 12, 'lr': 0.0003,'epochs': 15,  'desc': 'TRADES ε=0.130'},
}

BETA_WARMUP_EPOCHS = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOATTACK EVALUATION MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_autoattack(model, testloader, device, eps=0.031, n_samples=1000, bs=128):
    """Evaluate under AutoAttack and calculate per-class accuracies."""
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("\n[WARNING] autoattack not installed. Skipping AutoAttack evaluation.")
        print("Install with: pip install autoattack")
        return

    print(f"\n{'='*70}")
    print(f"AUTOATTACK EVALUATION (standard Linf, ε={eps:.4f}, n={n_samples})")
    print(f"{'='*70}")

    model.eval()

    # Collect sample subset
    all_imgs, all_lbls = [], []
    for imgs, lbls in testloader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
        if sum(x.size(0) for x in all_imgs) >= n_samples:
            break

    x_test = torch.cat(all_imgs, dim=0)[:n_samples].to(device)
    y_test = torch.cat(all_lbls, dim=0)[:n_samples].to(device)

    # Wrap model to return only logits
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

    wrapper = Wrapper(model).eval()

    # Clean check
    with torch.no_grad():
        clean_preds = wrapper(x_test).argmax(dim=1)
    clean_acc = 100.0 * clean_preds.eq(y_test).sum().item() / y_test.size(0)
    print(f"Clean accuracy (subset): {clean_acc:.2f}%")

    # Initialize AutoAttack
    # Epsilon is passed as-is (0.031)
    adversary = AutoAttack(wrapper, norm='Linf', eps=eps, version='standard', device=device, verbose=True)
    t0 = time.time()
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=bs)
    elapsed = time.time() - t0

    with torch.no_grad():
        adv_preds = wrapper(x_adv).argmax(dim=1)
    aa_correct = adv_preds.eq(y_test).sum().item()
    aa_acc = 100.0 * aa_correct / y_test.size(0)

    print(f"\nAutoAttack Robust Accuracy: {aa_acc:.2f}% ({aa_correct}/{y_test.size(0)}) in {elapsed:.0f}s")

    # Per-class AutoAttack breakdown
    print("\nPer-class Robust Accuracy:")
    print("-" * 35)
    for c in range(10):
        mask = y_test == c
        n_c = mask.sum().item()
        if n_c > 0:
            c_cl = 100.0 * clean_preds[mask].eq(c).sum().item() / n_c
            c_aa = 100.0 * adv_preds[mask].eq(c).sum().item() / n_c
            tag = " <<< KEY" if c in [2, 9] else ""  # 'car' (2) or 'truck' (9)
            print(f"  {RHANUnifiedSTL10.STL10_CLASSES[c]:<12}: Clean={c_cl:>5.1f}% | AA={c_aa:>5.1f}%{tag}")

    car_aa = 100.0 * adv_preds[y_test == 2].eq(2).sum().item() / max((y_test == 2).sum().item(), 1)
    truck_aa = 100.0 * adv_preds[y_test == 9].eq(9).sum().item() / max((y_test == 9).sum().item(), 1)
    print(f"\nKey Result (Car vs Truck): Car AA = {car_aa:.1f}%, Truck AA = {truck_aa:.1f}%")
    if car_aa > 0 and truck_aa > 0:
        print(">>> Resolution hypothesis confirmed! TDV pretraining is the next step.")
    else:
        print(">>> Collapse is deeper than resolution! TDV temporal pretraining is essential.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase(phase_id, model, trainloader, testloader, device, stl_min, stl_max, ckpt_dir, resume=False):
    cfg = CURRICULUM[phase_id]
    print(f"\n{'='*70}")
    print(f"PHASE {phase_id}: {cfg['desc']}")
    print(f"  ε={cfg['eps']}, β={cfg['beta']}, steps={cfg['steps']}, "
          f"lr={cfg['lr']}, epochs={cfg['epochs']}")
    print(f"{'='*70}")

    if phase_id == 0:
        model.freeze_stem(freeze=True)
    else:
        model.freeze_stem(freeze=False)

    if phase_id == 0:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'], weight_decay=1e-4
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'], momentum=0.9, weight_decay=1e-4
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01
    )
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    rolling_path = os.path.join(ckpt_dir, f'rhan_stl_pretrained_phase{phase_id}_rolling.pth')
    final_path   = os.path.join(ckpt_dir, f'rhan_stl_pretrained_phase{phase_id}_final.pth')
    best_path    = os.path.join(ckpt_dir, 'rhan_stl_pretrained_best.pth')

    start_epoch = 1
    best_acc = 0.0

    if resume and os.path.exists(rolling_path):
        ckpt = torch.load(rolling_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"Resumed from epoch {ckpt['epoch']}, best_acc={best_acc:.2f}%")

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        t0 = time.time()
        model.train()

        if epoch <= BETA_WARMUP_EPOCHS:
            effective_beta = cfg['beta'] * (0.3 + 0.7 * epoch / BETA_WARMUP_EPOCHS)
        else:
            effective_beta = cfg['beta']

        total_loss = n_total = correct = 0

        for imgs, lbls in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            if cfg['eps'] > 0:
                # TRADES Adversarial PGD generation
                model.eval()
                x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
                x_adv = torch.clamp(x_adv, stl_min, stl_max)

                for _ in range(cfg['steps']):
                    x_adv.requires_grad_(True)
                    with torch.enable_grad():
                        with autocast('cuda'):
                            logits_a = model(x_adv)
                            with torch.no_grad():
                                logits_c = model(imgs)
                            probs_c = F.softmax(logits_c.float(), dim=1)
                            loss_adv = F.kl_div(
                                F.log_softmax(logits_a.float(), dim=1),
                                probs_c, reduction='batchmean'
                            )
                    grad = torch.autograd.grad(loss_adv, x_adv)[0]
                    x_adv = x_adv.detach() + (cfg['eps'] / cfg['steps']) * grad.sign()
                    delta = torch.clamp(x_adv - imgs, -cfg['eps'], cfg['eps'])
                    x_adv = torch.clamp(imgs + delta, stl_min, stl_max).detach()
                model.train()

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                if cfg['eps'] == 0:
                    logits = model(imgs)
                    loss = ce_loss(logits, lbls)
                    train_logits = logits
                else:
                    logits_c = model(imgs)
                    logits_a = model(x_adv)
                    loss = ce_loss(logits_c, lbls) + effective_beta * F.kl_div(
                        F.log_softmax(logits_a.float(), dim=1),
                        F.softmax(logits_c.float().detach(), dim=1),
                        reduction='batchmean'
                    )
                    train_logits = logits_c

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            total_loss += loss.item() * B
            n_total += B
            correct += train_logits.argmax(1).eq(lbls).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits = model(imgs)
                val_correct += logits.argmax(1).eq(lbls).sum().item()
                val_total += lbls.size(0)
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / n_total

        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            marker = ' ★'

        torch.save({
            'epoch': epoch, 'phase': phase_id,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
        }, rolling_path)

        print(
            f"P{phase_id} Epoch {epoch:02d}/{cfg['epochs']} | "
            f"Loss:{total_loss/n_total:.3f} | "
            f"TrAcc:{train_acc:.1f}% TeAcc:{val_acc:.1f}% | "
            f"β_eff:{effective_beta:.2f} | {time.time()-t0:.0f}s{marker}"
        )

    torch.save(model.state_dict(), final_path)
    print(f"\nPhase {phase_id} complete. Best acc: {best_acc:.2f}%")
    return final_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=0, choices=list(CURRICULUM.keys()))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-ckpt', type=str, default='')
    parser.add_argument('--eval-samples', type=int, default=256)
    parser.add_argument('--eval-eps', type=float, default=0.031)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    trainloader, testloader, stl_min, stl_max = get_stl10_dataloaders(args.data_root, args.batch_size)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    model = RHANUnifiedSTL10().to(device)

    if args.eval_only:
        ckpt_name = args.eval_ckpt if args.eval_ckpt else 'rhan_stl_pretrained_best.pth'
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded evaluation checkpoint: {ckpt_path}")
        else:
            print(f"Error: Evaluation checkpoint {ckpt_path} not found.")
            sys.exit(1)
        evaluate_autoattack(model, testloader, device, bs=args.batch_size, n_samples=args.eval_samples, eps=args.eval_eps)
        return

    # Loading previous phase checkpoint if starting a new phase
    if args.phase > 0 and not args.resume:
        prev_ckpt = os.path.join(ckpt_dir, f'rhan_stl_pretrained_phase{args.phase-1}_final.pth')
        if os.path.exists(prev_ckpt):
            model.load_state_dict(torch.load(prev_ckpt, map_location=device))
            print(f"Loaded phase {args.phase-1} checkpoint: {prev_ckpt}")
        else:
            best_ckpt = os.path.join(ckpt_dir, 'rhan_stl_pretrained_best.pth')
            if os.path.exists(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
                print(f"Loaded best checkpoint: {best_ckpt}")
            else:
                print("WARNING: No previous checkpoint found. Training from scratch.")

    run_phase(
        phase_id=args.phase,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        stl_min=stl_min,
        stl_max=stl_max,
        ckpt_dir=ckpt_dir,
        resume=args.resume,
    )

    print("\nNext command:")
    if args.phase < 8:
        print(f"  python phase1_training/train_rhan_stl10_pretrained.py --phase {args.phase + 1}")
    else:
        print("  Curriculum complete. Run evaluation:")
        print("  python phase1_training/train_rhan_stl10_pretrained.py --eval-only")


if __name__ == '__main__':
    main()
