#!/usr/bin/env python3
"""
Experiment 2 — STL-10 Iterative Pseudo-label Round 2
===================================================

Starts from checkpoints/rhan_stl10_pseudolabel_best.pth (86.2% clean).
Generates pseudo-labels on 100K unlabeled STL-10 images at confidence=0.65.
Combines 5K real labeled + pseudo-labeled images in a balanced, weighted loader.
Fine-tunes under curriculum with TRADES (pure TRADES, no TDV consistency).
Uses batch size 64 with gradient accumulation 4 (effective 256).
PGD steps: 7 (was 10, reduces training time by 30%).
"""

import os
import sys
import time
import random
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_pretrained import RHANUnifiedSTL10
from train_rhan_stl10_tdv import get_stl10_dataloaders

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & BALANCED LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class STL10RawUnlabeledDataset(Dataset):
    def __init__(self, data_root='./data/stl10'):
        self.stl10 = torchvision.datasets.STL10(
            data_root, split='unlabeled', download=True
        )
        self.mean = (0.4467, 0.4398, 0.4066)
        self.std  = (0.2603, 0.2566, 0.2713)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return len(self.stl10)

    def __getitem__(self, idx):
        img, _ = self.stl10[idx]
        return self.transform(img), idx

class CombinedSTL10Dataset(Dataset):
    """
    Combines 5K real labeled + pseudo-labeled unlabeled images.
    Real labels get weight 1.0, pseudo-labels get weight 0.5.
    """
    def __init__(self, real_imgs, real_labels, 
                 unlabeled_dataset, pseudo_indices, pseudo_labels, transform=None):
        self.real_imgs = real_imgs
        self.real_labels = real_labels
        self.unlabeled_dataset = unlabeled_dataset
        self.pseudo_indices = pseudo_indices
        self.pseudo_labels = pseudo_labels
        self.transform = transform
        
        self.n_real = len(real_imgs)
        self.n_pseudo = len(pseudo_indices)
        print(f"Combined dataset: {self.n_real} real + "
              f"{self.n_pseudo} pseudo = {self.n_real+self.n_pseudo} total")
    
    def __len__(self):
        return self.n_real + self.n_pseudo
    
    def __getitem__(self, idx):
        if idx < self.n_real:
            img = self.real_imgs[idx]
            label = self.real_labels[idx]
            weight = 1.0
        else:
            pseudo_idx = self.pseudo_indices[idx - self.n_real].item()
            img, _ = self.unlabeled_dataset[pseudo_idx]
            label = self.pseudo_labels[idx - self.n_real]
            weight = 0.5
        
        if self.transform:
            img = self.transform(img)
        return img, label, torch.tensor(weight, dtype=torch.float32)

class BalancedBatchSampler(Sampler):
    """
    Ensures each batch contains an exact balance of real and pseudo-labeled samples.
    """
    def __init__(self, real_indices, pseudo_indices, batch_size=64):
        self.real_indices = real_indices
        self.pseudo_indices = pseudo_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.num_batches = max(len(real_indices), len(pseudo_indices)) // self.half_batch
        
    def __iter__(self):
        for _ in range(self.num_batches):
            real_batch = np.random.choice(self.real_indices, self.half_batch, replace=True)
            pseudo_batch = np.random.choice(self.pseudo_indices, self.half_batch, replace=True)
            batch = np.concatenate([real_batch, pseudo_batch])
            np.random.shuffle(batch)
            yield batch.tolist()
            
    def __len__(self):
        return self.num_batches

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — PSEUDO-LABEL GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_pseudo_labels(model, unlabeled_loader, device, confidence_threshold=0.65):
    """
    Generate pseudo-labels from 100K unlabeled STL-10 images.
    Uses the loaded checkpoint's predictions.
    Saves only the indices and predictions to minimize RAM footprint.
    """
    model.eval()
    pseudo_indices = []
    pseudo_labels = []
    confidence_scores = []
    
    print("Generating pseudo-labels from 100K unlabeled images...")
    with torch.no_grad():
        for batch_idx, (imgs, idx) in enumerate(unlabeled_loader):
            imgs = imgs.to(device)
            
            with autocast('cuda'):
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1)
            conf, pred = probs.max(1)
            
            mask = conf >= confidence_threshold
            if mask.sum() > 0:
                pseudo_indices.append(idx[mask.cpu()])
                pseudo_labels.append(pred[mask].cpu())
                confidence_scores.append(conf[mask].cpu())
            
            if batch_idx % 50 == 0:
                kept = sum(len(x) for x in pseudo_labels)
                print(f"  Batch {batch_idx}/195 | Kept: {kept} images")
    
    if len(pseudo_indices) == 0:
        print("WARNING: No high confidence pseudo labels generated!")
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0)

    pseudo_indices = torch.cat(pseudo_indices, dim=0)
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    confidence_scores = torch.cat(confidence_scores, dim=0)
    
    # Per-class distribution and mean confidence of pseudo-labels
    print("\nPseudo-label class distribution and mean confidence:")
    classes = ['airplane','bird','car','cat','deer',
               'dog','horse','monkey','ship','truck']
    for c in range(10):
        c_mask = (pseudo_labels == c)
        n = c_mask.sum().item()
        if n > 0:
            mean_conf = confidence_scores[c_mask].mean().item()
            print(f"  {classes[c]:<12}: {n:<6} images (mean confidence: {mean_conf:.4f})")
        else:
            print(f"  {classes[c]:<12}: 0      images")
    
    print(f"\nTotal pseudo-labeled: {len(pseudo_indices)} "
          f"/ 100000 ({100*len(pseudo_indices)/100000:.1f}%)")
    
    return pseudo_indices, pseudo_labels, confidence_scores

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — WEIGHTED TRADES LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def trades_loss_weighted(model, imgs, lbls, weights, 
                          x_adv, beta, stl_min, stl_max):
    """
    TRADES loss with per-sample weights.
    """
    ce = nn.CrossEntropyLoss(reduction='none')
    
    logits_c = model(imgs)
    logits_a = model(x_adv)
    
    l_ce = ce(logits_c, lbls)
    
    l_kl = F.kl_div(
        F.log_softmax(logits_a.float(), dim=1),
        F.softmax(logits_c.float().detach(), dim=1),
        reduction='none'
    ).sum(dim=1)
    
    l_total = (l_ce + beta * l_kl) * weights.to(l_ce.device)
    return l_total.mean()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training combined loader')
    parser.add_argument('--unlabeled-batch-size', type=int, default=512, help='Batch size for pseudo-label generation')
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Instantiate model
    model = RHANUnifiedSTL10().to(device)

    # 2. Load Round 1 checkpoint
    round1_ckpt = args.ckpt if args.ckpt else os.path.join(ckpt_dir, 'rhan_stl10_pseudolabel_best.pth')
    if os.path.exists(round1_ckpt):
        model.load_state_dict(torch.load(round1_ckpt, map_location=device))
        print(f"Loaded checkpoint for pseudo-labeling: {round1_ckpt}")
    else:
        print(f"Error: Round 1 checkpoint {round1_ckpt} not found! Required for pseudo-labeling.")
        sys.exit(1)

    # 3. Generate new pseudo-labels at confidence=0.65
    unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=False, num_workers=4)
    pseudo_indices, pseudo_lbls, _ = generate_pseudo_labels(model, unlabeled_loader, device, args.confidence_threshold)

    if len(pseudo_indices) == 0:
        print("Error: No pseudo-labels generated above confidence threshold. Exiting.")
        sys.exit(1)

    # Free VRAM cache
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        print(f"VRAM after pseudo-label generation: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 4. Load raw real labeled training data
    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    ])
    trainset_raw = torchvision.datasets.STL10(args.data_root, split='train', download=True)
    real_imgs = torch.stack([norm_transform(trainset_raw[i][0]) for i in range(len(trainset_raw))])
    real_labels = torch.tensor([trainset_raw[i][1] for i in range(len(trainset_raw))])

    # 5. Combined dataset & balanced loader
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
    ])
    combined_dataset = CombinedSTL10Dataset(real_imgs, real_labels, unlabeled_dataset, pseudo_indices, pseudo_lbls, transform=train_transform)
    
    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(real_imgs) + len(pseudo_indices)))
    sampler = BalancedBatchSampler(real_indices, pseudo_indices_list, batch_size=args.batch_size)
    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    _, testloader, stl_min, stl_max = get_stl10_dataloaders(args.data_root, batch_size=64)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, 'module') else model

    # Curriculum Setup with 7 PGD steps (and 8 for final stage)
    curriculum = [
        (1,  20, 0.031, 3.0, 7, 0.003),
        (21, 40, 0.062, 3.0, 7, 0.002),
        (41, 55, 0.094, 3.5, 7, 0.001),
        (56, 65, 0.125, 3.5, 8, 0.0005),
    ]

    scaler = GradScaler('cuda')
    best_acc = 0.0
    current_phase_start = None
    optimizer = None
    scheduler = None

    best_path = os.path.join(ckpt_dir, 'rhan_stl10_pseudolabel_v2_best.pth')
    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_pseudolabel_v2_rolling.pth')

    for epoch in range(1, 66):
        t0 = time.time()
        
        # Determine current phase parameters
        for p_start, p_end, eps, beta, steps, lr in curriculum:
            if p_start <= epoch <= p_end:
                phase_params = (eps, beta, steps)
                phase_lr = lr
                if current_phase_start != p_start:
                    current_phase_start = p_start
                    optimizer = optim.SGD(
                        model.parameters(), lr=phase_lr,
                        momentum=0.9, weight_decay=1e-4
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=p_end - p_start + 1, eta_min=phase_lr * 0.1
                    )
                    print(f"\n--- Epoch {epoch}: New optimizer phase {p_start}-{p_end} (lr={phase_lr}) ---")
                break
        
        eps, beta, steps = phase_params
        
        # Training loop
        model.train()
        total_loss = n_total = correct = 0
        num_batches = min(len(trainloader), 150)
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (imgs, lbls, weights) in enumerate(trainloader):
            if batch_idx >= 150:
                break
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            
            # TRADES Adversarial PGD generation
            model.eval()
            with torch.no_grad():
                with autocast('cuda'):
                    logits_c = model(imgs)
            probs_c = F.softmax(logits_c.float(), dim=1)

            x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
            x_adv = torch.clamp(x_adv, stl_min, stl_max)
            for _ in range(steps):
                x_adv.requires_grad_(True)
                with torch.enable_grad():
                    with autocast('cuda'):
                        logits_a = model(x_adv)
                        loss_adv = F.kl_div(
                            F.log_softmax(logits_a.float(), dim=1),
                            probs_c, reduction='batchmean'
                        )
                grad = torch.autograd.grad(loss_adv, x_adv)[0]
                x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                delta = torch.clamp(x_adv - imgs, -eps, eps)
                x_adv = torch.clamp(imgs + delta, stl_min, stl_max).detach()
            model.train()

            with autocast('cuda'):
                l_trades = trades_loss_weighted(model, imgs, lbls, weights, x_adv, beta, stl_min, stl_max)
                loss = l_trades / args.accum_steps

            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            B = imgs.size(0)
            total_loss += l_trades.item() * B
            
            with torch.no_grad():
                with autocast('cuda'):
                    logits_c = model(imgs)
            correct += logits_c.argmax(1).eq(lbls).sum().item()
            n_total += B
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{num_batches} | Loss: {l_trades.item():.4f}")

        if num_batches % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Validation (clean)
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
        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(raw_model.state_dict(), best_path)
            marker = ' ★'

        print(
            f"Epoch {epoch:02d}/65 (ε={eps:.3f}) | Loss:{total_loss/n_total:.3f} | "
            f"TrAcc:{100.*correct/n_total:.1f}% TeAcc:{val_acc:.1f}% | "
            f"{time.time()-t0:.0f}s{marker}"
        )

        torch.save({
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
        }, rolling_path)

    print(f"Training Complete. Best Model saved to {best_path}")

    # Launch full evaluation (AutoAttack + PGD sweep) automatically on the trained model
    print("\nLaunching final evaluation (AutoAttack + PGD sweep)...")
    eval_script = os.path.join(script_dir, 'eval_pseudolabel_full.py')
    try:
        subprocess.run([
            sys.executable, eval_script,
            "--checkpoint", "checkpoints/rhan_stl10_pseudolabel_v2_best.pth"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed: {e}")

if __name__ == '__main__':
    main()
