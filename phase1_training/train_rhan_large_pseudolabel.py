#!/usr/bin/env python3
"""
Experiment: Large model + Pseudo-label training on STL-10
=========================================================

1. Generates pseudo-labels using the BEST available model (rhan_stl10_pseudolabel_best.pth) at confidence threshold 0.65.
2. Combines 5K real labeled STL-10 images + the generated pseudo-labels.
3. Initializes the Large model (RHANLargeSTL10) from checkpoints/rhan_stl10_large_video_tdv.pth.
4. Trains under a 120-epoch curriculum with standard TRADES loss (no TDV loss).
5. Employs gradient checkpointing and FP16 mixed precision to fit in memory.
6. Saves the best model to checkpoints/rhan_stl10_large_pseudolabel_best.pth.
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
from model_rhan_stl10_large import RHANLargeSTL10
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
# PSEUDO-LABEL GENERATION
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
    
    print(f"\nTotal pseudo-labeled: {len(pseudo_indices)} / 100000 ({100*len(pseudo_indices)/100000:.1f}%)")
    
    return pseudo_indices, pseudo_labels, confidence_scores

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEIGHTED TRADES LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def trades_loss_weighted(model, imgs, lbls, weights, 
                          x_adv, beta):
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
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training combined loader (default: 16 for T4)')
    parser.add_argument('--unlabeled-batch-size', type=int, default=256, help='Batch size for pseudo-label generation (default: 256 for T4)')
    parser.add_argument('--accum-steps', type=int, default=16, help='Gradient accumulation steps (default: 16 for effective batch size 256)')
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--labeling-ckpt', type=str, default='')
    parser.add_argument('--target-ckpt', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Generate new pseudo-labels using the BEST labeling model
    labeling_model = RHANUnifiedSTL10().to(device)
    best_labeling_ckpt = args.labeling_ckpt if args.labeling_ckpt else os.path.join(ckpt_dir, 'rhan_stl10_pseudolabel_best.pth')
    if os.path.exists(best_labeling_ckpt):
        labeling_model.load_state_dict(torch.load(best_labeling_ckpt, map_location=device))
        print(f"Loaded labeling model checkpoint: {best_labeling_ckpt}")
    else:
        print(f"Error: Labeling checkpoint {best_labeling_ckpt} not found! Required for pseudo-labeling.")
        sys.exit(1)

    # Detect CPU cores to optimize dataloader workers for Google Colab (typically has 2 vCPUs)
    num_cpus = os.cpu_count() or 2
    num_workers = min(2, num_cpus) if os.path.exists('/content') else min(4, num_cpus)
    p_workers = num_workers > 0
    print(f"Optimal dataloader workers detected: {num_workers} (persistent_workers={p_workers})")

    unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=False, 
                                  num_workers=num_workers, pin_memory=True, persistent_workers=p_workers)
    pseudo_indices, pseudo_lbls, _ = generate_pseudo_labels(labeling_model, unlabeled_loader, device, args.confidence_threshold)

    # Free labeling model VRAM cache
    del labeling_model
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    if len(pseudo_indices) == 0:
        print("Error: No pseudo-labels generated above confidence threshold. Exiting.")
        sys.exit(1)

    if torch.cuda.is_available():
        print(f"VRAM after pseudo-label generation: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 2. Load raw real labeled training data
    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    ])
    trainset_raw = torchvision.datasets.STL10(args.data_root, split='train', download=True)
    real_imgs = torch.stack([norm_transform(trainset_raw[i][0]) for i in range(len(trainset_raw))])
    real_labels = torch.tensor([trainset_raw[i][1] for i in range(len(trainset_raw))])

    # 3. Combined dataset & balanced loader
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
    ])
    combined_dataset = CombinedSTL10Dataset(real_imgs, real_labels, unlabeled_dataset, pseudo_indices, pseudo_lbls, transform=train_transform)
    
    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(real_imgs) + len(pseudo_indices)))
    sampler = BalancedBatchSampler(real_indices, pseudo_indices_list, batch_size=args.batch_size)
    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=num_workers, 
                              pin_memory=True, persistent_workers=p_workers)

    _, testloader, stl_min, stl_max = get_stl10_dataloaders(args.data_root, batch_size=64)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    # 4. Instantiate target model (Large)
    model = RHANLargeSTL10().to(device)

    # 5. Load Target (TDV pretrained) checkpoint
    best_target_ckpt = args.target_ckpt if args.target_ckpt else os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv.pth')
    if os.path.exists(best_target_ckpt):
        ckpt = torch.load(best_target_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded target model checkpoint: {best_target_ckpt} (from wrapped state dict)")
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            print(f"Loaded target model checkpoint: {best_target_ckpt} (from state_dict)")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded target model checkpoint: {best_target_ckpt} (from raw state dict)")
    else:
        print(f"Warning: Pretrained target checkpoint {best_target_ckpt} not found! Initializing model randomly.")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, 'module') else model

    # Curriculum Setup (120 epochs total)
    # format: (start_epoch, end_epoch, eps, beta, steps, lr)
    curriculum = [
        (1,  40,  0.031, 2.0, 7,  0.003),
        (41, 80,  0.062, 2.0, 10, 0.002),
        (81, 120, 0.094, 2.5, 10, 0.001),
    ]

    scaler = GradScaler('cuda')
    best_acc = 0.0
    start_epoch = 1
    current_phase_start = None
    optimizer = None
    scheduler = None

    best_path = os.path.join(ckpt_dir, 'rhan_stl10_large_pseudolabel_best.pth')
    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_large_pseudolabel_rolling.pth')

    # 6. Automatic Resume Check
    if os.path.exists(rolling_path):
        print(f"\nFound rolling checkpoint at {rolling_path}. Attempting to resume...")
        checkpoint_data = torch.load(rolling_path, map_location=device)
        raw_model.load_state_dict(checkpoint_data['model'])
        best_acc = checkpoint_data.get('best_acc', 0.0)
        start_epoch = checkpoint_data['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch} (Best validation accuracy so far: {best_acc:.2f}%)")

    # 7. Training loop
    for epoch in range(start_epoch, 121):
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
                    # Restore state dicts if we are resuming right at the start of this session
                    if epoch == start_epoch and os.path.exists(rolling_path) and 'optimizer' in checkpoint_data:
                        optimizer.load_state_dict(checkpoint_data['optimizer'])
                        scheduler.load_state_dict(checkpoint_data['scheduler'])
                        print("Restored optimizer and scheduler state dicts.")
                    print(f"\n--- Epoch {epoch}: New optimizer phase {p_start}-{p_end} (lr={phase_lr}) ---")
                break
        
        eps, beta, steps = phase_params
        
        # Training loop
        model.train()
        total_loss = n_total = correct = 0
        num_batches = min(len(trainloader), 600)
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (imgs, lbls, weights) in enumerate(trainloader):
            if batch_idx >= 600:
                break
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            
            # TRADES Adversarial PGD generation
            model.eval()
            x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
            x_adv = torch.clamp(x_adv, stl_min, stl_max)
            for _ in range(steps):
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
                x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                delta = torch.clamp(x_adv - imgs, -eps, eps)
                x_adv = torch.clamp(imgs + delta, stl_min, stl_max).detach()
            model.train()

            with autocast('cuda'):
                l_trades = trades_loss_weighted(model, imgs, lbls, weights, x_adv, beta)
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
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches} | Loss: {l_trades.item():.4f}")

        if num_batches % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Validation (clean test)
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
            f"Epoch {epoch:03d}/120 (ε={eps:.3f}) | Loss:{total_loss/n_total:.3f} | "
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

    # 8. Launch full evaluation (AutoAttack + PGD sweep) automatically on the trained model
    print("\nLaunching final evaluation (AutoAttack + PGD sweep)...")
    eval_script = os.path.join(script_dir, '..', 'run_eval_stl10.py')
    try:
        subprocess.run([
            sys.executable, eval_script,
            "--model-size", "large",
            "--checkpoint", "checkpoints/rhan_stl10_large_pseudolabel_best.pth",
            "--samples", "1000",
            "--batch-size", "128"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed: {e}")

if __name__ == '__main__':
    main()
