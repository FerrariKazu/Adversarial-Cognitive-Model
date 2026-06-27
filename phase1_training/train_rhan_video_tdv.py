#!/usr/bin/env python3
"""
Experiment 2 — Real Video TDV on UCF-101 (for Colab A100)
=========================================================

Training loop for Video Temporal Difference Vision (TDV) pretraining and fine-tuning.
Extracts genuine consecutive frame pairs with real motion from UCF-101.
Supports base (RHANUnifiedSTL10) and large (RHANLargeSTL10) architectures.

Data download instructions:
  # UCF-101 download (13GB):
  # wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
  # unrar x UCF101.rar
  # 
  # Or on Colab:
  # !pip install gdown
  # !gdown --id 1rG-U339NkM-oXFoqL0EbCGPqFSAZPBjZ
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
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_pretrained import RHANUnifiedSTL10
from model_rhan_stl10_large import RHANLargeSTL10
from train_rhan_stl10_tdv import get_stl10_dataloaders

def get_raw_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

# UCF-101 Categories relevant to STL-10 visual concepts
UCF_RELEVANT_CATEGORIES = {
    # Vehicles (maps to car/truck/ship/airplane)
    'Biking': 'vehicle_motion',
    'Driving': 'vehicle_motion',
    'MotorBoating': 'ship',
    'Rowing': 'ship',
    
    # Animals (maps to horse/deer/dog/cat/bird)
    'HorseRiding': 'horse',
    'WalkingWithDog': 'dog',
    'TaiChi': 'human_motion',
    
    # Flying (maps to airplane/bird)
    'Diving': 'aerial',
    'Surfing': 'water_motion',
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UCF-101 TEMPORAL DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UCF101TemporalDataset(Dataset):
    """
    Extracts consecutive frame pairs from UCF-101 video clips dynamically.
    Avoids OOM and slow startup by loading and decoding frame pairs on the fly.
    """
    def __init__(self, ucf_root, categories=None, frame_skip=3, img_size=96, epoch_len=5000):
        self.video_paths = []
        self.frame_skip = frame_skip
        self.img_size = img_size
        self.epoch_len = epoch_len
        self.mock = False
        
        if not os.path.exists(ucf_root):
            print(f"WARNING: UCF-101 directory '{ucf_root}' not found. Generating mock dataset.")
            self.mock = True
            self.mock_pairs = []
            for _ in range(100):
                f1 = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                f2 = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                self.mock_pairs.append((f1, f2))
        else:
            for category in os.listdir(ucf_root):
                if categories and category not in categories:
                    continue
                cat_path = os.path.join(ucf_root, category)
                if not os.path.isdir(cat_path):
                    continue
                for video_file in os.listdir(cat_path):
                    if video_file.endswith('.avi'):
                        self.video_paths.append(os.path.join(cat_path, video_file))
            print(f"UCF-101 dataset located: found {len(self.video_paths)} videos across {len(categories) if categories else 'all'} categories.")
            if len(self.video_paths) == 0:
                print("WARNING: No video files found. Falling back to mock dataset.")
                self.mock = True
                self.mock_pairs = []
                for _ in range(100):
                    f1 = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                    f2 = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                    self.mock_pairs.append((f1, f2))

    def __len__(self):
        if self.mock:
            return len(self.mock_pairs)
        return self.epoch_len

    def __getitem__(self, idx):
        if self.mock:
            frame_t, frame_t1 = self.mock_pairs[idx % len(self.mock_pairs)]
        else:
            # Pick a video path based on index
            video_idx = idx % len(self.video_paths)
            video_path = self.video_paths[video_idx]
            
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= self.frame_skip + 1:
                cap.release()
                # Fallback: recursively pick another random video index
                return self.__getitem__(random.randint(0, len(self.video_paths) - 1))
            
            # Select random starting frame
            t = random.randint(0, total_frames - self.frame_skip - 1)
            
            # Read frame t
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret_t, frame_t = cap.read()
            
            # Read frame t + skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, t + self.frame_skip)
            ret_t1, frame_t1 = cap.read()
            
            cap.release()
            
            if not ret_t or not ret_t1 or frame_t is None or frame_t1 is None:
                # Fallback: recursively pick another random video index
                return self.__getitem__(random.randint(0, len(self.video_paths) - 1))
                
            frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
            frame_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2RGB)
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        
        return transform(frame_t), transform(frame_t1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TDV OBJECTIVES & ADV ATTACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def freeze_stem(model, freeze=True):
    raw_model = get_raw_model(model)
    stem_module = getattr(raw_model, 'stem', None)
    if stem_module is not None:
        for p in stem_module.parameters():
            p.requires_grad = not freeze
        print(f"Stem {'frozen' if freeze else 'unfrozen'}")
    else:
        print("WARNING: Model stem module not found.")

def tdv_loss_large(model, x_t, x_t1):
    """Causal TDV prediction loss on base or large model."""
    raw_model = get_raw_model(model)
    cls_t = raw_model.get_feature_vector(x_t)
    cls_t1 = raw_model.get_feature_vector(x_t1)
    
    z_t = raw_model.tdv_head(cls_t)
    z_t1 = raw_model.tdv_head(cls_t1)
    z_t1_detach = z_t1.detach()
    
    m_proj = raw_model.motion_encoder(x_t, x_t1)
    z_t1_pred = z_t + m_proj
    
    # 1. Prediction discrepancy
    l_pred = F.mse_loss(z_t1_pred, z_t1_detach)
    
    # 2. Variance loss (VICReg)
    std_t  = torch.sqrt(z_t.var(dim=0) + 1e-4)
    std_t1 = torch.sqrt(z_t1.var(dim=0) + 1e-4)
    l_var = (F.relu(1 - std_t) + F.relu(1 - std_t1)).mean()
    
    # 3. Variance loss on raw unprojected features
    std_cls_t = torch.sqrt(cls_t.var(dim=0) + 1e-4)
    std_cls_t1 = torch.sqrt(cls_t1.var(dim=0) + 1e-4)
    l_var_raw = (F.relu(1 - std_cls_t) + F.relu(1 - std_cls_t1)).mean()
    
    # 4. Covariance loss (decorrelation)
    B, D = z_t.shape
    z_tc = z_t - z_t.mean(dim=0)
    cov = (z_tc.T @ z_tc) / (B - 1)
    l_cov = (cov**2).sum() - (cov.diagonal()**2).sum()
    l_cov = l_cov / D
    
    loss = 25.0 * l_pred + 25.0 * l_var + 1.0 * l_cov + 25.0 * l_var_raw
    return loss, l_pred, l_var, l_cov

def pgd_attack_large(model, x_t, x_t1, eps=0.031, steps=3):
    """PGD attack targeting temporal prediction consistency."""
    raw_model = get_raw_model(model)
    model.eval()
    x_adv = x_t.clone().detach() + 0.001 * torch.randn_like(x_t)
    
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)
    stl_min = torch.tensor([-(m/s) for m, s in zip(mean, std)]).view(1,3,1,1).to(x_t.device)
    stl_max = torch.tensor([(1-m)/s for m, s in zip(mean, std)]).view(1,3,1,1).to(x_t.device)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)

    with torch.no_grad():
        z_t1 = raw_model.tdv_head(raw_model.get_feature_vector(x_t1)).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                z_t_adv = raw_model.tdv_head(raw_model.get_feature_vector(x_adv))
                m_proj = raw_model.motion_encoder(x_adv, x_t1)
                z_t1_pred = z_t_adv + m_proj
                loss = F.mse_loss(z_t1_pred, z_t1)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + (eps / steps) * grad.sign()
        delta = torch.clamp(x_adv - x_t, -eps, eps)
        x_adv = torch.clamp(x_t + delta, stl_min, stl_max).detach()

    model.train()
    return x_adv

def adversarial_tdv_loss_large(model, x_t, x_t1, eps=0.031, steps=3):
    """Enforces consistency on adversarially perturbed frame sequence."""
    x_t_adv = pgd_attack_large(model, x_t, x_t1, eps=eps, steps=steps)
    loss_clean, l_pred, l_var, l_cov = tdv_loss_large(model, x_t, x_t1)
    loss_adv, _, _, _ = tdv_loss_large(model, x_t_adv, x_t1)
    return 0.5 * loss_clean + 0.5 * loss_adv, l_pred, l_var, l_cov

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING PHASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_video_tdv(model, video_loader, device, ckpt_path, accum_steps=1):
    print("\n" + "="*70)
    print("RUNNING UCF-101 VIDEO TDV PRETRAINING (stem frozen)")
    print("="*70)
    
    freeze_stem(model, freeze=True)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler('cuda')
    
    start_epoch = 1
    resume_path = ckpt_path.replace('.pth', '_resume.pth')
    if os.path.exists(resume_path):
        print(f"Resuming video TDV training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        get_raw_model(model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Starting from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, 11):
        t0 = time.time()
        total_loss = total_pred = total_var = n_total = 0
        
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (x_t, x_t1) in enumerate(video_loader):
            x_t = x_t.to(device, non_blocking=True)
            x_t1 = x_t1.to(device, non_blocking=True)
            
            with autocast('cuda'):
                loss, l_pred, l_var, _ = tdv_loss_large(model, x_t, x_t1)
                loss = loss / accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            B = x_t.size(0)
            total_loss += (loss.item() * accum_steps) * B
            total_pred += l_pred.item() * B
            total_var  += l_var.item() * B
            n_total    += B
            
        if len(video_loader) % accum_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        scheduler.step()
        if n_total > 0:
            print(f"Epoch {epoch:02d}/10 | Loss: {total_loss/n_total:.4f} | l_pred: {total_pred/n_total:.4f} | l_var: {total_var/n_total:.4f} | {time.time()-t0:.0f}s")
        else:
            print(f"Epoch {epoch:02d}/10 | Loss: 0.0000 | l_pred: 0.0000 | l_var: 0.0000 | {time.time()-t0:.0f}s (No batches processed)")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': get_raw_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, resume_path)
        
    torch.save(get_raw_model(model).state_dict(), ckpt_path)
    print(f"Saved pre-trained video TDV model to {ckpt_path}")
    if os.path.exists(resume_path):
        os.remove(resume_path)

def run_trades_finetuning(model, trainloader, testloader, video_loader, device, ckpt_path, accum_steps=1):
    print("\n" + "="*70)
    print("RUNNING TRADES ADVERSARIAL FINE-TUNING (UCF-101 video temporal consistency)")
    print("="*70)
    
    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True
        
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()
    video_iter = iter(video_loader)
    
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)
    stl_min = torch.tensor([-(m/s) for m, s in zip(mean, std)]).view(1,3,1,1).to(device)
    stl_max = torch.tensor([(1-m)/s for m, s in zip(mean, std)]).view(1,3,1,1).to(device)
 
    # Curriculum Setup (extended to 120 epochs for complete convergence)
    curriculum = [
        (1,  40, 0.031, 2.0, 7,  0.003),
        (41, 80, 0.062, 2.0, 10, 0.002),
        (81, 120, 0.094, 2.5, 10, 0.001),
    ]
 
    current_phase_start = None
    optimizer = None
    scheduler = None
    best_acc = 0.0
    start_epoch = 1

    resume_path = ckpt_path.replace('.pth', '_resume.pth')
    if os.path.exists(resume_path):
        print(f"Resuming TRADES fine-tuning from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        get_raw_model(model).load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        
        # Pre-initialize the optimizer and scheduler for the resumed epoch's phase
        resume_epoch = checkpoint['epoch']
        for p_start, p_end, eps, beta, steps, lr in curriculum:
            if p_start <= resume_epoch <= p_end:
                current_phase_start = p_start
                optimizer = optim.SGD(
                    model.parameters(), lr=lr,
                    momentum=0.9, weight_decay=1e-4
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=p_end - p_start + 1, eta_min=lr * 0.1
                )
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Restored optimizer and scheduler state for phase starting at epoch {p_start}. Resuming at epoch {start_epoch}.")
                break
 
    for epoch in range(start_epoch, 121):
        t0 = time.time()
        
        # Determine curriculum parameters
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
        
        model.train()
        total_loss = total_tr = total_cons = n_total = correct = 0
        
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (imgs, lbls) in enumerate(trainloader):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            
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
                logits_c = model(imgs)
                logits_a = model(x_adv)
                
                l_trades = ce_loss(logits_c, lbls) + beta * F.kl_div(
                    F.log_softmax(logits_a.float(), dim=1),
                    F.softmax(logits_c.float().detach(), dim=1),
                    reduction='batchmean'
                )
                loss = l_trades / accum_steps
 
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
 
            B = imgs.size(0)
            total_loss += (loss.item() * accum_steps) * B
            total_tr   += l_trades.item() * B
            correct    += logits_c.argmax(1).eq(lbls).sum().item()
            n_total    += B
 
        if len(trainloader) % accum_steps != 0:
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
            torch.save(get_raw_model(model).state_dict(), ckpt_path)
            marker = ' ★'
 
        print(
            f"Epoch {epoch:02d}/120 (ε={eps:.3f}) | Loss:{total_loss/n_total:.3f} | "
            f"TrAcc:{100.*correct/n_total:.1f}% TeAcc:{val_acc:.1f}% | "
            f"{time.time()-t0:.0f}s{marker}"
        )
        
        # Save resume checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': get_raw_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, resume_path)
 
    print(f"Finetuning Complete. Model saved to {ckpt_path}")
    if os.path.exists(resume_path):
        os.remove(resume_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='tdv', choices=['tdv', 'trades'])
    parser.add_argument('--model-size', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model size: {args.model_size}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Instantiate model size
    if args.model_size == 'large':
        model = RHANLargeSTL10().to(device)
        ckpt_path = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv.pth')
    else:
        model = RHANUnifiedSTL10().to(device)
        ckpt_path = os.path.join(ckpt_dir, 'rhan_stl10_base_video_tdv.pth')

    # Load video dataset
    ucf_root = os.path.join(args.data_root, 'ucf101')
    video_dataset = UCF101TemporalDataset(ucf_root=ucf_root, categories=list(UCF_RELEVANT_CATEGORIES.keys()))
    
    # Ensure drop_last is only True if dataset size exceeds batch size to prevent ZeroDivisionError
    drop_last = len(video_dataset) > args.batch_size
    batch_size = min(args.batch_size, len(video_dataset)) if len(video_dataset) > 0 else args.batch_size
    video_loader = DataLoader(
        video_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=drop_last
    )

    # Load pre-trained checkpoint if it exists (before DataParallel wrapping)
    if args.phase == 'trades':
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded pretrained checkpoint: {ckpt_path}")
        else:
            print("WARNING: Pretrained TDV checkpoint not found! Fine-tuning from scratch.")

    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

        stl_data_root = os.path.join(args.data_root, 'stl10')
        trainloader, testloader, _, _ = get_stl10_dataloaders(data_root=stl_data_root, batch_size=64)
        run_trades_finetuning(model, trainloader, testloader, video_loader, device, ckpt_path, args.accum_steps)

if __name__ == '__main__':
    main()
