#!/usr/bin/env python3
"""
Experiment 2 & 3 — Real Video TDV on UCF-101 (for Colab A100/V100)
==================================================================

Training loop for Video Temporal Difference Vision (TDV) pretraining,
labeled head calibration, and adversarial fine-tuning.
Supports base (RHANUnifiedSTL10) and large (RHANLargeSTL10) architectures.
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

def sync_to_hf(file_path):
    try:
        from huggingface_hub import HfApi, create_repo
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            try:
                from kaggle_secrets import UserSecretsClient
                hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
        if hf_token:
            api = HfApi(token=hf_token)
            username = api.whoami()['name']
            repo_id = f"{username}/rhan-checkpoints"
            
            # Ensure repository exists
            try:
                create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=hf_token)
            except Exception:
                pass
                
            filename = os.path.basename(file_path)
            print(f"Syncing {filename} to Hugging Face ({repo_id})...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token
            )
            print("Sync complete.")
        else:
            print("WARNING: HF_TOKEN not found in environment or secrets. Skipping Hugging Face sync.")
    except Exception as e:
        print(f"Hugging Face sync failed: {e}")

def download_from_hf(file_path):
    try:
        from huggingface_hub import HfApi, hf_hub_download, create_repo
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            try:
                from kaggle_secrets import UserSecretsClient
                hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
        if hf_token:
            api = HfApi(token=hf_token)
            username = api.whoami()['name']
            repo_id = f"{username}/rhan-checkpoints"
            
            # Ensure repository exists
            try:
                create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=hf_token)
            except Exception:
                pass
                
            filename = os.path.basename(file_path)
            try:
                files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                if filename in files:
                    print(f"Downloading {filename} from Hugging Face ({repo_id})...")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=os.path.dirname(file_path),
                        local_dir_use_symlinks=False,
                        token=hf_token
                    )
                    print("Downloaded successfully from Hugging Face.")
                    return True
            except Exception as e:
                print(f"Hugging Face files check failed: {e}")
        else:
            print("WARNING: HF_TOKEN not found in environment or secrets. Skipping Hugging Face download.")
    except Exception as e:
        print(f"Hugging Face setup check failed: {e}")
    return False

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

class UCF101TemporalDataset(Dataset):
    """
    Loads consecutive frame pairs (t, t+1) from UCF-101 categories
    to enforce temporal causality z_t + m_t = z_{t+1}.
    """
    def __init__(self, ucf_root, categories, transform=None):
        self.ucf_root = ucf_root
        self.categories = categories
        self.samples = []
        
        # Build samples list: for each category, find video directories
        for cat in categories:
            cat_dir = os.path.join(ucf_root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for video_dir in sorted(os.listdir(cat_dir)):
                v_path = os.path.join(cat_dir, video_dir)
                if not os.path.isdir(v_path):
                    continue
                # List frame images in sorted order
                frames = sorted([f for f in os.listdir(v_path) if f.endswith(('.jpg', '.png'))])
                # Add consecutive frame pairs
                for i in range(len(frames) - 1):
                    self.samples.append((
                        os.path.join(v_path, frames[i]),
                        os.path.join(v_path, frames[i+1])
                    ))
                    
        self.to_tensor = T.Compose([
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713))
        ])
        print(f"Loaded UCF-101 dataset: found {len(self.samples)} frame pairs across {len(categories)} categories.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f1_path, f2_path = self.samples[idx]
        from PIL import Image
        img1 = Image.open(f1_path).convert('RGB')
        img2 = Image.open(f2_path).convert('RGB')
        
        # Temporal random crop matching to simulate motion translation
        w, h = img1.size
        i1, j1, h1, w1 = T.RandomResizedCrop.get_params(img1, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        # Add slight offset to frame 2 for motion encoding
        dy = random.randint(-4, 4)
        dx = random.randint(-4, 4)
        i2 = max(0, min(i1 + dy, h - h1))
        j2 = max(0, min(j1 + dx, w - w1))
        
        x_t = TF.resized_crop(img1, i1, j1, h1, w1, (96, 96))
        x_t1 = TF.resized_crop(img2, i2, j2, h1, w1, (96, 96))
        
        return self.to_tensor(x_t), self.to_tensor(x_t1)

def freeze_stem(model, freeze=True):
    raw_model = get_raw_model(model)
    for name, param in raw_model.named_parameters():
        if "stem" in name:
            param.requires_grad = not freeze

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TDV OBJECTIVES & ADV ATTACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tdv_loss_large(model, x_t, x_t1):
    raw_model = get_raw_model(model)
    
    # 1. Get representations for both timesteps
    z_t = raw_model.get_feature_vector(x_t)
    z_t1 = raw_model.get_feature_vector(x_t1)
    
    # 2. Get motion vector and projected targets
    m_t = raw_model.motion_encoder(x_t, x_t1)
    p_t = raw_model.tdv_head(z_t)
    p_t1 = raw_model.tdv_head(z_t1)
    
    # 3. TDV Loss components (VICReg variant)
    l_pred = F.mse_loss(p_t + m_t, p_t1)
    
    # Variance Regularization to prevent feature representation collapse
    std_t = torch.sqrt(p_t.var(dim=0) + 1e-4)
    std_t1 = torch.sqrt(p_t1.var(dim=0) + 1e-4)
    l_var = torch.mean(F.relu(1.0 - std_t)) + torch.mean(F.relu(1.0 - std_t1))
    
    # Covariance Decorrelation
    B, D = p_t.shape
    p_t_mean = p_t - p_t.mean(dim=0)
    p_t1_mean = p_t1 - p_t1.mean(dim=0)
    cov_t = (p_t_mean.T @ p_t_mean) / (B - 1)
    cov_t1 = (p_t1_mean.T @ p_t1_mean) / (B - 1)
    
    l_cov = (cov_t.pow(2).sum() - cov_t.diagonal().pow(2).sum()) / D + \
            (cov_t1.pow(2).sum() - cov_t1.diagonal().pow(2).sum()) / D
            
    total_loss = 25.0 * l_pred + 25.0 * l_var + 1.0 * l_cov
    return total_loss, l_pred, l_var, l_cov

def pgd_attack_large(model, x_t, x_t1, eps=0.031, steps=3):
    model.eval()
    x_adv = x_t.clone().detach() + torch.empty_like(x_t).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, -2.0, 2.0)
    
    alpha = eps / steps if steps > 0 else 0.0
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss, _, _, _ = tdv_loss_large(model, x_adv, x_t1)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_t, -eps, eps)
        x_adv = torch.clamp(x_t + delta, -2.0, 2.0).detach()
    return x_adv

def adversarial_tdv_loss_large(model, x_t, x_t1, eps=0.031, steps=3):
    x_t_adv = pgd_attack_large(model, x_t, x_t1, eps=eps, steps=steps)
    loss_clean, l_pred, l_var, l_cov = tdv_loss_large(model, x_t, x_t1)
    loss_adv, _, _, _ = tdv_loss_large(model, x_t_adv, x_t1)
    return 0.5 * loss_clean + 0.5 * loss_adv, l_pred, l_var, l_cov

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING PHASES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_video_tdv(model, video_loader, device, ckpt_path, accum_steps=1):
    print("\n" + "="*70)
    print("PHASE 0: RUNNING UCF-101 VIDEO TDV PRETRAINING (stem frozen)")
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
    if not os.path.exists(resume_path):
        download_from_hf(resume_path)
        
    if os.path.exists(resume_path):
        print(f"Resuming video TDV training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
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
        sync_to_hf(resume_path)
        
    torch.save(get_raw_model(model).state_dict(), ckpt_path)
    print(f"Saved pre-trained video TDV model to {ckpt_path}")
    sync_to_hf(ckpt_path)
    if os.path.exists(resume_path):
        os.remove(resume_path)

def run_label_calibration(model, trainloader, testloader, device, ckpt_path):
    print("\n" + "="*70)
    print("PHASE 1: LABELED CLASSIFIER HEAD CALIBRATION (backbone frozen)")
    print("="*70)
    
    # Freeze the entire backbone, only train the classifier head
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(1, 11):
        t0 = time.time()
        model.train()
        total_loss = correct = n_total = 0
        for imgs, lbls in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(imgs)
                loss = ce_loss(logits, lbls)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            B = imgs.size(0)
            total_loss += loss.item() * B
            correct += logits.argmax(1).eq(lbls).sum().item()
            n_total += B
            
        # Validation on clean test split
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
        print(f"Calibration Epoch {epoch:02d}/10 | Loss: {total_loss/n_total:.4f} | TrAcc: {100.*correct/n_total:.1f}% TeAcc: {val_acc:.1f}% | {time.time()-t0:.0f}s")
        
    torch.save(get_raw_model(model).state_dict(), ckpt_path)
    print(f"Classifier head calibration complete. Saved model to {ckpt_path}")
    sync_to_hf(ckpt_path)

def run_trades_finetuning(model, trainloader, testloader, device, ckpt_path, accum_steps=1):
    print("\n" + "="*70)
    print("PHASE 2: RUNNING TRADES ADVERSARIAL FINE-TUNING")
    print("="*70)
    
    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True
        
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()
    
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
    if not os.path.exists(resume_path):
        download_from_hf(resume_path)
        
    if os.path.exists(resume_path):
        print(f"Resuming TRADES fine-tuning from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
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
        total_loss = total_tr = n_total = correct = 0
        
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
            sync_to_hf(ckpt_path)
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
        sync_to_hf(resume_path)
 
    print(f"Finetuning Complete. Model saved to {ckpt_path}")
    if os.path.exists(resume_path):
        os.remove(resume_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='tdv', choices=['tdv', 'label', 'trades'])
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
        tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_pretrained.pth')
        labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_labeled.pth')
        final_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv.pth')
    else:
        model = RHANUnifiedSTL10().to(device)
        tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_base_video_tdv_pretrained.pth')
        labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_base_video_tdv_labeled.pth')
        final_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_base_video_tdv.pth')

    # Load datasets
    stl_data_root = os.path.join(args.data_root, 'stl10')
    trainloader, testloader, _, _ = get_stl10_dataloaders(data_root=stl_data_root, batch_size=64)

    if args.phase == 'tdv':
        ucf_root = os.path.join(args.data_root, 'ucf101')
        video_dataset = UCF101TemporalDataset(ucf_root=ucf_root, categories=list(UCF_RELEVANT_CATEGORIES.keys()))
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
        run_video_tdv(model, video_loader, device, tdv_ckpt, args.accum_steps)
        
    elif args.phase == 'label':
        if not os.path.exists(tdv_ckpt):
            download_from_hf(tdv_ckpt)
        if os.path.exists(tdv_ckpt):
            model.load_state_dict(torch.load(tdv_ckpt, map_location=device, weights_only=False), strict=False)
            print(f"Loaded pretrained backbone weights from {tdv_ckpt}")
        else:
            print("WARNING: Pre-trained backbone checkpoint not found! Calibrating from scratch.")
        run_label_calibration(model, trainloader, testloader, device, labeled_ckpt)
        
    elif args.phase == 'trades':
        if not os.path.exists(labeled_ckpt):
            download_from_hf(labeled_ckpt)
        if os.path.exists(labeled_ckpt):
            model.load_state_dict(torch.load(labeled_ckpt, map_location=device, weights_only=False))
            print(f"Loaded calibrated head checkpoint: {labeled_ckpt}")
        else:
            print("WARNING: Calibrated checkpoint not found! Running TRADES from scratch.")
            
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
            
        run_trades_finetuning(model, trainloader, testloader, device, final_ckpt, args.accum_steps)

if __name__ == '__main__':
    main()
