#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./phase1_training'))

from phase1_training.model_rhan_stl10_large import RHANLargeSTL10
from phase1_training.dataset_stl10 import STL10_MEAN, STL10_STD, STL10_MIN, STL10_MAX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STL10RawUnlabeledDataset(Dataset):
    def __init__(self, data_root='./data/stl10'):
        self.stl10 = torchvision.datasets.STL10(
            data_root, split='unlabeled', download=False
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
    def __init__(self, real_imgs, real_labels, 
                 unlabeled_dataset, pseudo_indices, pseudo_labels, transform=None):
        self.real_imgs = real_imgs
        self.real_labels = real_labels
        self.pseudo_indices = pseudo_indices
        self.pseudo_labels = pseudo_labels
        self.transform = transform
        
        self.n_real = len(real_imgs)
        self.n_pseudo = len(pseudo_indices)
        
        # Cache normalized pseudo-labeled images in RAM using pre-allocated tensor to avoid OOM
        print(f"Caching {self.n_pseudo} pseudo-labeled images in RAM...", flush=True)
        self.cached_pseudo_imgs = torch.zeros(self.n_pseudo, 3, 96, 96, dtype=torch.float16)
        temp_loader = DataLoader(
            torch.utils.data.Subset(unlabeled_dataset, pseudo_indices.tolist()),
            batch_size=256, shuffle=False, num_workers=0,
            pin_memory=False
        )
        idx_start = 0
        for batch_imgs, _ in temp_loader:
            num_imgs = batch_imgs.size(0)
            self.cached_pseudo_imgs[idx_start : idx_start + num_imgs] = batch_imgs.to(torch.float16)
            idx_start += num_imgs
        
        print(f"Combined dataset: {self.n_real} real + "
              f"{self.n_pseudo} pseudo = {self.n_real+self.n_pseudo} total. Caching complete.", flush=True)
    
    def __len__(self):
        return self.n_real + self.n_pseudo
    
    def __getitem__(self, idx):
        if idx < self.n_real:
            img = self.real_imgs[idx]
            label = self.real_labels[idx]
            weight = 1.0
        else:
            img = self.cached_pseudo_imgs[idx - self.n_real].to(torch.float32)
            label = self.pseudo_labels[idx - self.n_real]
            weight = 0.5
        
        if self.transform:
            img = self.transform(img)
        return img, label, torch.tensor(weight, dtype=torch.float32)

class BalancedBatchSampler(Sampler):
    def __init__(self, real_indices, pseudo_indices, batch_size):
        self.real_indices = real_indices
        self.pseudo_indices = pseudo_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        
    def __iter__(self):
        import random
        random.shuffle(self.real_indices)
        random.shuffle(self.pseudo_indices)
        
        real_idx = 0
        pseudo_idx = 0
        
        while real_idx + self.half_batch <= len(self.real_indices) and pseudo_idx + self.half_batch <= len(self.pseudo_indices):
            batch = []
            for _ in range(self.half_batch):
                batch.append(self.real_indices[real_idx])
                real_idx += 1
            for _ in range(self.half_batch):
                batch.append(self.pseudo_indices[pseudo_idx])
                pseudo_idx += 1
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(self.real_indices), len(self.pseudo_indices)) // self.half_batch

def trades_loss_weighted(model, x_natural, y, weights, x_adv, beta):
    criterion_kl = nn.KLDivLoss(reduction='none')
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    
    loss_natural = F.cross_entropy(logits_natural, y, reduction='none')
    p_natural = F.softmax(logits_natural.float(), dim=1)
    loss_robust = criterion_kl(F.log_softmax(logits_adv.float(), dim=1), p_natural).sum(dim=1)
    
    loss = loss_natural + beta * loss_robust
    return (loss * weights).mean()

def main():
    print(f"Device: {device}")
    
    # 1. Setup mock data loader
    print("Setting up dataset loader...")
    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    ])
    
    # Load raw datasets locally (avoid downloading if possible)
    trainset_raw = torchvision.datasets.STL10('./data/stl10', split='train', download=False)
    real_imgs = torch.stack([norm_transform(trainset_raw[i][0]) for i in range(len(trainset_raw))])
    real_labels = torch.tensor([trainset_raw[i][1] for i in range(len(trainset_raw))])
    
    unlabeled_dataset = STL10RawUnlabeledDataset('./data/stl10')
    
    # Mock pseudo-labels
    mock_pseudo_indices = torch.arange(0, 5000)
    mock_pseudo_labels = torch.randint(0, 10, (5000,))
    
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
    ])
    
    combined_dataset = CombinedSTL10Dataset(
        real_imgs, real_labels, unlabeled_dataset, mock_pseudo_indices, mock_pseudo_labels, transform=train_transform
    )
    
    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(real_imgs) + len(mock_pseudo_indices)))
    
    sampler = BalancedBatchSampler(real_indices, pseudo_indices_list, batch_size=32)
    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)
    
    # Bounds for PGD clamping
    stl_min = torch.tensor(STL10_MIN).view(1,3,1,1).to(device)
    stl_max = torch.tensor(STL10_MAX).view(1,3,1,1).to(device)

    # 2. Instantiate Model
    print("Instantiating model...")
    model = RHANLargeSTL10().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
        
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler('cuda')
    
    steps = 10
    eps = 0.062
    beta = 2.0
    
    # Warm-up step
    print("Warming up...")
    model.eval()
    dummy_x = torch.randn(2, 3, 96, 96, device=device)
    dummy_y = torch.randint(0, 10, (2,), device=device)
    dummy_w = torch.ones(2, device=device)
    with autocast('cuda'):
        loss = trades_loss_weighted(model, dummy_x, dummy_y, dummy_w, dummy_x, beta)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    
    print("\nStarting profiling over 20 steps...")
    
    # Timers
    data_times = []
    gpu_transfer_times = []
    forward_c_times = []
    pgd_times = []
    forward_t_times = []
    backward_times = []
    opt_times = []
    total_step_times = []
    
    t_start = time.time()
    
    # CUDA events for profiling
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    loader_iter = iter(trainloader)
    
    for step in range(20):
        # Time data loading
        t0 = time.time()
        try:
            imgs, lbls, weights = next(loader_iter)
        except StopIteration:
            loader_iter = iter(trainloader)
            imgs, lbls, weights = next(loader_iter)
        t_data = (time.time() - t0) * 1000.0 # ms
        data_times.append(t_data)
        
        # Start total step timing
        torch.cuda.synchronize()
        step_start_t = time.time()
        
        # Host to Device transfer
        start_event.record()
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        gpu_transfer_times.append(start_event.elapsed_time(end_event))
        
        # Forward pass (clean)
        model.eval()
        start_event.record()
        with torch.no_grad():
            with autocast('cuda'):
                logits_c = model(imgs)
        end_event.record()
        torch.cuda.synchronize()
        forward_c_times.append(start_event.elapsed_time(end_event))
        
        # PGD adversarial generation
        start_event.record()
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
        end_event.record()
        torch.cuda.synchronize()
        pgd_times.append(start_event.elapsed_time(end_event))
        
        # Forward pass (TRADES loss)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start_event.record()
        with autocast('cuda'):
            l_trades = trades_loss_weighted(model, imgs, lbls, weights, x_adv, beta)
            loss = l_trades / 1 # accum steps = 1
        end_event.record()
        torch.cuda.synchronize()
        forward_t_times.append(start_event.elapsed_time(end_event))
        
        # Backward pass
        start_event.record()
        scaler.scale(loss).backward()
        end_event.record()
        torch.cuda.synchronize()
        backward_times.append(start_event.elapsed_time(end_event))
        
        # Optimizer step
        start_event.record()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        end_event.record()
        torch.cuda.synchronize()
        opt_times.append(start_event.elapsed_time(end_event))
        
        step_end_t = time.time()
        total_step_times.append((step_end_t - step_start_t) * 1000.0)
        
    print("\n" + "="*50)
    print("          PROFILING RESULTS (Average of 20 steps)")
    print("="*50)
    print(f"Data loading (CPU)    : {sum(data_times)/len(data_times):.2f} ms")
    print(f"GPU Transfer          : {sum(gpu_transfer_times)/len(gpu_transfer_times):.2f} ms")
    print(f"Clean Forward Pass    : {sum(forward_c_times)/len(forward_c_times):.2f} ms")
    print(f"PGD Generation (10s)  : {sum(pgd_times)/len(pgd_times):.2f} ms")
    print(f"Trades Forward Pass   : {sum(forward_t_times)/len(forward_t_times):.2f} ms")
    print(f"Backward Pass         : {sum(backward_times)/len(backward_times):.2f} ms")
    print(f"Optimizer Step        : {sum(opt_times)/len(opt_times):.2f} ms")
    print("-"*50)
    avg_total = sum(total_step_times)/len(total_step_times)
    print(f"Average Step Time     : {avg_total:.2f} ms")
    print(f"Estimated Throughput  : {32 * 1000.0 / avg_total:.2f} images/sec")
    print("="*50)

if __name__ == '__main__':
    main()
