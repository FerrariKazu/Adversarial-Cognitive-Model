#!/usr/bin/env python3
"""
RHAN-v6 Phase 0: Noise Estimator Pretraining
============================================
Trains a lightweight network (~200K parameters) to predict the perturbation
level (epsilon) of CIFAR-10 images. 
Generates corrupted batches on-the-fly with 50% L_infty uniform noise and 
50% Gaussian-smoothed noise.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import get_dataloaders

# Model architecture as defined in the RHAN-v6 plan
class NoiseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # -> 1x1
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output normalized in [0, 1]
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoints directory
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, 'noise_estimator_pretrained.pth')

    # Load data
    trainloader, testloader = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')

    # Initialize model, optimizer, scheduler
    model = NoiseEstimator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # CIFAR-10 stats for clipping bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    print("\n" + "="*50)
    print("RHAN-v6 Phase 0: Pretraining Noise Estimator")
    print("="*50)
    print(f"  Epochs: 30")
    print(f"  Batch Size: 256")
    print(f"  Save Path: {save_path}")
    print("="*50 + "\n")

    for epoch in range(30):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for imgs, _ in trainloader:
            imgs = imgs.to(device)
            B = imgs.size(0)

            # Generate random epsilon levels [0.0, 0.250]
            eps = torch.rand(B, 1, 1, 1, device=device) * 0.250

            # L_infty uniform noise
            uniform_noise = (torch.rand_like(imgs) * 2 - 1) * eps

            # Gaussian smoothed noise
            gaussian_noise = torch.randn_like(imgs)
            smoothed_noise = F.avg_pool2d(gaussian_noise, kernel_size=3, stride=1, padding=1)
            smoothed_noise = torch.clamp(smoothed_noise, -1.0, 1.0) * eps

            # Select noise type randomly (50% uniform, 50% smoothed)
            mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).float()
            noise = mask * uniform_noise + (1.0 - mask) * smoothed_noise

            # Inject noise and clip
            noisy_imgs = torch.max(torch.min(imgs + noise, cifar_max), cifar_min)

            # Labels are normalized epsilons [0.0, 1.0]
            labels = eps.view(B, 1) / 0.250

            # Forward
            optimizer.zero_grad(set_to_none=True)
            preds = model(noisy_imgs)
            loss = F.mse_loss(preds, labels)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B

        scheduler.step()
        train_loss = train_loss / len(trainloader.dataset)

        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs, _ in testloader:
                imgs = imgs.to(device)
                B = imgs.size(0)

                eps = torch.rand(B, 1, 1, 1, device=device) * 0.250
                uniform_noise = (torch.rand_like(imgs) * 2 - 1) * eps
                gaussian_noise = torch.randn_like(imgs)
                smoothed_noise = F.avg_pool2d(gaussian_noise, kernel_size=3, stride=1, padding=1)
                smoothed_noise = torch.clamp(smoothed_noise, -1.0, 1.0) * eps

                mask = (torch.rand(B, 1, 1, 1, device=device) < 0.5).float()
                noise = mask * uniform_noise + (1.0 - mask) * smoothed_noise

                noisy_imgs = torch.max(torch.min(imgs + noise, cifar_max), cifar_min)
                labels = eps.view(B, 1) / 0.250

                preds = model(noisy_imgs)
                loss = F.mse_loss(preds, labels)
                test_loss += loss.item() * B

        test_loss = test_loss / len(testloader.dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/30 | Train MSE: {train_loss:.5f} | Test MSE: {test_loss:.5f} | {elapsed:.1f}s")

    # Save pretrained weights
    torch.save(model.state_dict(), save_path)
    print(f"\nPretraining finished! Checkpoint saved to: {save_path}\n")

if __name__ == '__main__':
    main()
