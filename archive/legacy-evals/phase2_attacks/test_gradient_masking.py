#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from phase1_training.model_rhan import RHAN
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

def evaluate_random_noise(model, dataloader, device, cifar_min, cifar_max, epsilon, max_samples=1000):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            if total >= max_samples:
                break
            images = images.to(device)
            labels = labels.to(device)
            
            # Apply uniform random noise in [-epsilon, epsilon]
            noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
            noisy_images = images + noise
            
            # Clamp to valid image range
            noisy_images = torch.max(torch.min(noisy_images, cifar_max), cifar_min)
            
            outputs = model(noisy_images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total

def evaluate_pgd(model, dataloader, device, cifar_min, cifar_max, epsilon, alpha, steps, max_samples=1000):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        if total >= max_samples:
            break
        images = images.to(device)
        labels = labels.to(device)
        
        _, predicted = pgd_attack(
            model, images, labels,
            epsilon=epsilon, alpha=alpha, steps=steps,
            device=device, clip_min=cifar_min, clip_max=cifar_max,
            random_start=True
        )
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = RHAN(num_classes=10).to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'rhan_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Loaded RHAN checkpoint.")
    
    # Load data
    _, testloader = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')
    
    # CIFAR bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)
    
    epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]
    
    print("\n" + "="*60)
    print("TEST 1: RANDOM NOISE BASELINE (Checking for Gradient Masking)")
    print("="*60)
    print(f"{'Epsilon':<10} | {'Random Noise Acc':<20}")
    print("-" * 35)
    
    for eps in epsilons:
        acc = evaluate_random_noise(model, testloader, device, cifar_min, cifar_max, eps)
        print(f"ε={eps:<9.2f} | {acc:>18.2f}%")
        
    print("\n" + "="*60)
    print("TEST 2: PGD-100 ROBUSTNESS AT ε=0.05")
    print("="*60)
    print("Running PGD with 100 iterations (alpha=0.01) at ε=0.05...")
    
    # Run PGD-100 at eps=0.05
    acc_pgd100 = evaluate_pgd(
        model, testloader, device, cifar_min, cifar_max,
        epsilon=0.05, alpha=0.01, steps=100
    )
    
    print(f"PGD-100 Accuracy at ε=0.05: {acc_pgd100:.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()
