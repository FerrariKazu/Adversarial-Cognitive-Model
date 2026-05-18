#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model_rhan import RHAN
from dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    
    # CRITICAL: persistent_workers=False avoids the PyTorch IPC deadlock
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
        prefetch_factor=3,
    )

    # 2. Load model
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'rhan_v2_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: Could not find checkpoint {ckpt_path}")
        return

    eval_model = RHAN(num_classes=10, head_type='linear').to(device)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    eval_model.eval()

    # 3. Setup PGD bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    def eval_pgd(model, loader, eps, alpha, steps, max_samples=1000):
        correct = 0
        total = 0
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            _, predicted = pgd_attack(
                model, images, labels,
                epsilon=eps, alpha=alpha, steps=steps,
                device=device, clip_min=cifar_min, clip_max=cifar_max,
            )
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    print(f"\n{'='*70}")
    print(f"GRADIENT MASKING CHECK — PGD-20 vs PGD-100 at ε=0.05")
    print(f"{'='*70}")
    
    print("Running PGD-20 at ε=0.05...")
    acc_pgd20 = eval_pgd(eval_model, testloader, eps=0.05, alpha=0.01, steps=20)
    print(f"  PGD-20:  {acc_pgd20:.2f}%")

    print("Running PGD-100 at ε=0.05...")
    acc_pgd100 = eval_pgd(eval_model, testloader, eps=0.05, alpha=0.005, steps=100)
    print(f"  PGD-100: {acc_pgd100:.2f}%")

    gap = acc_pgd20 - acc_pgd100
    print(f"\n  Gap (PGD-20 - PGD-100): {gap:.2f}%")
    if gap < 5:
        print(f"  ✓ Gap < 5% → NO gradient masking detected. Proceed to adversarial training.")
    elif gap < 10:
        print(f"  ⚠ Gap 5-10% → Mild gradient masking. Proceed with caution.")
    else:
        print(f"  ✗ Gap > 10% → SIGNIFICANT gradient masking. Alert Mina before continuing.")

if __name__ == '__main__':
    main()
