#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase1_training'))

from phase1_training.model_rhan_v9 import RHANv9
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../checkpoints/rhan_v9_sail.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    model = RHANv9(head_type='cosine').to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # Logit-only wrapper
    class W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    wrapper = W(model)

    # Get data
    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=0, model_name='resnet')
    testloader = DataLoader(testloader_raw.dataset, batch_size=16, shuffle=False, num_workers=0)

    # Setup bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # Eval Clean & PGD-20
    clean_correct = 0
    pgd_correct = 0
    total = 0
    max_samples = 1000

    print("Evaluating clean and PGD-20 (eps=0.031) on 1000 test samples...")
    for imgs, lbls in testloader:
        if total >= max_samples:
            break
        imgs, lbls = imgs.to(device), lbls.to(device)
        B = imgs.size(0)

        with torch.no_grad():
            clean_logits = wrapper(imgs)
            clean_preds = clean_logits.argmax(dim=1)
            clean_correct += clean_preds.eq(lbls).sum().item()

        # Run PGD
        _, pgd_preds = pgd_attack(
            wrapper, imgs, lbls,
            epsilon=0.031, alpha=0.008, steps=20,
            device=device, clip_min=cifar_min, clip_max=cifar_max
        )
        pgd_correct += pgd_preds.eq(lbls).sum().item()
        total += B

    print("=" * 60)
    print(f"Clean Accuracy:  {100. * clean_correct / total:.2f}%")
    print(f"PGD-20 Accuracy: {100. * pgd_correct / total:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
