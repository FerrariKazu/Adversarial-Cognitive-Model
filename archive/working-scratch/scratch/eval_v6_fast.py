import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_v6 import RHANv6
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = "checkpoints/rhan_v6_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return
        
    model = RHANv6(head_type='cosine').to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Disable gradients for parameters
    for p in model.parameters():
        p.requires_grad = False
        
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out
            
    wrapper = Wrapper(model)
    
    _, testloader_raw = get_dataloaders(batch_size=64, num_workers=2, model_name='resnet')
    testloader = DataLoader(testloader_raw.dataset, batch_size=64, shuffle=False)
    
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)
    
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 256  # small sample count to be fast and memory friendly
    
    print("\nEvaluating RHAN-v6 (Fast)...")
    for eps in epsilons:
        correct = 0
        total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, labels in testloader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(
                    wrapper, images, labels, epsilon=eps, alpha=alpha,
                    steps=20, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True
                )
            else:
                adv_images = images
            with torch.no_grad():
                outputs = wrapper(adv_images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / max(total, 1)
        print(f"  ε={eps:.2f} → Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
