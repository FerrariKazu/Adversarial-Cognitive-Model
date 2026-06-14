import os
import sys
import torch
import argparse
from torch import nn
from model_rhan_v5 import RHANv5
from dataset import get_dataloaders

# pgd.py lives in phase2_attacks/, one level up from phase1_training/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2_attacks'))
from pgd import pgd_attack

def denormalize(tensor, mean, std):
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='RHAN v8 PGD-100 Evaluation')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('checkpoints', 'rhan_v8_best.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=200, help='Number of images to evaluate (max)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = RHANv5(head_type='cosine').to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Checkpoint may be a raw state_dict or wrapped in a dict with 'model' key
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load test data (normalized)
    _, testloader = get_dataloaders(batch_size=128, num_workers=0, model_name='resnet')

    # Prepare car/truck subset
    car_truck_imgs = []
    car_truck_lbls = []
    for imgs, lbls in testloader:
        mask = (lbls == 1) | (lbls == 9)
        if mask.any():
            car_truck_imgs.append(imgs[mask])
            car_truck_lbls.append(lbls[mask])
    car_truck_imgs = torch.cat(car_truck_imgs)[:args.batch_size].to(device)
    car_truck_lbls = torch.cat(car_truck_lbls)[:args.batch_size].to(device)

    # Normalization params
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)

    # Wrapper expects [0,1] images and normalizes internally
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.mean = mean
            self.std = std
        def forward(self, x):
            x_norm = (x - self.mean) / self.std
            out = self.m(x_norm)
            return out[0] if isinstance(out, tuple) else out

    wrapper = Wrapper(model)

    # Denormalize to [0,1] — wrapper handles re-normalization internally
    car_truck_imgs_01 = torch.clamp(car_truck_imgs * std + mean, 0.0, 1.0)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    print('Running PGD-100 evaluation on cars/trucks')
    for eps in epsilons:
        alpha = max(eps / 10, 0.001) if eps > 0 else 0.0
        adv, _ = pgd_attack(wrapper, car_truck_imgs_01, car_truck_lbls,
                             epsilon=eps, alpha=alpha, steps=100,
                             device=device, clip_min=None, clip_max=None, random_start=True)
        adv = torch.clamp(adv, 0.0, 1.0)
        with torch.no_grad():
            preds = wrapper(adv).max(1)[1]
            acc = (preds == car_truck_lbls).float().mean().item() * 100.0
        print(f'ε={eps:.2f} -> PGD-100 accuracy: {acc:.2f}%')

if __name__ == '__main__':
    main()
