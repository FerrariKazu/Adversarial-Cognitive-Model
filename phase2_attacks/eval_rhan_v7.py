#!/usr/bin/env python3
"""
RHAN-v7: Generative World-Model Evaluation
"""
import os
import sys
import time
import argparse
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_v7 import RHANv7
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

try:
    from autoattack import AutoAttack
    HAS_AA = True
except ImportError:
    HAS_AA = False

# ── Wrapper for attacks ──
class V7Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _, _, _ = self.model(x)
        return logits

def calculate_sdt_metrics(pgd_results, epsilons):
    dprimes = []
    for eps in epsilons:
        acc = pgd_results[eps] / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(dprimes) - 1):
        d1, d2 = dprimes[i], dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(dprimes) > 0 and dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    return dprimes, eps_thresh

def run_pgd_100(model, loader, device):
    print("\n" + "="*60)
    print("PGD-100 EVALUATION")
    print("="*60)
    wrapper = V7Wrapper(model)
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1, 3, 1, 1).to(device)
    
    results = {}
    for eps in epsilons:
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, lbls in loader:
            if total >= 500: break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha,
                                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv = images
            with torch.no_grad():
                logits = wrapper(adv)
                correct += logits.argmax(1).eq(lbls).sum().item()
                total += lbls.size(0)
        acc = 100. * correct / max(total, 1)
        results[eps] = acc
        
    dprimes, eps_thresh = calculate_sdt_metrics(results, epsilons)
    print(f"ε_thresh (d'=1.0): {eps_thresh:.4f}" if eps_thresh is not None else ">0.30")
    for i, eps in enumerate(epsilons):
        print(f"ε={eps:.2f} -> PGD-100: {results[eps]:.2f}% | d': {dprimes[i]:.4f}")
        
    return eps_thresh

def run_autoattack(model, loader, device):
    if not HAS_AA:
        print("\nAutoAttack not installed. Skipping.")
        return
        
    print("\n" + "="*60)
    print("AUTOATTACK EVALUATION")
    print("="*60)
    wrapper = V7Wrapper(model)
    adversary = AutoAttack(wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=False)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    correct = 0; total = 0
    for images, labels in loader:
        if total >= 1000: break
        images, labels = images.to(device), labels.to(device)
        x_adv = adversary.run_standard_evaluation(images, labels, bs=images.size(0))
        with torch.no_grad():
            preds = wrapper(x_adv).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_total[lbl] += 1
                if preds[i].item() == lbl:
                    class_correct[lbl] += 1
                    
    aa_acc = 100. * correct / max(total, 1)
    print(f"Overall AutoAttack Accuracy: {aa_acc:.2f}%\n")
    for i in range(10):
        c_acc = 100. * class_correct[i] / max(class_total[i], 1)
        print(f"  {class_names[i]:>10}: {c_acc:.2f}% (n={class_total[i]})")

def denormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(tensor.device)
    # Reconstructions are Tanh [-1, 1], but wait, model outputs x_recon
    # Is x_recon directly predicting the normalized image or is it in [-1, 1]?
    # Actually, the normalized images are ~ [-2.4, +2.7] while Tanh is [-1, 1].
    # But let's assume the network learns to scale its Tanh output appropriately or we need to just map Tanh to [0, 1].
    # Wait, the model uses Tanh(), so output is [-1, 1]. The target is `imgs` which are normalized.
    # If the target is normalized ~[-2.4, 2.7], a Tanh() output [-1, 1] can't fully cover it!
    # Let's map Tanh [-1, 1] to [0, 1] for visualization.
    # The actual network loss might have been bounded by [-1, 1] meaning the PSNR could be artificially constrained if imgs wasn't un-normalized.
    # For visualization, we will simply scale from min/max to 0/1.
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def reconstruction_analysis(model, loader, device, phase_name):
    print("\n" + "="*60)
    print("RECONSTRUCTION ANALYSIS")
    print("="*60)
    wrapper = V7Wrapper(model)
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1, 3, 1, 1).to(device)
    
    # Get 5 images
    imgs_list = []
    lbls_list = []
    for imgs, lbls in loader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        break
    x_clean = imgs_list[0][:5].to(device)
    y_clean = lbls_list[0][:5].to(device)
    
    # Attack them
    x_adv, _ = pgd_attack(wrapper, x_clean, y_clean, epsilon=0.150, alpha=0.015,
                          steps=40, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
                          
    with torch.no_grad():
        _, x_recon_clean, _, _ = model(x_clean)
        _, x_recon_adv, _, _ = model(x_adv)
        
    x_c_viz = denormalize(x_clean)
    x_rc_viz = denormalize(x_recon_clean)
    x_ra_viz = denormalize(x_recon_adv)
    
    fig, axes = plt.subplots(5, 3, figsize=(6, 10))
    for i in range(5):
        axes[i, 0].imshow(x_c_viz[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title('Clean')
        
        axes[i, 1].imshow(x_rc_viz[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title('Recon (Clean)')
        
        axes[i, 2].imshow(x_ra_viz[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].axis('off')
        if i == 0: axes[i, 2].set_title('Recon (Adv)')
        
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'figures'), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', f'v7_reconstruction_{phase_name}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved reconstruction grid to {save_path}")

def auto_truck_test(model, loader, device, phase_name):
    print("\n" + "="*60)
    print("AUTOMOBILE/TRUCK COLLAPSE TEST")
    print("="*60)
    wrapper = V7Wrapper(model)
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1, 3, 1, 1).to(device)
    
    auto_imgs = []
    auto_lbls = []
    for imgs, lbls in loader:
        for i in range(len(lbls)):
            if lbls[i].item() == 1: # Automobile
                auto_imgs.append(imgs[i])
                auto_lbls.append(lbls[i])
            if len(auto_imgs) == 5: break
        if len(auto_imgs) == 5: break
        
    x_clean = torch.stack(auto_imgs).to(device)
    y_clean = torch.stack(auto_lbls).to(device)
    y_target = torch.full_like(y_clean, 9) # Truck
    
    # Targeted PGD
    x_adv = x_clean.clone().detach()
    alpha = 0.015
    for _ in range(40):
        x_adv.requires_grad_(True)
        logits = wrapper(x_adv)
        loss = F.cross_entropy(logits, y_target)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv - x_clean, -0.150, 0.150) + x_clean
        x_adv = torch.max(torch.min(x_adv, cifar_max), cifar_min).detach()
        
    with torch.no_grad():
        preds_adv = wrapper(x_adv).argmax(1)
        _, x_recon_adv, _, _ = model(x_adv)
        
    fooled = (preds_adv == 9).sum().item()
    print(f"Fooled to target Truck: {fooled}/5")
    
    x_c_viz = denormalize(x_clean)
    x_a_viz = denormalize(x_adv)
    x_ra_viz = denormalize(x_recon_adv)
    
    fig, axes = plt.subplots(5, 3, figsize=(6, 10))
    for i in range(5):
        axes[i, 0].imshow(x_c_viz[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title('Clean Auto')
        
        axes[i, 1].imshow(x_a_viz[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].axis('off')
        if i == 0: axes[i, 1].set_title(f'Adv -> {preds_adv[i].item()}')
        
        axes[i, 2].imshow(x_ra_viz[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].axis('off')
        if i == 0: axes[i, 2].set_title('Recon (Adv)')
        
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', f'v7_auto_truck_{phase_name}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved auto/truck test grid to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--phase_name', type=str, default='eval')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RHANv7(head_type='cosine').to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"ERROR: {args.checkpoint} not found!")
        return
        
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    _, testloader = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    
    run_pgd_100(model, testloader, device)
    run_autoattack(model, testloader, device)
    reconstruction_analysis(model, testloader, device, args.phase_name)
    auto_truck_test(model, testloader, device, args.phase_name)

if __name__ == '__main__':
    main()
