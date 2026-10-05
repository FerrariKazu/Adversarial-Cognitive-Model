#!/usr/bin/env python3
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_pretrained import RHANUnifiedSTL10
from model_rhan_stl10_large import RHANLargeSTL10
from train_rhan_video_tdv import get_stl10_dataloaders

def pgd_attack_classification(model, x, y, eps=0.031, steps=100, stl_min=None, stl_max=None):
    model.eval()
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)
    
    alpha = max(eps / 10, 0.001) if eps > 0 else 0.0
    if eps == 0:
        return x
        
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with autocast('cuda'):
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, stl_min, stl_max).detach()
        
    return x_adv

def main():
    parser = argparse.ArgumentParser(description='RHAN STL-10 PGD-100 Evaluation')
    parser.add_argument('--model-size', type=str, default='base', choices=['base', 'large'],
                        help='Model size architecture to instantiate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (defaults to size-specific path)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--samples', type=int, default=512, help='Number of test samples to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model size: {args.model_size}")

    # Load model architecture
    if args.model_size == 'large':
        model = RHANLargeSTL10().to(device)
        default_ckpt = '../checkpoints/rhan_stl10_large_video_tdv.pth'
    else:
        model = RHANUnifiedSTL10().to(device)
        default_ckpt = '../checkpoints/rhan_stl10_base_video_tdv.pth'

    ckpt_rel = args.checkpoint if args.checkpoint is not None else default_ckpt
    script_dir = os.path.dirname(__file__)
    ckpt_path = os.path.abspath(os.path.join(script_dir, ckpt_rel))
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded resume checkpoint (epoch {ckpt['epoch']}): {ckpt_path}")
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            print(f"Loaded checkpoint state_dict: {ckpt_path}")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded raw state dict checkpoint: {ckpt_path}")
    else:
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get dataloaders
    _, testloader, stl_min, stl_max = get_stl10_dataloaders(batch_size=args.batch_size)
    stl_min = stl_min.to(device)
    stl_max = stl_max.to(device)

    # Get subset of test samples
    all_imgs = []
    all_lbls = []
    for imgs, lbls in testloader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
        if sum(x.size(0) for x in all_imgs) >= args.samples:
            break
            
    x_test = torch.cat(all_imgs, dim=0)[:args.samples].to(device)
    y_test = torch.cat(all_lbls, dim=0)[:args.samples].to(device)

    epsilons = [0.0, 0.015, 0.031, 0.062, 0.094, 0.125, 0.150]
    print(f"\nEvaluating PGD-100 on {x_test.size(0)} samples across epsilons...")
    print("=" * 60)

    # Find clean accuracy first
    with torch.no_grad():
        with autocast('cuda'):
            clean_logits = model(x_test)
        clean_preds = clean_logits.argmax(dim=1)
        clean_acc = 100.0 * clean_preds.eq(y_test).sum().item() / y_test.size(0)
    print(f"Clean accuracy: {clean_acc:.2f}%")
    print("-" * 60)

    epsthresh = None

    for eps in epsilons:
        t0 = time.time()
        # Perform attack in batches to prevent memory limits
        adv_imgs_list = []
        for i in range(0, x_test.size(0), args.batch_size):
            x_batch = x_test[i:i+args.batch_size]
            y_batch = y_test[i:i+args.batch_size]
            x_adv = pgd_attack_classification(model, x_batch, y_batch, eps=eps, steps=100, stl_min=stl_min, stl_max=stl_max)
            adv_imgs_list.append(x_adv)
        
        x_adv_all = torch.cat(adv_imgs_list, dim=0)
        
        with torch.no_grad():
            with autocast('cuda'):
                logits = model(x_adv_all)
            preds = logits.argmax(dim=1)
            acc = 100.0 * preds.eq(y_test).sum().item() / y_test.size(0)
            
        print(f"ε = {eps:.3f} | PGD-100 Accuracy: {acc:.2f}% | Time: {time.time()-t0:.1f}s")
        
        # Estimate threshold where accuracy falls below 50% of clean accuracy
        if epsthresh is None and acc <= (clean_acc / 2.0):
            epsthresh = eps

    print("=" * 60)
    if epsthresh is not None:
        print(f"Estimated ε_thresh (accuracy falls below 50% of clean): ~{epsthresh:.3f}")
    else:
        print(f"Estimated ε_thresh: > {epsilons[-1]:.3f}")

if __name__ == '__main__':
    main()
