"""
Unified adversarial generation script for all models.
Usage: python generate_adv_all_models.py --model [resnet|vit]
"""

import sys
import os
import yaml
import numpy as np
import torch
import argparse
from tqdm import tqdm
import psutil

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")
    print("Close other applications before continuing.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model import CIFARResNet
from phase1_training.model_vit import CIFARViT
from phase1_training.model_efficientnet import CIFAREfficientNet
from phase1_training.dataset import get_dataloaders
from phase1_training.dataset_vit import get_dataloaders_vit
from fgsm import fgsm_attack
from pgd import pgd_attack
from cw import cw_attack

import os

MODELS = {
    'resnet': {
        'ckpt': os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'best.pth'),
        'class': CIFARResNet,
        'out': os.path.join(os.path.dirname(__file__), 'adv_images', 'resnet'),
        'input_size': 32,
        'loader_fn': get_dataloaders
    },
    'vit': {
        'ckpt': os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth'),
        'class': CIFARViT,
        'out': os.path.join(os.path.dirname(__file__), 'adv_images', 'vit'),
        'input_size': 224,
        'loader_fn': get_dataloaders_vit
    },
    'efficientnet': {
        'ckpt': os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'efficientnet_best.pth'),
        'class': CIFAREfficientNet,
        'out': os.path.join(os.path.dirname(__file__), 'adv_images', 'efficientnet'),
        'input_size': 224,
        'loader_fn': get_dataloaders_vit # Uses same resize as ViT
    }
}

def generate_for_attack(model, testloader, attack_fn, device, attack_name, save_dir, cifar_min, cifar_max, **attack_kwargs):
    save_path = os.path.join(save_dir, f"{attack_name}_images.npy")
    
    # --- Resume capability: skip if file already exists and is valid ---
    if os.path.exists(save_path):
        try:
            existing = np.load(save_path, mmap_mode='r')
            n_samples = len(testloader.dataset)
            _, _, h, w = next(iter(testloader))[0].shape
            if existing.shape == (n_samples, 3, h, w) and existing.dtype == np.float32:
                print(f"  ⏭ {attack_name}: {save_path} already exists with correct shape {existing.shape} — skipping")
                del existing
                return save_path
            else:
                print(f"  ⚠ {attack_name}: {save_path} exists but shape {existing.shape} doesn't match expected ({n_samples}, 3, {h}, {w}) — regenerating")
                del existing
        except Exception as e:
            print(f"  ⚠ {attack_name}: {save_path} exists but is corrupt ({e}) — regenerating")
    
    # Pre-allocate memmap array to avoid OOM (10000 images * 3 * 224 * 224 * 4 bytes = ~6GB)
    n_samples = len(testloader.dataset)
    _, _, h, w = next(iter(testloader))[0].shape
    mmap_arr = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=(n_samples, 3, h, w))

    pbar = tqdm(testloader, desc=f"  {attack_name}", leave=True)
    start_idx = 0
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        adv_images, _ = attack_fn(model, images, labels, device=device, **attack_kwargs)
        
        # Clamp to valid normalized pixel range before saving
        adv_images = torch.max(torch.min(adv_images, cifar_max), cifar_min)
        
        batch_size = images.size(0)
        mmap_arr[start_idx:start_idx+batch_size] = adv_images.cpu().numpy()
        start_idx += batch_size
        
        # Memory cleanup
        del images, labels, adv_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Flush to disk
    mmap_arr.flush()
    return save_path

def verify_file(filepath):
    data = np.load(filepath)
    print(f"    ✓ {os.path.basename(filepath)}: shape={data.shape}, dtype={data.dtype}, range=[{data.min():.4f}, {data.max():.4f}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'vit', 'efficientnet'], required=True)
    args = parser.parse_args()

    cfg = MODELS[args.model]
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
    with open(config_path, 'r') as f:
        attack_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = cfg['class']().to(device)
    if cfg['ckpt'] is not None:
        model.load_state_dict(torch.load(cfg['ckpt'], map_location=device))
        print(f"Loaded {args.model} checkpoint from: {cfg['ckpt']}")
    else:
        print(f"Loaded {args.model} with pretrained weights")
    model.eval()

    # Load data
    batch_size = 16 if args.model in ['vit', 'efficientnet'] else 32
    _, testloader = cfg['loader_fn'](batch_size=batch_size, num_workers=2)
    print(f"Using batch size: {batch_size}")
    
    save_dir = cfg['out']
    os.makedirs(save_dir, exist_ok=True)

    epsilons = attack_config['epsilons']
    pgd_steps = attack_config.get('pgd_steps', 20)
    pgd_alpha = attack_config.get('pgd_alpha', 0.01)

    # CIFAR normalization stats for clamping
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.5141,  2.6078,  2.7537]).view(1, 3, 1, 1).to(device)

    saved_files = []

    labels_path = os.path.join(save_dir, 'labels.npy')
    if os.path.exists(labels_path):
        print(f"\n⏭ labels.npy already exists — skipping")
        all_labels = np.load(labels_path)
    else:
        print("\nSaving true labels...")
        all_labels = []
        for _, labels in testloader:
            all_labels.append(labels.numpy())
        all_labels = np.concatenate(all_labels, axis=0)
        np.save(labels_path, all_labels)
        print(f"    ✓ labels.npy saved.")

    # print(f"\nFGSM Attack Generation")
    # for eps in epsilons:
    #     name = f"fgsm_eps{eps:.2f}"
    #     path = generate_for_attack(model, testloader, fgsm_attack, device, name, save_dir, cifar_min, cifar_max, epsilon=eps)
    #     saved_files.append(path)

    print(f"\nPGD Attack Generation")
    for eps in epsilons:
        name = f"pgd_eps{eps:.2f}"
        path = generate_for_attack(model, testloader, pgd_attack, device, name, save_dir, cifar_min, cifar_max, epsilon=eps, alpha=pgd_alpha, steps=pgd_steps)
        saved_files.append(path)

    # if args.model != 'vit':
    #     print(f"\nC&W L2 Attack Generation")
    #     path = generate_for_attack(model, testloader, cw_attack, device, "cw", save_dir, cifar_min, cifar_max)
    #     saved_files.append(path)

    print(f"\nVerification:")
    verify_file(labels_path)
    for fp in saved_files:
        verify_file(fp)

if __name__ == '__main__':
    main()
