import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import argparse

from model import CIFARResNet
from model_vit import CIFARViT
from dataset import get_dataloaders, CLASSES
from dataset_vit import get_dataloaders_vit

def main():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 Models')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'efficientnet'],
                        help='Model architecture to evaluate (default: resnet)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (default: checkpoints/best.pth or vit_small_best.pth)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model Selection
    if args.model == 'resnet':
        model = CIFARResNet().to(device)
        default_checkpoint = 'checkpoints/best.pth'
        _, testloader = get_dataloaders(batch_size=128, num_workers=4, data_dir='../data')
    elif args.model == 'vit':
        model = CIFARViT().to(device)
        default_checkpoint = 'checkpoints/vit_small_best.pth'
        _, testloader = get_dataloaders_vit(batch_size=64, num_workers=4, data_dir='../data')
    elif args.model == 'efficientnet':
        from model_efficientnet import CIFAREfficientNet
        model = CIFAREfficientNet().to(device)
        default_checkpoint = 'checkpoints/efficientnet_best.pth'
        _, testloader = get_dataloaders_vit(batch_size=32, num_workers=4, data_dir='../data')
    
    checkpoint_path = args.checkpoint if args.checkpoint else default_checkpoint
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Evaluating with random weights.")
        
    model.eval()
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    # -------------------------------------------------------------------------
    # Clean Accuracy & Confusion Matrix Data Gathering
    # -------------------------------------------------------------------------
    all_preds = []
    all_targets = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"Evaluating {args.model}...")
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Global Accuracy
    correct = (all_preds == all_targets).sum()
    overall_acc = 100.0 * correct / len(all_targets)
    print(f"\nOverall {args.model.upper()} Test Accuracy: {overall_acc:.2f}%\n")
    
    # Per-Class Accuracy
    print("Per-Class Accuracy:")
    print("-" * 25)
    for i in range(10):
        class_mask = (all_targets == i)
        class_correct = (all_preds[class_mask] == all_targets[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc = 100.0 * class_correct / class_total
        print(f"{CLASSES[i]:>12}: {class_acc:.2f}%")
        
    # Plotting Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'CIFAR-10 Clean Accuracy Confusion Matrix ({args.model.upper()})')
    plt.tight_layout()
    cm_path = f'confusion_matrix_{args.model}.png'
    plt.savefig(cm_path)
    print(f"\nSaved {cm_path}")
    
    # VRAM Usage
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak VRAM Usage: {peak_vram:.2f} MB")

if __name__ == '__main__':
    main()
