import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from model import CIFARResNet
from dataset import get_dataloaders, CLASSES

def main():
    # -------------------------------------------------------------------------
    # Model Loading and VRAM profiling
    # -------------------------------------------------------------------------
    # 1. WHAT: Initializes the model, loads saved weights, tracks parameters.
    # 2. WHY: We need to restore the exact state of our best model to analyze it.
    # 3. OBSERVE: Prints total parameter count (~11 Million) and initial VRAM.
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CIFARResNet().to(device)
    checkpoint_path = 'checkpoints/best.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Evaluating with random weights.")
        
    model.eval()
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    _, testloader = get_dataloaders(batch_size=128, num_workers=4, data_dir='../data')
    
    # -------------------------------------------------------------------------
    # Clean Accuracy & Confusion Matrix Data Gathering
    # -------------------------------------------------------------------------
    # 1. WHAT: Iterates over the test set, collecting predictions and true labels.
    # 2. WHY: Required to calculate per-class accuracy and generate the matrix.
    # 3. OBSERVE: VRAM usage will peak as batches are moved to the GPU.
    # -------------------------------------------------------------------------
    all_preds = []
    all_targets = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
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
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%\n")
    
    # -------------------------------------------------------------------------
    # Per-Class Accuracy
    # -------------------------------------------------------------------------
    # 1. WHAT: Calculates accuracy for each individual CIFAR-10 category.
    # 2. WHY: A model might have 90% overall accuracy but completely fail on "bird"
    #         vs "airplane". Knowing per-class vulnerabilities is crucial for SDT.
    # 3. OBSERVE: A printed table showing accuracy per class.
    # -------------------------------------------------------------------------
    print("Per-Class Accuracy:")
    print("-" * 25)
    for i in range(10):
        class_mask = (all_targets == i)
        class_correct = (all_preds[class_mask] == all_targets[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc = 100.0 * class_correct / class_total
        print(f"{CLASSES[i]:>12}: {class_acc:.2f}%")
        
    # -------------------------------------------------------------------------
    # Plotting Confusion Matrix
    # -------------------------------------------------------------------------
    # 1. WHAT: Visualizing exactly which classes the model confuses.
    # 2. WHY: Essential for adversarial cognitive research. If a dog is perturbed, 
    #         does the model think it's a cat (animal) or a truck (vehicle)?
    # 3. OBSERVE: Creates 'confusion_matrix.png' in the current directory.
    # -------------------------------------------------------------------------
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('CIFAR-10 Clean Accuracy Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nSaved confusion_matrix.png")
    
    # VRAM Usage
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak VRAM Usage: {peak_vram:.2f} MB")

if __name__ == '__main__':
    main()
