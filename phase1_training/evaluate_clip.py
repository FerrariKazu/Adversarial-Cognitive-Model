import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

from model_clip import CIFARClip

# Hardcoded CIFAR-10 classes for formatting the output
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading CLIP ViT-B/32 (Zero-Shot)...")
    model = CIFARClip().to(device)
    
    # Load CIFAR-10 test set with ToTensor() ONLY - CLIP handles its own normalization
    transform = transforms.ToTensor()
    
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    
    print("Evaluating Zero-Shot CLIP...")
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    for inputs, targets in testloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Predict uses zero-shot cosine similarity under the hood
        logits = model.predict(inputs)
        _, predicted = logits.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    overall_acc = 100. * correct / total
    print(f"\nOverall CLIP Zero-Shot Test Accuracy: {overall_acc:.2f}%\n")
    
    print("Per-Class Accuracy:")
    print("-" * 25)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    for i in range(10):
        class_mask = (all_targets == i)
        class_correct = (all_preds[class_mask] == all_targets[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc = 100.0 * class_correct / class_total
        print(f"{CLASSES[i]:>12}: {class_acc:.2f}%")

if __name__ == '__main__':
    main()
