import torch
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))
from model_vit import CIFARViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
model = CIFARViT().to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth')
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Load Labels
labels_path = os.path.join(os.path.dirname(__file__), 'adv_images', 'vit', 'labels.npy')
labels = np.load(labels_path)
labels_tensor = torch.tensor(labels, device=device)

epsilons = [0.10, 0.20, 0.30]

print("Evaluating remaining PGD epsilons from pre-generated .npy files...")
for eps in epsilons:
    images_path = os.path.join(os.path.dirname(__file__), 'adv_images', 'vit', f"pgd_eps{eps:.2f}_images.npy")
    # Load with memmap
    adv_images_np = np.load(images_path, mmap_mode='r')
    
    correct = 0
    total = len(labels)
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_images = torch.tensor(adv_images_np[i:i+batch_size], device=device)
            batch_labels = labels_tensor[i:i+batch_size]
            
            outputs = model(batch_images)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            
    acc = 100.0 * correct / total
    print(f"PGD | {eps:.2f} | {acc:.2f}%")
