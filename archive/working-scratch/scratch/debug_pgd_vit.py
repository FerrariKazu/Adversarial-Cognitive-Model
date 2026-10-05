import torch
import torch.nn as nn
import sys
sys.path.append('phase1_training')
from model_vit import CIFARViT
from dataset_vit import get_dataloaders_vit
import yaml
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFARViT().to(device)
model.load_state_dict(torch.load('phase1_training/checkpoints/vit_small_best.pth', map_location=device))
model.eval()

_, testloader = get_dataloaders_vit(batch_size=32)
images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)

epsilon = 0.30
alpha = 0.01
steps = 20

# CIFAR normalization stats for clamping
cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
cifar_max = torch.tensor([ 2.5141,  2.6078,  2.7537]).view(1, 3, 1, 1).to(device)

adv_images = images.clone().detach()
adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
adv_images = torch.max(torch.min(adv_images, cifar_max), cifar_min).detach()

print(f"Step | Loss | Grad Norm | Acc")
for i in range(steps):
    adv_images.requires_grad_(True)
    outputs = model(adv_images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    preds = outputs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    
    model.zero_grad()
    loss.backward()
    
    grad_norm = adv_images.grad.norm().item()
    print(f"{i:4d} | {loss.item():.4f} | {grad_norm:.4e} | {acc:.2%}")
    
    adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
    delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
    adv_images = (images + delta).detach()
    adv_images = torch.max(torch.min(adv_images, cifar_max), cifar_min)

outputs = model(adv_images)
preds = outputs.argmax(dim=1)
final_acc = (preds == labels).float().mean().item()
print(f"Final Acc: {final_acc:.2%}")
