#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase1_training'))

from phase1_training.model_rhan_v9 import RHANv9
from phase1_training.dataset import get_dataloaders
from autoattack import AutoAttack

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../checkpoints/rhan_v9_sail.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    model = RHANv9(head_type='cosine').to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # Logit-only wrapper
    class W(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    wrapper = W(model)

    # Get data
    _, testloader_raw = get_dataloaders(batch_size=16, num_workers=0, model_name='resnet')
    
    # Load 100 test images
    imgs_list, lbls_list = [], []
    for imgs, lbls in testloader_raw:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        if sum(x.size(0) for x in imgs_list) >= 100:
            break
            
    x_test = torch.cat(imgs_list, dim=0)[:100].to(device)
    y_test = torch.cat(lbls_list, dim=0)[:100].to(device)
    
    print(f"Evaluating AutoAttack on {x_test.size(0)} images...")

    adversary = AutoAttack(wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=16)

    with torch.no_grad():
        logits = wrapper(x_adv)
        preds = logits.argmax(1)
        correct = (preds == y_test).sum().item()
        
    aa_acc = correct / x_test.size(0)

    print("=" * 60)
    print(f"AutoAttack (eps=0.031) Accuracy: {aa_acc*100:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
