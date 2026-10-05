import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_training.model_rhan_v9 import RHANv9
from phase1_training.train_sail import get_dataloaders
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RHANv9(head_type='cosine').to(device)

ckpt = torch.load('checkpoints/rhan_trades_phase_c_final.pth', map_location=device)
model_state = model.state_dict()
filtered_state = {k: v for k, v in ckpt.items() if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(filtered_state, strict=False)

_, testloader_raw = get_dataloaders(batch_size=64, num_workers=0, model_name='resnet')
testloader = DataLoader(testloader_raw.dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Test 1: Standard eval
model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, lbls in testloader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        correct += logits.argmax(1).eq(lbls).sum().item()
        total += lbls.size(0)
print(f"Standard Eval Acc: {100. * correct / total:.2f}%")

# Test 2: Bypassed eval (disable predictive coder by setting error_scale to 0)
model.predictive_coder.error_scale.data.fill_(0.0)

correct = total = 0
with torch.no_grad():
    for imgs, lbls in testloader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        correct += logits.argmax(1).eq(lbls).sum().item()
        total += lbls.size(0)
print(f"Bypassed Eval Acc: {100. * correct / total:.2f}%")
