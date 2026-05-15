import torch
from torchvision import models
import os

checkpoint_path = 'phase1_training/checkpoints/shaperesnet50_best_v2.pth'
state = torch.load(checkpoint_path, map_location='cpu')

if 'state_dict' in state:
    state = state['state_dict']
elif 'model' in state:
    state = state['model']

model = models.resnet50(weights=None)
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state.keys())

print(f"Total model keys: {len(model_keys)}")
print(f"Total ckpt keys: {len(ckpt_keys)}")

missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f"Direct missing: {len(missing)}")
print(f"Direct unexpected: {len(unexpected)}")

# Check with module. prefix
ckpt_keys_clean = {k[7:] if k.startswith('module.') else k for k in ckpt_keys}
missing_clean = model_keys - ckpt_keys_clean
unexpected_clean = ckpt_keys_clean - model_keys

print(f"Clean missing: {len(missing_clean)}")
print(f"Clean unexpected: {len(unexpected_clean)}")

if len(missing_clean) > 0:
    print("Sample missing keys:", list(missing_clean)[:5])
if len(unexpected_clean) > 0:
    print("Sample unexpected keys:", list(unexpected_clean)[:5])
