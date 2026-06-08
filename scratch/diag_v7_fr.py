#!/usr/bin/env python3
"""
Diagnostic: Check why FR loss is exactly zero in RHAN-v7.
"""
import os
import sys
import copy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_v7 import RHANv7
from phase1_training.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, 'rhan_v7_best.pth')
model = RHANv7(head_type='cosine').to(device)
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()

# Re-initialize perceptual_critic (same as training script)
model.perceptual_critic = copy.deepcopy(model.stem_low).to(device)
for p in model.perceptual_critic.parameters():
    p.requires_grad = False
model.perceptual_critic.eval()

# Get a batch of data
_, testloader = get_dataloaders(batch_size=8, num_workers=0, model_name='resnet')
imgs, lbls = next(iter(testloader))
imgs = imgs.to(device)

# Create a slightly perturbed version (simulating adversarial)
x_adv = imgs + 0.062 * torch.sign(torch.randn_like(imgs))
x_adv = x_adv.detach()

# Forward pass
with torch.no_grad():
    logits, x_recon, mu, log_var = model(x_adv)

print(f"\nInput shape:       {imgs.shape}")
print(f"Adv shape:         {x_adv.shape}")
print(f"Recon shape:      {x_recon.shape}")
print(f"Recon range:      [{x_recon.min():.4f}, {x_recon.max():.4f}]")
print(f"Input range:      [{imgs.min():.4f}, {imgs.max():.4f}]")

# Check if recon equals input
print(f"\nRecon == Input?   {torch.equal(x_recon, imgs)}")
print(f"Recon ≈ Input?    {(x_recon - imgs).abs().max().item():.6f}")

# Check perceptual critic outputs
x_low_orig, _ = model.separate_frequencies(imgs)
x_low_recon, _ = model.separate_frequencies(x_recon)

print(f"\nx_low_orig shape:  {x_low_orig.shape}")
print(f"x_low_recon shape: {x_low_recon.shape}")
print(f"x_low_orig range:  [{x_low_orig.min():.4f}, {x_low_orig.max():.4f}]")
print(f"x_low_recon range: [{x_low_recon.min():.4f}, {x_low_recon.max():.4f}]")

feats_orig  = model.perceptual_critic(x_low_orig)
feats_recon = model.perceptual_critic(x_low_recon)

print(f"\nfeats_orig shape:  {feats_orig.shape}")
print(f"feats_recon shape: {feats_recon.shape}")
print(f"feats_orig range:  [{feats_orig.min():.4f}, {feats_orig.max():.4f}]")
print(f"feats_recon range: [{feats_recon.min():.4f}, {feats_recon.max():.4f}]")
print(f"feats_orig mean:   {feats_orig.mean():.6f}")
print(f"feats_recon mean:  {feats_recon.mean():.6f}")
print(f"feats_orig std:    {feats_orig.std():.6f}")
print(f"feats_recon std:   {feats_recon.std():.6f}")

# Check if features are identical
print(f"\nFeatures identical? {torch.equal(feats_orig, feats_recon)}")
print(f"Feature diff max:   {(feats_orig - feats_recon).abs().max().item():.8f}")
print(f"Feature diff mean:  {(feats_orig - feats_recon).abs().mean().item():.8f}")

# Normalize and compute MSE
feats_orig_n  = F.normalize(feats_orig.flatten(1), dim=1)
feats_recon_n = F.normalize(feats_recon.flatten(1), dim=1)

print(f"\nNormalized orig norm:  {feats_orig_n[0].norm().item():.6f}")
print(f"Normalized recon norm: {feats_recon_n[0].norm().item():.6f}")
print(f"Cosine similarity:     {F.cosine_similarity(feats_orig_n, feats_recon_n, dim=1).mean().item():.6f}")
print(f"MSE after normalize:   {F.mse_loss(feats_recon_n, feats_orig_n).item():.8f}")

# Check perceptual critic BatchNorm stats
print(f"\n--- Perceptual Critic BatchNorm Stats ---")
for name, module in model.perceptual_critic.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        print(f"  {name}:")
        print(f"    running_mean: [{module.running_mean.min():.4f}, {module.running_mean.max():.4f}]")
        print(f"    running_var:  [{module.running_var.min():.4f}, {module.running_var.max():.4f}]")
        print(f"    num_batches_tracked: {module.num_batches_tracked}")
