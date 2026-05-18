import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model_rhan import RHAN
from phase1_training.dataset import get_dataloaders, CLASSES
from phase2_attacks.pgd import pgd_attack

# Configuration
OUTPUT_DIR_CLEAN = os.path.join(os.path.dirname(__file__), 'figures', 'rhan_clean', 'feedback')
OUTPUT_DIR_ADV = os.path.join(os.path.dirname(__file__), 'figures', 'rhan_adv', 'feedback')
OUTPUT_DIR_COMBINED = os.path.join(os.path.dirname(__file__), 'figures', 'combined')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR Normalization Stats (for denormalization)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

def denormalize(img_tensor):
    """(3, H, W) normalized -> (H, W, 3) in [0, 1]"""
    img = img_tensor.cpu() * STD + MEAN
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def forward_with_gates(model, x):
    """Run forward pass and return logits along with recurrent feedback gates."""
    # Stage 1: Conv Stem
    stem_features = model.stem(x)  # (B, 512, 8, 8)
    
    # Stage 2: Tokeniser
    tokens = model.tokeniser(stem_features)  # (B, 65, 512)
    
    # Stage 3: Transformer
    attended = model.transformer(tokens)  # (B, 65, 512)
    
    # Stage 4: Recurrent Feedback
    gates = []
    current = attended
    for t in range(model.feedback.num_recurrent_steps):
        cls_token = current[:, :1, :]
        spatial = model.feedback.tokens_to_spatial(current)
        feedback = model.feedback.feedback_conv(spatial)
        g = model.feedback.gate(feedback)  # (B, 512, 8, 8)
        gates.append(g.detach().cpu())
        
        modulated = stem_features + g * feedback
        modulated_tokens = model.feedback.spatial_to_tokens(modulated, cls_token)
        current = model.transformer(modulated_tokens)
        
    refined = current
    cls_output = refined[:, 0, :]
    logits = model.head(cls_output)
    
    return logits, gates

def get_mean_gate_map(gates, step_idx=0):
    """Extract mean gate map across all channels for a given recurrent step."""
    # gates[step_idx]: (1, 512, 8, 8)
    g = gates[step_idx][0]  # (512, 8, 8)
    # Average across channels
    return g.mean(dim=0)  # (8, 8)

def compute_entropy(gate_map):
    """Computes Shannon entropy of the gating map distribution."""
    p = gate_map.flatten()
    p = p / (p.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
    return entropy

def generate_feedback_visuals(model_name, model, testloader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(DEVICE)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(DEVICE)
    
    # Select 1 representative image per class
    images_by_class = {}
    labels_by_class = {}
    
    for images, labels in testloader:
        for img, lbl in zip(images, labels):
            c = lbl.item()
            if c not in images_by_class:
                images_by_class[c] = img
                labels_by_class[c] = lbl
            if len(images_by_class) == 10:
                break
        if len(images_by_class) == 10:
            break
            
    print(f"\nGenerating Feedback Attention Maps for {model_name.upper()}...")
    
    # For entropy statistics
    entropy_clean = []
    entropy_adv = []
    
    for c in range(10):
        img = images_by_class[c].to(DEVICE).unsqueeze(0)
        lbl = labels_by_class[c].to(DEVICE).unsqueeze(0)
        
        # 1. Generate adversarial image under PGD-100 epsilon=0.10
        a = max(0.10 / 10, 0.001)
        adv_img, _ = pgd_attack(
            model, img, lbl,
            epsilon=0.10, alpha=a, steps=100,
            device=DEVICE, clip_min=cifar_min, clip_max=cifar_max,
        )
        
        # 2. Extract gates
        with torch.no_grad():
            _, gates_clean = forward_with_gates(model, img)
            _, gates_adv = forward_with_gates(model, adv_img)
            
        map_clean = get_mean_gate_map(gates_clean, step_idx=0)
        map_adv = get_mean_gate_map(gates_adv, step_idx=0)
        
        entropy_clean.append(compute_entropy(map_clean))
        entropy_adv.append(compute_entropy(map_adv))
        
        # 3. Plotting
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=150)
        
        # Panel 1: Clean Image
        axes[0].imshow(denormalize(img[0]))
        axes[0].set_title("Clean Input")
        
        # Panel 2: Feedback Gate (Clean)
        m_c = map_clean.unsqueeze(0).unsqueeze(0)
        m_c = F.interpolate(m_c, size=(32, 32), mode='bilinear', align_corners=False)[0, 0]
        m_c = (m_c - m_c.min()) / (m_c.max() - m_c.min() + 1e-12)
        axes[1].imshow(denormalize(img[0]))
        axes[1].imshow(m_c.numpy(), cmap='viridis', alpha=0.6)
        axes[1].set_title(f"Clean Feedback (H={entropy_clean[-1]:.2f})")
        
        # Panel 3: Adv Image
        axes[2].imshow(denormalize(adv_img[0]))
        axes[2].set_title("Adversarial Input")
        
        # Panel 4: Feedback Gate (Adv)
        m_a = map_adv.unsqueeze(0).unsqueeze(0)
        m_a = F.interpolate(m_a, size=(32, 32), mode='bilinear', align_corners=False)[0, 0]
        m_a = (m_a - m_a.min()) / (m_a.max() - m_a.min() + 1e-12)
        axes[3].imshow(denormalize(adv_img[0]))
        axes[3].imshow(m_a.numpy(), cmap='viridis', alpha=0.6)
        axes[3].set_title(f"Adv Feedback (H={entropy_adv[-1]:.2f})")
        
        # Panel 5: Feedback Gating Shift (Adv - Clean)
        diff = map_adv - map_clean
        m_d = diff.unsqueeze(0).unsqueeze(0)
        m_d = F.interpolate(m_d, size=(32, 32), mode='bilinear', align_corners=False)[0, 0]
        im = axes[4].imshow(m_d.numpy(), cmap='RdBu_r', vmin=-m_d.abs().max(), vmax=m_d.abs().max())
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
        axes[4].set_title("Gating Shift (Adv - Clean)")
        
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
            
        plt.suptitle(f"{model_name.upper()} Gating Dynamics: {CLASSES[c].upper()}", fontsize=14, y=0.98)
        out_path = os.path.join(output_dir, f"feedback_{CLASSES[c]}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved map for {CLASSES[c]} to {out_path}")
        
    # Generate unified entropy comparison plot
    plt.figure(figsize=(6, 5), dpi=150)
    plt.bar(["Clean", "Adversarial"], [np.mean(entropy_clean), np.mean(entropy_adv)], 
            yerr=[np.std(entropy_clean), np.std(entropy_adv)], 
            color=['#3b82f6', '#ef4444'], capsize=5, alpha=0.85, width=0.5)
    plt.title(f"Recurrent Gating Entropy: Clean vs Adv ({model_name.upper()})", fontsize=12)
    plt.ylabel("Mean Gating Entropy (nats)")
    plt.grid(axis='y', alpha=0.3)
    entropy_path = os.path.join(output_dir, "gating_entropy_comparison.png")
    plt.savefig(entropy_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved entropy plot to {entropy_path}")

def main():
    os.makedirs(OUTPUT_DIR_COMBINED, exist_ok=True)
    
    # Load Dataloader
    _, testloader_raw = get_dataloaders(batch_size=256, num_workers=2, model_name='resnet')
    from torch.utils.data import DataLoader
    testloader = DataLoader(testloader_raw.dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # 1. Analyze RHAN-clean
    print("\nLoading RHAN-clean...")
    model_clean = RHAN(num_classes=10, head_type='linear').to(DEVICE)
    ckpt_clean = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'rhan_v2_best.pth')
    model_clean.load_state_dict(torch.load(ckpt_clean, map_location=DEVICE))
    model_clean.eval()
    
    for p in model_clean.parameters():
        p.requires_grad = False
        
    generate_feedback_visuals("rhan-clean", model_clean, testloader, OUTPUT_DIR_CLEAN)
    
    # 2. Analyze RHAN-adv
    print("\nLoading RHAN-adv...")
    model_adv = RHAN(num_classes=10, head_type='linear').to(DEVICE)
    ckpt_adv = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'rhan_adv_best.pth')
    model_adv.load_state_dict(torch.load(ckpt_adv, map_location=DEVICE))
    model_adv.eval()
    
    for p in model_adv.parameters():
        p.requires_grad = False
        
    generate_feedback_visuals("rhan-adv", model_adv, testloader, OUTPUT_DIR_ADV)

if __name__ == '__main__':
    main()
