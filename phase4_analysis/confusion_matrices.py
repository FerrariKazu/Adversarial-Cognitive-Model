import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import psutil

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from phase1_training.dataset import CLASSES

# Configuration
OUTPUT_DIR_COMBINED = os.path.join(os.path.dirname(__file__), 'figures', 'combined')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_predictions(model, images_np, device):
    """Runs inference and returns predicted classes."""
    model.eval()
    preds = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(images_np), batch_size):
            batch = torch.tensor(images_np[i:i+batch_size], device=device)
            logits = model(batch)
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            del batch, logits
            
    return np.array(preds)

def analyze_confusions(cm_diff):
    """Identifies the top 3 off-diagonal increases (new errors)."""
    diff_masked = cm_diff.copy()
    np.fill_diagonal(diff_masked, -1) # Ignore the diagonal (lost accuracy)
    
    # Get top 3 indices
    flat_indices = np.argsort(diff_masked.ravel())[::-1][:3]
    top_3 = []
    for idx in flat_indices:
        r, c = np.unravel_index(idx, cm_diff.shape)
        top_3.append((r, c, cm_diff[r, c]))
    return top_3

def plot_model_suite(model_name, cm_clean, cm_adv, cm_diff):
    """Plots the 3-panel matrix for a single model."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)
    
    # Clean CM
    sns.heatmap(cm_clean, annot=True, fmt='.1f', cmap='Blues', ax=axes[0],
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    axes[0].set_title(f"{model_name.upper()} — Clean CM (%)", fontsize=14)
    
    # Adv CM
    sns.heatmap(cm_adv, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    axes[1].set_title(f"{model_name.upper()} — PGD ε=0.10 CM (%)", fontsize=14)
    
    # Diff CM
    sns.heatmap(cm_diff, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=axes[2],
                xticklabels=CLASSES, yticklabels=CLASSES)
    axes[2].set_title("Difference (Adv - Clean)", fontsize=14)
    
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'figures', model_name, 'confusion')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_confusion_suite.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return out_path

def main():
    os.makedirs(OUTPUT_DIR_COMBINED, exist_ok=True)
    
    target_models = ['resnet', 'vit', 'efficientnet', 'shaperesnet']
    all_top_3 = {}

    for m_name in target_models:
        print(f"\nAnalyzing Confusions for {m_name.upper()}...")
        cfg = MODELS[m_name]
        
        # Load Model
        mclass = cfg['class']
        ckpt_path = cfg.get('ckpt')
        if mclass.__name__ == 'ShapeResNet':
            model = mclass(num_classes=10, weights_path=ckpt_path).to(DEVICE)
        else:
            model = mclass().to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        # Load Data
        img_dir = cfg['out']
        labels = np.load(os.path.join(img_dir, 'labels.npy'))
        clean_imgs = np.load(os.path.join(img_dir, 'pgd_eps0.00_images.npy'), mmap_mode='r')
        adv_imgs = np.load(os.path.join(img_dir, 'pgd_eps0.10_images.npy'), mmap_mode='r')
        
        # Get Predictions
        print("  Running inference (clean)...")
        preds_clean = get_predictions(model, clean_imgs, DEVICE)
        print("  Running inference (adversarial)...")
        preds_adv = get_predictions(model, adv_imgs, DEVICE)
        
        # Compute Matrices (normalized by true class)
        cm_clean = confusion_matrix(labels, preds_clean, labels=np.arange(10), normalize='true') * 100.0
        cm_adv = confusion_matrix(labels, preds_adv, labels=np.arange(10), normalize='true') * 100.0
        cm_diff = cm_adv - cm_clean
        
        # Plot and Analyze
        plot_model_suite(m_name, cm_clean, cm_adv, cm_diff)
        all_top_3[m_name] = analyze_confusions(cm_diff)
        
        del model
        torch.cuda.empty_cache()

    # --- Headline Findings ---
    print("\n" + "="*60)
    print("TOP ADVERSARIAL CONFUSIONS (True → Predicted)")
    print("="*60)
    for m_name, top in all_top_3.items():
        print(f"\n{m_name.upper()}:")
        for r, c, val in top:
            print(f"  {CLASSES[r].upper()} → {CLASSES[c].upper()} : +{val:.1f}% shift")
    print("="*60)

    # --- SCIENTIFIC COMMENTARY ---
    # 1. SEMANTIC VS RANDOM
    #    Semantically similar confusions (e.g., cat → dog, truck → automobile) suggest that 
    #    the model's high-level features are still picking up on relevant "animacy" or 
    #    "vehicle" attributes, but the attack has pushed it across a fine-grained boundary.
    #    Random confusions (e.g., airplane → frog) suggest that the adversarial noise has 
    #    completely obliterated the model's feature extraction, leading to catastrophic 
    #    misclassification.
    #
    # 2. TEXTURE VS SHAPE PATTERNS
    #    ResNet (texture-biased) often fails in ways that are hard to predict, as it is 
    #    easily fooled by local pixel patterns. ShapeResNet (shape-biased) should ideally 
    #    fail in more semantically consistent ways, as its global feature extraction is 
    #    more robust to local texture disruption.
    #
    # 3. METRIC OF CONCEPTUAL STABILITY
    #    The "Difference" matrix highlights the exactly which classes are the most 
    #    vulnerable to becoming "sinks" for adversarial examples.

    print(f"\nAll confusion matrices saved to phase4_analysis/figures/{{model}}/confusion/")

if __name__ == '__main__':
    main()
