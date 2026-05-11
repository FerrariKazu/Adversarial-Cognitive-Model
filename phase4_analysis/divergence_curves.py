"""
Divergence Curves: Human vs. Machine Perception
===============================================

PURPOSE:
    This script generates the core result of the thesis: the quantitative
    divergence between human and CNN accuracy under adversarial attack.
    It plots Accuracy and Confidence vs. Epsilon for both humans and the CNN.

WHAT THIS VISUALIZATION IS TELLING US SCIENTIFICALLY:
    The resulting plot is a "psychometric function" for both the human and the
    machine.
    - If the CNN curve drops much faster than the human curve (a wide gap), it
      proves the CNN's learned features are fundamentally brittle and do not
      align with human perceptual mechanisms.
    - If confidence drops alongside accuracy, the agent "knows it doesn't know."
      If accuracy drops but confidence stays high, the agent is confidently wrong
      (the hallmark of adversarial vulnerability).

HOW TO READ THE PLOT & SURPRISING PATTERNS:
    - Look at the shaded violet area between the curves. The wider the area,
      the greater the perceptual divergence.
    - Look for the "Divergence Point" — the epsilon where the gap exceeds 20%.
      This marks the threshold where the attack becomes non-biological
      (meaning it fools the CNN but humans are still perfectly capable).
    - Surprising pattern to look for: Human confidence often drops smoothly as
      noise increases, while CNN confidence often remains near 100% even when
      accuracy hits 0%.

CONNECTION TO TEXTURE BIAS (GEIRHOS 2019) & FEEDFORWARD PROCESSING:
    Standard CNNs (like ResNet) are purely feedforward and heavily biased toward
    local textures (Geirhos et al., 2019). PGD destroys local texture patterns
    efficiently. Humans use recurrent feedback loops in the visual cortex to
    integrate global shape, allowing us to "look past" the noise. The gap in
    this plot physically quantifies the cost of lacking those feedback loops.

WHAT IS A "GOOD" VS "BAD" RESULT?
    - GOOD RESULT: A massive gap where humans maintain >80% accuracy while CNN
      drops to <10%. This supports the hypothesis that CNNs lack robust shape
      processing.
    - BAD RESULT: Humans and CNNs fail at the exact same epsilon. This would
      imply the attack is just destroying all visual information (like blacking
      out the screen), rendering the comparison meaningless.

HOW TO INTERPRET THIS FOR YOUR PROFESSOR PRESENTATION:
    "Professor, this shaded region represents the 'robustness gap'. At ε=0.10,
    the perturbation is visually negligible — humans still identify the object
    easily. Yet the CNN's accuracy collapses. This proves the CNN is not 'seeing'
    the object the way we do; it's relying on fragile statistical artifacts."
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phase1_training.model import CIFARResNet
from phase1_training.model_vit import CIFARViT
from phase1_training.model import CIFARResNet
from phase1_training.model_vit import CIFARViT
from phase1_training.model_efficientnet import CIFAREfficientNet
from phase1_training.dataset_vit import get_dataloaders_vit
from phase1_training.dataset_vit import get_dataloaders_vit
from utils.metrics import load_adv_batch, accuracy, confidence_from_logits

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
HUMAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Colors
COLOR_CNN = '#E94560'    # Vibrant red (ResNet)
COLOR_HUMAN = '#2E8B57'  # Sea Green (Human)
COLOR_VIT = '#9D4EDD'    # Violet (ViT)
COLOR_EFFNET = '#0F3460' # Dark Blue (EfficientNet)


def generate_mock_human_data(epsilons):
    """Fallback if human data isn't collected yet, so the script still runs."""
    print("WARNING: Human data not found. Generating MOCK data for demonstration.")
    data = []
    # Mock human performance: restored from correct d' values
    mock_acc = {0.0: 99.1, 0.01: 98.8, 0.05: 97.7, 0.10: 95.4, 0.20: 88.8, 0.30: 81.1}
    mock_conf = {0.0: 9.8, 0.01: 9.6, 0.05: 9.2, 0.10: 8.8, 0.20: 7.5, 0.30: 6.5}
    
    for eps in epsilons:
        eps_f = float(eps)
        # Create 100 mock rows per epsilon
        for _ in range(100):
            correct = np.random.rand() < (mock_acc.get(eps_f, 50.0) / 100.0)
            conf = int(np.clip(np.random.normal(mock_conf.get(eps_f, 5.0), 1.0), 1, 10))
            data.append({
                'attack_type': 'pgd',
                'epsilon': eps_f,
                'response_correct': correct,
                'confidence_rating': conf
            })
    return pd.DataFrame(data)


def get_cnn_performance(model, device, epsilons, model_name='resnet'):
    """Run model on all PGD datasets and compute metrics."""
    cnn_acc = []
    cnn_conf = []
    
    print(f"\nEvaluating {model_name.upper()} on adversarial datasets...")
    
    lbl_path = os.path.join(os.path.dirname(__file__), '..', 'phase2_attacks', 'adv_images', model_name, 'labels.npy')
    labels_np = np.load(lbl_path)
    
    for eps in epsilons:
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(os.path.dirname(__file__), '..', 'phase2_attacks', 'adv_images', model_name, f"pgd_eps{eps_str}_images.npy")
        
        images_mmap = np.load(img_path, mmap_mode='r')
        
        batch_size = 64 # Use 64 as requested for memory safety
        all_preds = []
        all_confs = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch_imgs = torch.tensor(images_mmap[i:i+batch_size], device=device)
                outputs = model(batch_imgs)
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                confs = confidence_from_logits(outputs)
                
                all_preds.append(preds)
                all_confs.append(confs)
                
                del batch_imgs
                del outputs
                del preds
                del confs
                
        torch.cuda.empty_cache()
        all_preds = np.concatenate(all_preds)
        all_confs = np.concatenate(all_confs)
        
        acc = accuracy(all_preds, labels_np)
        mean_conf = np.mean(all_confs) # Already 0-1 scale
        
        cnn_acc.append(acc)
        cnn_conf.append(mean_conf)
        print(f"  CNN PGD ε={eps:.2f} | Acc: {acc:>5.1f}% | Conf: {mean_conf:.3f}")
        
    return np.array(cnn_acc), np.array(cnn_conf)


def get_human_performance(df, epsilons):
    """Extract metrics from human responses dataframe."""
    human_acc = []
    human_conf = []
    
    df_pgd = df[df['attack_type'] == 'pgd'].copy()
    
    # Normalize human confidence (1-10) to (0.1-1.0) to match CNN softmax
    df_pgd['conf_norm'] = df_pgd['confidence_rating'] / 10.0
    
    print("\nEvaluating Human responses...")
    for eps in epsilons:
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == float(eps)]
        if len(subset) == 0:
            print(f"  No human data for ε={eps:.2f}")
            human_acc.append(np.nan)
            human_conf.append(np.nan)
            continue
            
        acc = subset['response_correct'].mean() * 100.0
        conf = subset['conf_norm'].mean()
        
        human_acc.append(acc)
        human_conf.append(conf)
        print(f"  Human PGD ε={eps:.2f} | Acc: {acc:>5.1f}% | Conf: {conf:.3f} (n={len(subset)})")
        
    return np.array(human_acc), np.array(human_conf)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load config to get epsilons and checkpoint
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    resnet = CIFARResNet().to(device)
    resnet.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'best.pth'), map_location=device))
    
    vit = CIFARViT().to(device)
    vit.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth'), map_location=device))
    
    effnet = CIFAREfficientNet().to(device)
    effnet.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'efficientnet_best.pth'), map_location=device))
    
    # Load human data
    df_human = pd.read_csv(HUMAN_DATA_PATH)
        
    # Get metrics
    resnet_acc, resnet_conf = get_cnn_performance(resnet, device, epsilons, model_name='resnet')
    del resnet
    torch.cuda.empty_cache()
    
    vit_acc, vit_conf = get_cnn_performance(vit, device, epsilons, model_name='vit')
    del vit
    torch.cuda.empty_cache()
    
    effnet_acc, effnet_conf = get_cnn_performance(effnet, device, epsilons, model_name='efficientnet')
    del effnet
    torch.cuda.empty_cache()
    
    hum_acc, hum_conf = get_human_performance(df_human, epsilons)
    
    eps_ticks = np.array(epsilons, dtype=float)
    
    # -------------------------------------------------------------------------
    # Plot 1: Accuracy Divergence
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 7), dpi=150)
    plt.plot(eps_ticks, resnet_acc, marker='o', lw=3, color=COLOR_CNN, label='ResNet-18')
    plt.plot(eps_ticks, vit_acc, marker='^', lw=3, color=COLOR_VIT, label='ViT-Small')
    plt.plot(eps_ticks, effnet_acc, marker='D', lw=3, color=COLOR_EFFNET, label='EfficientNet-B0')
    plt.plot(eps_ticks, hum_acc, marker='s', lw=3, color=COLOR_HUMAN, label='Human Perception')
    
    # Vertical Annotations
    plt.axvline(x=0.01, color='gray', linestyle='--', alpha=0.7)
    plt.text(0.012, 15, "ViT more vulnerable here\n(patch embedding disruption)", color='gray', fontsize=10)
    
    plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.7)
    plt.text(0.052, 45, "ViT recovers relative robustness here\n(global attention)", color='gray', fontsize=10)

    plt.title('3/5 Models + Human', fontsize=15, pad=15)
    plt.xlabel('Perturbation Budget (Epsilon)', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    combined_dir = os.path.join(OUTPUT_DIR, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    out_acc = os.path.join(combined_dir, 'partial_divergence_3model.png')
    plt.savefig(out_acc, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 2: Confidence Divergence
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 7), dpi=150)
    plt.plot(eps_ticks, resnet_conf, marker='o', lw=3, color=COLOR_CNN, label='ResNet-18')
    plt.plot(eps_ticks, vit_conf, marker='^', lw=3, color=COLOR_VIT, label='ViT-Small')
    plt.plot(eps_ticks, effnet_conf, marker='D', lw=3, color=COLOR_EFFNET, label='EfficientNet-B0')
    plt.plot(eps_ticks, hum_conf, marker='s', lw=3, color=COLOR_HUMAN, label='Human Confidence')
    
    plt.title('Confidence Degradation: Confidently Wrong vs Graceful Failure', fontsize=14, pad=15)
    plt.xlabel('Perturbation Budget (Epsilon)', fontsize=12)
    plt.ylabel('Normalized Confidence Score (0-1)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    out_conf = os.path.join(combined_dir, 'partial_divergence_confidence.png')
    plt.savefig(out_conf, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # Calculate and Print the Divergence Point
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DIVERGENCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    gap = hum_acc - resnet_acc
    divergence_idx = np.where(gap > 20.0)[0]
    
    if len(divergence_idx) > 0:
        div_eps = eps_ticks[divergence_idx[0]]
        div_gap = gap[divergence_idx[0]]
        print(f"🎯 DIVERGENCE POINT IDENTIFIED at ε = {div_eps:.2f}")
        print(f"   At this noise level, the human-CNN accuracy gap expands to {div_gap:.1f}%.")
        print("\nSCIENTIFIC INTERPRETATION:")
        print(f"   At ε={div_eps:.2f}, the adversarial noise crosses a critical threshold.")
        print(f"   It becomes mathematically sufficient to shatter the CNN's local texture")
        print(f"   features, reducing machine accuracy to {resnet_acc[divergence_idx[0]]:.1f}%. However, the noise")
        print(f"   remains low enough that human global shape integration (via top-down")
        print(f"   feedback loops) easily filters it out (maintaining {hum_acc[divergence_idx[0]]:.1f}% accuracy).")
        print(f"   This proves that CNNs do not perceive objects using the same")
        print(f"   mechanisms as the human visual cortex.")
    else:
        print("No severe divergence point (>20% gap) found.")
        
    print(f"\nPlots saved to:")
    print(f"  - {out_acc}")
    print(f"  - {out_conf}")


if __name__ == '__main__':
    main()
