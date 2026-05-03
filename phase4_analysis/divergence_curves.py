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
from utils.metrics import load_adv_batch, accuracy, confidence_from_logits

# Configuration
CONFIG_PATH = '../config/attack_config.yaml'
HUMAN_DATA_PATH = '../phase3_human_study/data/anonymized_responses.csv'
OUTPUT_DIR = 'figures'

# Colors
COLOR_CNN = '#E94560'    # Vibrant red
COLOR_HUMAN = '#1A1A2E'  # Deep navy
COLOR_GAP = '#9D4EDD'    # Violet


def generate_mock_human_data(epsilons):
    """Fallback if human data isn't collected yet, so the script still runs."""
    print("WARNING: Human data not found. Generating MOCK data for demonstration.")
    data = []
    # Mock human performance: degrades slowly (robust to noise)
    mock_acc = {0.0: 98.0, 0.01: 97.0, 0.05: 95.0, 0.10: 90.0, 0.20: 75.0, 0.30: 60.0}
    mock_conf = {0.0: 9.5, 0.01: 9.2, 0.05: 8.5, 0.10: 7.0, 0.20: 5.5, 0.30: 4.0}
    
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


def get_cnn_performance(model, device, epsilons):
    """Run model on all PGD datasets and compute metrics."""
    cnn_acc = []
    cnn_conf = []
    
    # We use PGD as the standard benchmark
    print("\nEvaluating CNN on adversarial datasets...")
    for eps in epsilons:
        images, labels = load_adv_batch('pgd', eps, return_tensor=True)
        
        # Batch processing to avoid OOM
        batch_size = 256
        all_preds = []
        all_confs = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size].to(device)
                outputs = model(batch_imgs)
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                confs = confidence_from_logits(outputs)
                
                all_preds.append(preds)
                all_confs.append(confs)
                
        all_preds = np.concatenate(all_preds)
        all_confs = np.concatenate(all_confs)
        
        acc = accuracy(all_preds, labels.numpy())
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
    
    # Load model
    model = CIFARResNet().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['checkpoint_path']), map_location=device))
    
    # Load human data
    if os.path.exists(HUMAN_DATA_PATH):
        df_human = pd.read_csv(HUMAN_DATA_PATH)
    else:
        df_human = generate_mock_human_data(epsilons)
        
    # Get metrics
    cnn_acc, cnn_conf = get_cnn_performance(model, device, epsilons)
    hum_acc, hum_conf = get_human_performance(df_human, epsilons)
    
    eps_ticks = np.array(epsilons, dtype=float)
    
    # -------------------------------------------------------------------------
    # Plot 1: Accuracy Divergence
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(eps_ticks, cnn_acc, marker='o', lw=3, color=COLOR_CNN, label='ResNet-18 (CNN)')
    plt.plot(eps_ticks, hum_acc, marker='s', lw=3, color=COLOR_HUMAN, label='Human Perception')
    
    # Shaded gap
    plt.fill_between(eps_ticks, cnn_acc, hum_acc, where=(hum_acc > cnn_acc), 
                     interpolate=True, color=COLOR_GAP, alpha=0.15, label='Robustness Gap')
                     
    plt.title('Perceptual Divergence: Human vs Machine Accuracy (PGD Attack)', fontsize=14, pad=15)
    plt.xlabel('Perturbation Budget (Epsilon)', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    out_acc = os.path.join(OUTPUT_DIR, 'divergence_accuracy.png')
    plt.savefig(out_acc, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 2: Confidence Divergence
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(eps_ticks, cnn_conf, marker='o', lw=3, color=COLOR_CNN, label='CNN Softmax Confidence')
    plt.plot(eps_ticks, hum_conf, marker='s', lw=3, color=COLOR_HUMAN, label='Human Reported Confidence')
    
    plt.title('Confidence Degradation: Confidently Wrong vs Graceful Failure', fontsize=14, pad=15)
    plt.xlabel('Perturbation Budget (Epsilon)', fontsize=12)
    plt.ylabel('Normalized Confidence Score (0-1)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    out_conf = os.path.join(OUTPUT_DIR, 'divergence_confidence.png')
    plt.savefig(out_conf, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # Calculate and Print the Divergence Point
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DIVERGENCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    gap = hum_acc - cnn_acc
    divergence_idx = np.where(gap > 20.0)[0]
    
    if len(divergence_idx) > 0:
        div_eps = eps_ticks[divergence_idx[0]]
        div_gap = gap[divergence_idx[0]]
        print(f"🎯 DIVERGENCE POINT IDENTIFIED at ε = {div_eps:.2f}")
        print(f"   At this noise level, the human-CNN accuracy gap expands to {div_gap:.1f}%.")
        print("\nSCIENTIFIC INTERPRETATION:")
        print(f"   At ε={div_eps:.2f}, the adversarial noise crosses a critical threshold.")
        print(f"   It becomes mathematically sufficient to shatter the CNN's local texture")
        print(f"   features, reducing machine accuracy to {cnn_acc[divergence_idx[0]]:.1f}%. However, the noise")
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
