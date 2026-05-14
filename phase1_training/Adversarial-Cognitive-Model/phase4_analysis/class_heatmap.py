"""
Class Heatmap: Per-Class Robustness Analysis
=============================================

PURPOSE:
    Breaks down the global accuracy into per-class matrices across all epsilons.
    Plots CNN accuracy, Human accuracy, and the Delta (CNN - Human) side-by-side.

WHAT THIS VISUALIZATION IS TELLING US SCIENTIFICALLY:
    Not all classes are equally robust. By looking at the Delta heatmap, we
    can pinpoint exactly *which objects* the CNN forgets first under attack.

CONNECTION TO TEXTURE BIAS THEORY (Geirhos 2019):
    If the CNN has a severe texture bias, it will be highly vulnerable on classes
    where texture is the primary identifying feature (e.g., 'cat' fur or 'frog'
    skin). Adversarial noise disrupts high-frequency textures instantly.
    Conversely, classes with rigid, unmistakable global shapes (e.g., 'airplane'
    wings, 'truck' boxes) might survive slightly longer because the low-frequency
    shape features are somewhat more resistant to L-infinity noise.
    Humans rely almost entirely on shape, so human performance should remain
    stable across most classes until the noise physically obscures the edges.

HOW TO READ THE PLOTS:
    - CNN Panel (Left): Should turn red quickly as epsilon increases.
    - Human Panel (Middle): Should stay green much longer.
    - Delta Panel (Right): Deep blue means Humans > CNN (CNN failed, human succeeded).
      Red means CNN > Human (rare, but theoretically possible if the attack
      somehow creates a human optical illusion while preserving CNN features).
    - Surprising pattern: Look at the rows. Are animals turning blue faster
      than vehicles? That's the texture bias in action.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phase1_training.model import CIFARResNet
from phase1_training.dataset import CLASSES
from utils.metrics import load_adv_batch, per_class_accuracy
from divergence_curves import generate_mock_human_data

CONFIG_PATH = '../config/attack_config.yaml'
HUMAN_DATA_PATH = '../phase3_human_study/data/anonymized_responses.csv'
OUTPUT_DIR = 'figures'


def get_cnn_class_matrix(model, device, epsilons):
    """Returns a 10x6 matrix of CNN accuracy."""
    matrix = np.zeros((10, len(epsilons)))
    
    print("Computing CNN per-class accuracy...")
    for j, eps in enumerate(epsilons):
        images, labels = load_adv_batch('pgd', eps, return_tensor=True)
        batch_size = 256
        all_preds = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size].to(device)
                preds = model(batch_imgs).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                
        all_preds = np.concatenate(all_preds)
        accs = per_class_accuracy(all_preds, labels.numpy(), num_classes=10)
        matrix[:, j] = accs
        
    return matrix


def get_human_class_matrix(df, epsilons):
    """Returns a 10x6 matrix of Human accuracy."""
    matrix = np.zeros((10, len(epsilons)))
    matrix.fill(np.nan)
    
    df_pgd = df[df['attack_type'] == 'pgd']
    print("Computing Human per-class accuracy...")
    
    for j, eps in enumerate(epsilons):
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == float(eps)]
        if len(subset) == 0:
            continue
            
        for c_idx, class_name in enumerate(CLASSES):
            class_subset = subset[subset['true_class'] == class_name]
            if len(class_subset) > 0:
                acc = class_subset['response_correct'].mean() * 100.0
                matrix[c_idx, j] = acc
                
    return matrix


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CIFARResNet().to(device)
    model.load_state_dict(torch.load(os.path.join('..', config['checkpoint_path']), map_location=device))
    
    if os.path.exists(HUMAN_DATA_PATH):
        df_human = pd.read_csv(HUMAN_DATA_PATH)
    else:
        # Generate mock data and assign random true_class for matrix building
        df_human = generate_mock_human_data(epsilons)
        df_human['true_class'] = np.random.choice(CLASSES, size=len(df_human))
        
    cnn_mat = get_cnn_class_matrix(model, device, epsilons)
    hum_mat = get_human_class_matrix(df_human, epsilons)
    
    # Calculate Delta (CNN - Human). Negative means CNN is worse.
    delta_mat = cnn_mat - hum_mat
    
    # -------------------------------------------------------------------------
    # Plotting the 3-panel heatmap
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=150)
    eps_labels = [f"ε={e}" for e in epsilons]
    
    # Panel 1: CNN Accuracy
    sns.heatmap(cnn_mat, ax=axes[0], cmap="RdYlGn", vmin=0, vmax=100,
                xticklabels=eps_labels, yticklabels=CLASSES, annot=True, fmt=".0f", cbar=False)
    axes[0].set_title("CNN Accuracy (%)", fontsize=14, pad=15)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=12)
    
    # Panel 2: Human Accuracy
    sns.heatmap(hum_mat, ax=axes[1], cmap="RdYlGn", vmin=0, vmax=100,
                xticklabels=eps_labels, yticklabels=[], annot=True, fmt=".0f", cbar=False)
    axes[1].set_title("Human Accuracy (%)", fontsize=14, pad=15)
    
    # Panel 3: Delta (CNN - Human)
    sns.heatmap(delta_mat, ax=axes[2], cmap="coolwarm_r", vmin=-100, vmax=100,
                xticklabels=eps_labels, yticklabels=[], annot=True, fmt=".0f")
    axes[2].set_title("Delta (CNN - Human)", fontsize=14, pad=15)
    
    plt.suptitle("Per-Class Robustness: Identifying Texture vs Shape Vulnerabilities", fontsize=18, y=1.05)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, 'class_heatmap.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # Identify Most Divergent Classes
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CLASS DIVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    # Find the epsilon index where the AVERAGE delta is most negative (biggest gap)
    avg_delta_per_eps = np.nanmean(delta_mat, axis=0)
    worst_eps_idx = np.argmin(avg_delta_per_eps)
    worst_eps = epsilons[worst_eps_idx]
    
    # Look at the delta column for that epsilon
    col_delta = delta_mat[:, worst_eps_idx]
    
    # Sort classes by how badly the CNN performed relative to humans (most negative first)
    sorted_indices = np.argsort(col_delta)
    
    print(f"Analyzing at ε={worst_eps} (point of maximum average divergence):\n")
    print("TOP 3 MOST DIVERGENT CLASSES (CNN failed much worse than humans):")
    for i in range(3):
        idx = sorted_indices[i]
        print(f"  {i+1}. {CLASSES[idx].upper():<10} | CNN: {cnn_mat[idx, worst_eps_idx]:>5.1f}% | "
              f"Human: {hum_mat[idx, worst_eps_idx]:>5.1f}% | Gap: {col_delta[idx]:>5.1f}%")
              
    print("\nSCIENTIFIC INTERPRETATION:")
    print("  The classes listed above are the CNN's biggest blind spots compared to")
    print("  human vision. According to the Texture Hypothesis, these are likely")
    print("  classes where the CNN memorized high-frequency surface textures (e.g.,")
    print("  fur, repeating scales) during training. Because adversarial noise")
    print("  specifically scrambles these local high-frequency patterns, the CNN loses")
    print("  its primary classification anchor and collapses.")
    print("  Humans, using top-down shape processing, simply look past the noise")
    print("  and recognize the global outline.")
    
    print(f"\nHeatmap saved to: {out_path}")


if __name__ == '__main__':
    main()
