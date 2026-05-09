"""
SDT Visualization: d-Prime Curves and Per-Class Sensitivity Heatmaps
====================================================================

PURPOSE:
    Generates the two key SDT figures for the paper/presentation:

    PLOT 1 — d' vs Epsilon (CNN vs Human)
        The SDT equivalent of the accuracy divergence curve, but on a
        bias-free sensitivity scale. The d'=1.0 detection threshold line
        is marked, and the exact crossing points are annotated.

    PLOT 2 — Per-Class d' Heatmap at ε=0.10
        Side-by-side comparison of CNN and human perceptual sensitivity
        for each of the 10 CIFAR-10 classes at a moderate perturbation level.
        This reveals which specific object categories the CNN loses
        sensitivity to first (texture-dependent classes like cats, dogs)
        vs which categories humans maintain sensitivity on.

COLOR SCHEME:
    Matches the color palette from phase4_analysis/divergence_curves.py:
        CNN:   #E94560 (vibrant red)
        Human: #1A1A2E (deep navy)
        Gap:   #9D4EDD (violet)

HOW TO READ THE d' CURVE:
    • d'=1.0 dashed line is the "detection threshold" — below this line,
      the system is at near-chance discrimination.
    • The vertical distance between curves at any epsilon is the bias-free
      sensitivity gap. Unlike accuracy gaps, this cannot be explained by
      the CNN simply "shifting its criterion" — it truly lost the ability
      to distinguish classes.
    • The annotated crossing points give the exact epsilon values for the
      headline finding.

WHAT THIS MEANS AT A CONFERENCE:
    This figure is the centerpiece of a poster or talk. A reviewer will
    immediately understand d' curves and the 1.0 threshold from decades
    of psychophysics convention. No explanation needed — just the figure
    speaks the language of the field.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phase1_training.dataset import CLASSES

# Paths
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'results', 'partial_sdt_results.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Color scheme (matches Phase 4 divergence_curves.py)
COLOR_CNN = '#E94560'     # Vibrant red (ResNet)
COLOR_HUMAN = '#2E8B57'   # Sea Green (Human)
COLOR_VIT = '#9D4EDD'     # Violet (ViT)
COLOR_THRESHOLD = '#FFD700'  # Gold for the d'=1.0 line


def load_results():
    """Load the SDT results CSV produced by sdt_analysis.py."""
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"SDT results not found at {RESULTS_CSV}.\n"
            f"Run sdt_analysis.py first to generate the results."
        )
    return pd.read_csv(RESULTS_CSV)


def find_threshold_crossing(epsilons, d_primes, threshold=1.0):
    """
    Find the epsilon where d' crosses below a threshold.

    Uses linear interpolation between the last above-threshold point and
    the first below-threshold point for a more precise estimate.

    Returns (crossing_epsilon, last_above_dp, first_below_dp) or None.
    """
    for i in range(1, len(d_primes)):
        if d_primes[i] < threshold and d_primes[i - 1] >= threshold:
            # Linear interpolation for precise crossing
            frac = (d_primes[i - 1] - threshold) / (d_primes[i - 1] - d_primes[i])
            crossing = epsilons[i - 1] + frac * (epsilons[i] - epsilons[i - 1])
            return crossing
    # Check if already below at the start
    if len(d_primes) > 0 and d_primes[0] < threshold:
        return epsilons[0]
def plot_dprime_vs_epsilon(df):
    """
    PLOT 1: d-prime vs Epsilon for ResNet, ViT, and Humans.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Compute mean d' across all classes
    resnet_df = df[df['system'] == 'ResNet'].groupby('epsilon')['d_prime'].mean().reset_index()
    vit_df = df[df['system'] == 'ViT'].groupby('epsilon')['d_prime'].mean().reset_index()
    human_df = df[df['system'] == 'Human'].groupby('epsilon')['d_prime'].mean().reset_index()

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # d'=1.0 threshold line
    ax.axhline(y=1.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=2.0,
               alpha=0.8, label="Detection Threshold (d′ = 1.0)", zorder=1)

    # Plot Curves
    ax.plot(resnet_df['epsilon'], resnet_df['d_prime'], marker='o', markersize=8, linewidth=3,
            color=COLOR_CNN, label='ResNet-18', zorder=3)
    ax.plot(vit_df['epsilon'], vit_df['d_prime'], marker='^', markersize=8, linewidth=3,
            color=COLOR_VIT, label='ViT-Small', zorder=3)
    ax.plot(human_df['epsilon'], human_df['d_prime'], marker='s', markersize=8, linewidth=3,
            color=COLOR_HUMAN, label='Human Perception', zorder=3)

    # --- Annotate threshold crossings ---
    res_cross = find_threshold_crossing(resnet_df['epsilon'].values, resnet_df['d_prime'].values)
    vit_cross = find_threshold_crossing(vit_df['epsilon'].values, vit_df['d_prime'].values)
    hum_cross = find_threshold_crossing(human_df['epsilon'].values, human_df['d_prime'].values)

    if res_cross is not None:
        ax.annotate(f'ResNet: ε={res_cross:.3f}', xy=(res_cross, 1.0),
                    xytext=(res_cross + 0.01, 1.5), fontsize=9, fontweight='bold', color=COLOR_CNN,
                    arrowprops=dict(arrowstyle='->', color=COLOR_CNN, lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_CNN, alpha=0.9))

    if vit_cross is not None:
        ax.annotate(f'ViT: ε={vit_cross:.3f}', xy=(vit_cross, 1.0),
                    xytext=(vit_cross + 0.01, 0.5), fontsize=9, fontweight='bold', color=COLOR_VIT,
                    arrowprops=dict(arrowstyle='->', color=COLOR_VIT, lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_VIT, alpha=0.9))

    # --- Labels and styling ---
    ax.set_title("Signal Detection Analysis: Sensitivity Collapse Comparison",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Perturbation Budget (Epsilon)', fontsize=13)
    ax.set_ylabel("Sensitivity Index (d′)", fontsize=13)
    ax.set_ylim(-2.5, 6.0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc='upper right')

    # Shade the "below threshold" zone
    ax.fill_between([-0.01, 0.31], -3, 1.0, color='red', alpha=0.03, zorder=0)
    ax.text(0.3, 0.5, 'Near-chance threshold (d′=1.0)', fontsize=10,
            ha='right', va='center', fontstyle='italic', color='red', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'partial_dprime_curves.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot 1 saved: {out_path}")

    return res_cross, vit_cross, hum_cross


def plot_perclass_dprime_heatmap(df, target_epsilon=0.10):
    """
    PLOT 2: Per-class d-prime heatmap at a fixed epsilon, CNN vs Human side-by-side.

    PLAIN LANGUAGE:
        This heatmap answers: "At moderate adversarial noise (ε=0.10), which
        specific object categories does the CNN lose sensitivity to, and are
        those the same categories humans struggle with?"

        If the CNN loses sensitivity to 'cat' and 'dog' (texture-heavy classes)
        but humans maintain full sensitivity, that's direct evidence of
        texture bias per Geirhos et al. (2019).

    COLOR CODING:
        Green = high d' (strong sensitivity)
        Red   = low d' (weak/lost sensitivity)
        The delta panel shows where CNN and human diverge class-by-class.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter to target epsilon
    eps_df = df[df['epsilon'].round(2) == target_epsilon]
    if len(eps_df) == 0:
        # Fallback: use closest available epsilon
        available = sorted(df['epsilon'].unique())
        target_epsilon = min(available, key=lambda x: abs(x - target_epsilon))
        eps_df = df[df['epsilon'].round(2) == round(target_epsilon, 2)]
        print(f"  ⚠️  Using closest epsilon: {target_epsilon:.2f}")

    cnn_data = eps_df[eps_df['system'] == 'ResNet'].sort_values('class_idx')
    human_data = eps_df[eps_df['system'] == 'Human'].sort_values('class_idx')

    if len(cnn_data) == 0 or len(human_data) == 0:
        print("  ⚠️  Insufficient data for heatmap. Skipping.")
        return

    cnn_dprime = cnn_data['d_prime'].values.reshape(1, -1)
    human_dprime = human_data['d_prime'].values.reshape(1, -1)
    delta = cnn_dprime - human_dprime

    class_labels = [CLASSES[int(i)] for i in cnn_data['class_idx'].values]

    # --- Create 3-panel heatmap ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=150)

    # Panel 1: ResNet d'
    sns.heatmap(cnn_dprime, ax=axes[0], cmap='RdYlGn', vmin=0, vmax=5,
                xticklabels=class_labels, yticklabels=['ResNet'],
                annot=True, fmt='.2f', cbar_kws={'label': "d′"},
                linewidths=0.5)
    axes[0].set_title(f"ResNet Perceptual Sensitivity (d′) at ε={target_epsilon:.2f}",
                      fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Panel 2: Human d'
    sns.heatmap(human_dprime, ax=axes[1], cmap='RdYlGn', vmin=0, vmax=5,
                xticklabels=class_labels, yticklabels=['Human'],
                annot=True, fmt='.2f', cbar_kws={'label': "d′"},
                linewidths=0.5)
    axes[1].set_title(f"Human Perceptual Sensitivity (d′) at ε={target_epsilon:.2f}",
                      fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    # Panel 3: Delta (ResNet - Human)
    sns.heatmap(delta, ax=axes[2], cmap='coolwarm_r', vmin=-4, vmax=4,
                xticklabels=class_labels, yticklabels=['Δ (ResNet−Human)'],
                annot=True, fmt='.2f', cbar_kws={'label': "Δd′"},
                linewidths=0.5)
    axes[2].set_title(f"Sensitivity Gap (ResNet − Human) at ε={target_epsilon:.2f}",
                      fontsize=13, fontweight='bold', pad=10)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

    plt.suptitle("Per-Class Signal Detection Sensitivity Comparison",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f'perclass_dprime_eps{target_epsilon:.2f}.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot 2 saved: {out_path}")


def main():
    print("Loading SDT results...")
    df = load_results()

    print(f"  Found {len(df)} rows: {df['system'].nunique()} systems, "
          f"{df['epsilon'].nunique()} epsilon levels, {df['class'].nunique()} classes\n")

    # Plot 1: d' vs epsilon
    res_cross, vit_cross, hum_cross = plot_dprime_vs_epsilon(df)

    # Plot 2: Per-class heatmap at ε=0.10
    plot_perclass_dprime_heatmap(df, target_epsilon=0.10)

    # Print summary
    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    if res_cross is not None:
        print(f"  ResNet threshold crossing: ε = {res_cross:.3f}")
    if vit_cross is not None:
        print(f"  ViT threshold crossing:    ε = {vit_cross:.3f}")
    if hum_cross is not None:
        print(f"  Human threshold crossing:  ε = {hum_cross:.3f}")
    else:
        print(f"  Human: d' stays above 1.0 across all tested epsilons")
    print(f"\n  Figures saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
