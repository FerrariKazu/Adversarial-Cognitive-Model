#!/usr/bin/env python3
"""
Visualization script to generate comparative heatmaps and sensitivity curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("competition/output", exist_ok=True)
    
    # Load stats
    standard_file = "competition/output/standard_stats.npz"
    epoch45_file = "competition/output/epoch45_stats.npz"
    
    if not os.path.exists(standard_file) or not os.path.exists(epoch45_file):
        print("Error: Evaluation stats files not found. Run evaluate_comparison.py first.")
        return

    std_data = np.load(standard_file, allow_pickle=True)
    ep45_data = np.load(epoch45_file, allow_pickle=True)
    
    # ── 1. Plot d-prime Sensitivity Decay ──────────────────────────────────────
    print("Generating d-prime decay curve comparison...")
    epsilons = std_data['epsilons']
    dp_avg_std = std_data['dp_avg']
    dp_avg_ep45 = ep45_data['dp_avg']
    
    # Human baseline from literature
    human_eps = np.array([0.00, 0.01, 0.05, 0.10, 0.20, 0.30])
    human_dp = np.array([4.790, 4.567, 3.985, 3.368, 2.440, 1.769])
    
    plt.figure(figsize=(8, 6))
    plt.plot(human_eps, human_dp, 'g-o', label='Human Baseline', linewidth=2.5, markersize=8)
    plt.plot(epsilons, dp_avg_ep45, 'b-s', label='Epoch 45 Checkpoint (TRADES + PL)', linewidth=2.0, markersize=6)
    plt.plot(epsilons, dp_avg_std, 'r--x', label='Standard Large Model (TDV pre-trained)', linewidth=2.0, markersize=6)
    
    # Add d' = 1.0 threshold line
    plt.axhline(y=1.0, color='gray', linestyle=':', label="Perceptual Threshold (d' = 1.0)")
    
    plt.title("Sensitivity d' Decay Across Perturbation Levels", fontsize=14, fontweight='bold')
    plt.xlabel("Perturbation Magnitude (L_inf Epsilon)", fontsize=12)
    plt.ylabel("Sensitivity Index (d')", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig("competition/output/dprime_decay_comparison.png", dpi=150)
    plt.close()
    
    # ── 2. Plot Class-wise AutoAttack Robustness Heatmap ────────────────────────
    print("Generating class-wise AutoAttack robustness heatmap...")
    std_class = std_data['class_stats'].item()
    ep45_class = ep45_data['class_stats'].item()
    
    classes = sorted(list(std_class.keys()))
    
    # AutoAttack Clean & Robust Accuracy arrays
    std_clean = [std_class[c]['clean'] for c in classes]
    ep45_clean = [ep45_class[c]['clean'] for c in classes]
    
    std_aa = [std_class[c]['aa_wb'] for c in classes]
    ep45_aa = [ep45_class[c]['aa_wb'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, std_aa, width, label='Standard Large (TDV)', color='#EF4444')
    rects2 = ax.bar(x + width/2, ep45_aa, width, label='Epoch 45 (TRADES + PL)', color='#3B82F6')
    
    ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
    ax.set_title('Class-wise Robustness Under White-Box AutoAttack (Epsilon=0.031)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig("competition/output/class_robustness_heatmap.png", dpi=150)
    plt.close()
    
    # ── 3. Plot Bypass Invariance Gap comparison ───────────────────────────────
    print("Generating bypass invariance gap comparison...")
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ['Standard Large', 'Epoch 45']
    wb_accs = [std_data['wb_acc'], ep45_data['wb_acc']]
    gb_accs = [std_data['gb_acc'], ep45_data['gb_acc']]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects_wb = ax.bar(x - width/2, wb_accs, width, label='White-Box (Feedback Active)', color='#10B981')
    rects_gb = ax.bar(x + width/2, gb_accs, width, label='Gray-Box (Bypass Gradient)', color='#6366F1')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Bypass Invariance: White-Box vs Gray-Box', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    autolabel(rects_wb)
    autolabel(rects_gb)
    
    plt.tight_layout()
    plt.savefig("competition/output/bypass_invariance_comparison.png", dpi=150)
    plt.close()
    
    print("Figures generated successfully inside competition/output/")

if __name__ == "__main__":
    main()
