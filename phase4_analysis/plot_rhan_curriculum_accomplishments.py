#!/usr/bin/env python3
"""
Plots curriculum training accomplishments and concept ablation analysis.
Generates:
1. concept_activation_ablation.png
2. sdt_sensitivity_decay.png
3. pgd_accuracy_decay.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directories exist
os.makedirs('figures', exist_ok=True)
os.makedirs('phase4_analysis/figures/combined', exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD CONCEPT ABLATION RESULTS
# ----------------------------------------------------------------------
concept_results_path = 'concept_ablation_results.npz'
if os.path.exists(concept_results_path):
    data = np.load(concept_results_path)
    concepts = []
    # Find unique concept names
    for key in data.keys():
        if key.endswith('_phase_b_acc'):
            concepts.append(key[:-12])
            
    concept_data = []
    for c in concepts:
        b_acc = float(data[f'{c}_phase_b_acc'])
        c_acc = float(data[f'{c}_phase_c_acc'])
        impr = float(data[f'{c}_improvement'])
        concept_data.append({
            'concept': c,
            'phase_b': b_acc,
            'phase_c': c_acc,
            'improvement': impr
        })
    # Sort by improvement descending
    concept_data = sorted(concept_data, key=lambda x: x['improvement'], reverse=True)
else:
    print("Warning: concept_ablation_results.npz not found. Using exact printed concept data.")
    concept_data = [
        {'concept': 'quadrupedal', 'phase_b': 0.968, 'phase_c': 0.970, 'improvement': 0.002},
        {'concept': 'has_tail', 'phase_b': 0.968, 'phase_c': 0.970, 'improvement': 0.002},
        {'concept': 'water_vehicle', 'phase_b': 0.985, 'phase_c': 0.983, 'improvement': -0.002},
        {'concept': 'has_wheels', 'phase_b': 0.989, 'phase_c': 0.986, 'improvement': -0.003},
        {'concept': 'has_cab', 'phase_b': 0.989, 'phase_c': 0.986, 'improvement': -0.003},
        {'concept': 'vehicle_body', 'phase_b': 0.989, 'phase_c': 0.986, 'improvement': -0.003},
        {'concept': 'ground_vehicle', 'phase_b': 0.989, 'phase_c': 0.986, 'improvement': -0.003},
        {'concept': 'flying', 'phase_b': 0.971, 'phase_c': 0.967, 'improvement': -0.004},
        {'concept': 'pointed_ears', 'phase_b': 0.928, 'phase_c': 0.923, 'improvement': -0.005},
        {'concept': 'tall_profile', 'phase_b': 0.928, 'phase_c': 0.923, 'improvement': -0.005},
        {'concept': 'elongated_body', 'phase_b': 0.904, 'phase_c': 0.895, 'improvement': -0.009},
        {'concept': 'wide_profile', 'phase_b': 0.939, 'phase_c': 0.926, 'improvement': -0.012},
    ]

# ----------------------------------------------------------------------
# PLOT 1: CONCEPT ACTIVATION ABLATION (PHASE B VS PHASE C)
# ----------------------------------------------------------------------
sns.set_theme(style="whitegrid")

concepts_list = [d['concept'].replace('_', ' ').title() for d in concept_data]
b_accs = [d['phase_b'] * 100 for d in concept_data]
c_accs = [d['phase_c'] * 100 for d in concept_data]
improvements = [d['improvement'] * 100 for d in concept_data]

y = np.arange(len(concepts_list))
height = 0.35

fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
# Dark Slate Blue for Phase B vs Vibrant Violet for Phase C
rects1 = ax.barh(y - height/2, b_accs, height, label='Phase B (ε=0.100, β=6.0)', color='#1E293B', alpha=0.9)
rects2 = ax.barh(y + height/2, c_accs, height, label='Phase C (ε=0.150, β=5.0)', color='#8B5CF6', alpha=0.9)

ax.set_xlabel('Linear Probe Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Concept Representation Ablation: Curriculum Phase B vs. Phase C', fontsize=15, fontweight='bold', pad=20)
ax.set_yticks(y)
ax.set_yticklabels(concepts_list, fontsize=11, fontweight='medium')
ax.set_xlim(80, 101) # Focus on top 20% range to display subtle gaps
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=11)

# Annotate improvements/regressions
for i, (b, c, impr) in enumerate(zip(b_accs, c_accs, improvements)):
    sign = '+' if impr >= 0 else ''
    color = '#10B981' if impr >= 0 else '#EF4444' # Emerald green or rose red
    ax.text(max(b, c) + 0.3, i - 0.05, f"{sign}{impr:.1f}%", color=color, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('figures/concept_activation_ablation.png', bbox_inches='tight')
plt.savefig('phase4_analysis/figures/combined/concept_activation_ablation.png', bbox_inches='tight')
plt.close()
print("Saved concept_activation_ablation.png successfully.")

# ----------------------------------------------------------------------
# PLOT 2: SDT SENSITIVITY DECAY (d' vs Epsilon)
# ----------------------------------------------------------------------
epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]

dprime_data = {
    'Human (Visual Cognition)': ([4.790, 4.567, 3.985, 3.368, 2.440, 1.769], '#10B981', 'o', '-'),      # Emerald Green
    'RHAN-trades-curriculum (Ours)': ([2.748, 2.589, 2.159, 1.696, 0.877, 0.010], '#D97706', 'D', '-'),  # Amber Orange
    'RHAN-TRADES-Hardened': ([3.260, 3.032, 2.238, 1.357, -0.094, -1.664], '#8B5CF6', 's', '--'),       # Purple
    'RHAN-v5-TRADES': ([3.383, 3.186, 2.230, 1.231, -0.291, -1.602], '#EC4899', '^', '--'),             # Pink
    'RHAN-v3 (Unified Recurrent)': ([3.710, 3.189, 1.983, 0.753, -1.039, -3.044], '#3B82F6', 'v', ':'),  # Blue
    'ResNet-18 (Feedforward)': ([4.426, 2.687, -0.771, -1.707, -1.913, -1.880], '#EF4444', 'x', '-.'),   # Red
    'ViT-Small (Transformer)': ([4.931, 1.814, -0.154, -0.909, -1.242, -1.469], '#6B7280', '*', '-.')    # Gray
}

plt.figure(figsize=(11, 8), dpi=300)
# Near-chance threshold (d'=1.0)
plt.axhline(y=1.0, color='#374151', linestyle='--', alpha=0.7, linewidth=1.5, label="Near-chance Threshold (d'=1.0)")

# Shade the chance/blind zone below d'=1.0
plt.fill_between([-0.02, 0.32], -3.5, 1.0, color='#F3F4F6', alpha=0.5, label='Chance-Performance Region')

for name, (y_vals, color, marker, ls) in dprime_data.items():
    plt.plot(epsilons, y_vals, marker=marker, label=name, color=color, linestyle=ls, linewidth=2.5, markersize=8)

plt.title("Signal Detection Theory (SDT): Perceptual Sensitivity Collapse", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Adversarial Perturbation Budget (Epsilon, L_inf)", fontsize=12, labelpad=10)
plt.ylabel("Sensitivity Index (d')", fontsize=12, labelpad=10)
plt.xlim(-0.01, 0.31)
plt.ylim(-3.5, 5.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10.5, loc='lower left', frameon=True, facecolor='white', edgecolor='none')

# Annotate epsilon threshold for curriculum
plt.axvline(x=0.185, color='#D97706', linestyle=':', alpha=0.8, linewidth=1.5)
plt.text(0.188, -2.8, "RHAN-curriculum\nε_thresh = 0.185", color='#D97706', fontweight='bold', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Annotate ResNet-18 collapse threshold
plt.axvline(x=0.030, color='#EF4444', linestyle=':', alpha=0.8, linewidth=1.5)
plt.text(0.033, -2.8, "ResNet-18\nε_thresh = 0.030", color='#EF4444', fontweight='bold', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig('figures/sdt_sensitivity_decay.png', bbox_inches='tight')
plt.savefig('phase4_analysis/figures/combined/sdt_sensitivity_decay.png', bbox_inches='tight')
plt.close()
print("Saved sdt_sensitivity_decay.png successfully.")


# ----------------------------------------------------------------------
# PLOT 3: PGD ACCURACY DECAY
# ----------------------------------------------------------------------
acc_data = {
    'Human (Visual Cognition)': ([73.33, np.nan, 69.17, 59.17, 62.22, 58.61], '#10B981', 'o', '-'),      # Emerald Green
    'RHAN-trades-curriculum (Ours)': ([78.12, 75.00, 65.23, 52.93, 29.49, 10.16], '#D97706', 'D', '-'),  # Amber Orange
    'RHAN-TRADES-Hardened': ([86.33, 83.01, 67.19, 43.16, 8.59, 0.20], '#8B5CF6', 's', '--'),       # Purple
    'RHAN-v5-TRADES': ([87.30, 84.77, 65.82, 37.89, 5.47, 0.20], '#EC4899', '^', '--'),             # Pink
    'ResNet-18 (Feedforward)': ([95.82, 75.57, 2.84, 0.21, 0.02, 0.00], '#EF4444', 'x', '-.'),   # Red
    'ViT-Small (Transformer)': ([97.80, 55.18, 8.80, 2.78, 1.12, 0.58], '#6B7280', '*', '-.')    # Gray
}

plt.figure(figsize=(11, 8), dpi=300)

for name, (y_vals, color, marker, ls) in acc_data.items():
    y_vals_arr = np.array(y_vals)
    mask = ~np.isnan(y_vals_arr)
    plt.plot(np.array(epsilons)[mask], y_vals_arr[mask], marker=marker, label=name, color=color, linestyle=ls, linewidth=2.5, markersize=8)

plt.title("Adversarial Robustness: PGD-100 Accuracy Decay", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Adversarial Perturbation Budget (Epsilon, L_inf)", fontsize=12, labelpad=10)
plt.ylabel("Classification Accuracy (%)", fontsize=12, labelpad=10)
plt.xlim(-0.01, 0.31)
plt.ylim(-2, 102)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10.5, loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.text(0.18, 64, "Humans maintain ~60% accuracy\neven at extreme noise (ε=0.30)", color='#10B981', fontweight='semibold', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.text(0.06, 15, "RHAN-curriculum preserves\nrobust accuracy far longer\nthan standard models.", color='#D97706', fontweight='semibold', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig('figures/pgd_accuracy_decay.png', bbox_inches='tight')
plt.savefig('phase4_analysis/figures/combined/pgd_accuracy_decay.png', bbox_inches='tight')
plt.close()
print("Saved pgd_accuracy_decay.png successfully.")
