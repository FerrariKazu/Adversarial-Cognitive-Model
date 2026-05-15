import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Configuration
SDT_RESULTS_PATH = 'phase5_sdt/results/sdt_results_v4.csv'
OUTPUT_DIR = 'phase4_analysis/figures/combined/threshold_summary'

# Exact colors from project
COLORS = {
    'resnet': '#E94560',
    'vit': '#7C3AED',
    'efficientnet': '#0F3460',
    'shaperesnet': '#16A34A',
    'human': '#22C55E',
    'bagnet': '#558B2F'
}

def find_threshold(x, y, threshold_val, kind='linear'):
    """Interpolates to find where y crosses threshold_val."""
    if len(x) < 2: return np.nan
    try:
        # We need y to be monotonic for unique solution, 
        # but adversarial decay is usually monotonic enough.
        f = interp1d(y, x, kind=kind, fill_value="extrapolate")
        return float(f(threshold_val))
    except:
        return np.nan

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SDT_RESULTS_PATH):
        print(f"Error: {SDT_RESULTS_PATH} not found. Run sdt_analysis.py first.")
        return

    df = pd.read_csv(SDT_RESULTS_PATH)
    # Normalize system names to lowercase for color mapping
    df['system_key'] = df['system'].str.lower()
    
    # Aggregate metrics per system and epsilon
    summary = df.groupby(['system_key', 'epsilon']).agg({
        'hit_rate': 'mean',
        'd_prime': 'mean'
    }).reset_index()

    systems = summary['system_key'].unique()
    data = []

    for sys in systems:
        sys_df = summary[summary['system_key'] == sys].sort_values('epsilon')
        eps = sys_df['epsilon'].values
        acc = sys_df['hit_rate'].values * 100.0
        dp = sys_df['d_prime'].values
        
        # Clean Accuracy (at eps=0)
        clean_acc = acc[0]
        
        # 50% Accuracy Threshold
        if sys == 'human':
            thresh_50 = 0.40 # Beyond our test range
        else:
            thresh_50 = find_threshold(eps, acc, 50.0)
            # Clip to valid range
            thresh_50 = max(0, min(thresh_50, 0.40))
            
        # d'=1.0 Threshold
        if sys == 'human':
            thresh_dp = 0.40
        else:
            thresh_dp = find_threshold(eps, dp, 1.0)
            thresh_dp = max(0, min(thresh_dp, 0.40))

        data.append({
            'System': sys,
            'Clean Acc': clean_acc,
            '50% Threshold': thresh_50,
            'DP_Threshold': thresh_dp
        })

    results_df = pd.DataFrame(data).sort_values('50% Threshold', ascending=False)

    # --- Figure 1: Accuracy Threshold Bar Chart ---
    plt.figure(figsize=(10, 6), dpi=150)
    # Re-sort for fragile-to-robust (ascending)
    plot_df = results_df.sort_values('50% Threshold')
    bars = plt.barh(plot_df['System'].str.upper(), plot_df['50% Threshold'], 
                    color=[COLORS.get(s, 'gray') for s in plot_df['System']])
    plt.axvline(x=0.05, color='black', linestyle='--', alpha=0.5)
    plt.text(0.052, -0.5, "Critical Threshold (ε=0.05)", fontsize=10, color='gray')
    plt.title('Robustness Threshold: Epsilon Where Accuracy Drops Below 50%', fontsize=14)
    plt.xlabel('Epsilon (Perturbation Budget)')
    plt.xlim(0, 0.45)
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_threshold_bars.png'), bbox_inches='tight')
    plt.close()

    # --- Figure 2: SDT Threshold Bar Chart ---
    plt.figure(figsize=(10, 6), dpi=150)
    plot_df = results_df.sort_values('DP_Threshold')
    plt.barh(plot_df['System'].str.upper(), plot_df['DP_Threshold'], 
             color=[COLORS.get(s, 'gray') for s in plot_df['System']])
    plt.axvline(x=0.05, color='black', linestyle='--', alpha=0.5)
    plt.title('Perceptual Threshold: Epsilon Where d\' Drops Below 1.0', fontsize=14)
    plt.xlabel('Epsilon (Perturbation Budget)')
    plt.xlim(0, 0.45)
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'sdt_threshold_bars.png'), bbox_inches='tight')
    plt.close()

    # --- Figure 3: Accuracy-Robustness Landscape ---
    plt.figure(figsize=(10, 8), dpi=150)
    for i, row in results_df.iterrows():
        sys = row['System']
        plt.scatter(row['Clean Acc'], row['50% Threshold'], 
                    color=COLORS.get(sys, 'gray'), s=200, edgecolors='black', zorder=5)
        plt.text(row['Clean Acc']+0.5, row['50% Threshold']+0.005, sys.upper(), 
                 fontsize=11, fontweight='bold')

    # Ideal diagonal
    plt.plot([80, 100], [0, 0.40], color='gray', linestyle=':', alpha=0.3, label='Ideal Tradeoff')
    plt.title('The Accuracy-Robustness Landscape', fontsize=16)
    plt.xlabel('Clean Accuracy (%)')
    plt.ylabel('Robustness Threshold (Epsilon @ 50%)')
    plt.grid(True, alpha=0.2)
    plt.xlim(80, 100)
    plt.ylim(-0.01, 0.45)
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_robustness_landscape.png'), bbox_inches='tight')
    plt.close()

    # --- Ranking Table ---
    print("\n" + "="*80)
    print("HEADLINE ROBUSTNESS RANKING")
    print("="*80)
    print(f"{'Model':<15} | {'Clean Acc':<10} | {'50% Threshold':<15} | {'DP=1.0 Thresh':<15}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        h_50 = ">0.30" if row['System'] == 'human' else f"{row['50% Threshold']:.4f}"
        h_dp = ">0.30" if row['System'] == 'human' else f"{row['DP_Threshold']:.4f}"
        print(f"{row['System'].upper():<15} | {row['Clean Acc']:<10.2f} | {h_50:<15} | {h_dp:<15}")
    print("="*80)
    
    print(f"\nHeadline figures saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
