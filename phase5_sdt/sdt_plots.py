import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RESULTS_CSV = 'phase5_sdt/results/sdt_results_v4.csv'
OUTPUT_DIR = 'phase5_sdt/figures'

# Project-wide color scheme
COLORS = {
    'Resnet': '#E94560',
    'Vit': '#7C3AED',
    'Efficientnet': '#0F3460',
    'Shaperesnet': '#16A34A',
    'Bagnet': '#F97316',
    'Human': '#22C55E'
}

def find_threshold_precise(epsilons, d_primes, threshold=1.0):
    epsilons = np.array(epsilons)
    d_primes = np.array(d_primes)
    sort_idx = np.argsort(epsilons)
    epsilons = epsilons[sort_idx]
    d_primes = d_primes[sort_idx]
    for i in range(len(d_primes) - 1):
        d1, d2 = d_primes[i], d_primes[i+1]
        e1, e2 = epsilons[i], epsilons[i+1]
        if (d1 >= threshold and d2 <= threshold) or (d1 <= threshold and d2 >= threshold):
            return e1 + (threshold - d1) * (e2 - e1) / (d2 - d1)
    return None

def plot_final_dprime_curves():
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} missing. Run sdt_analysis.py first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(12, 8), dpi=150)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label="Near-chance threshold (d'=1.0)")

    systems = df['system'].unique()
    
    for sys in systems:
        sys_df = df[df['system'] == sys].groupby('epsilon')['d_prime'].mean().reset_index()
        eps = sys_df['epsilon'].values
        dp = sys_df['d_prime'].values
        
        plt.plot(eps, dp, marker='o', label=sys, color=COLORS.get(sys, 'gray'), linewidth=2.5, markersize=7)
        
        # Add threshold annotation
        thresh = find_threshold_precise(eps, dp)
        if thresh is not None and sys != 'Human':
            plt.axvline(x=thresh, color=COLORS.get(sys, 'gray'), linestyle=':', alpha=0.4)
            plt.text(thresh, 0.2 + (0.3 * list(systems).index(sys)), f"ε={thresh:.3f}", 
                     color=COLORS.get(sys, 'gray'), fontweight='bold', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title("Signal Detection Theory: Perceptual Sensitivity Collapse (5/7 Models)", fontsize=16, pad=20)
    plt.xlabel("Perturbation Budget (Epsilon)", fontsize=14)
    plt.ylabel("Sensitivity Index (d')", fontsize=14)
    plt.xlim(-0.01, 0.35)
    plt.ylim(-0.5, 6.0)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=12, loc='upper right')
    
    # Shade the "blind zone"
    plt.fill_between([-0.05, 0.40], -1, 1.0, color='gray', alpha=0.05)

    out_path = os.path.join(OUTPUT_DIR, 'final_dprime_5model.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Final SDT figure saved: {out_path}")

if __name__ == '__main__':
    plot_final_dprime_curves()
