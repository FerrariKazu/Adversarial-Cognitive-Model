"""
Human Deep Analysis — Psychophysics Data Processing
===================================================

This script performs a multi-dimensional analysis of the human psychophysics 
results to identify patterns in human visual robustness.

ANALYSES PERFORMED:
1. Per-class sensitivity (Which objects are hardest for humans under noise?)
2. Metacognitive Calibration (Do humans know when they are wrong?)
3. Hardware Bias (Does viewing on a smartphone vs desktop impact robustness?)
4. Data Quality Control (Identifying low-effort participants)
5. Robustness Monotonicity (Does performance strictly degrade with noise?)

Outputs:
- Console report
- Figures in phase4_analysis/figures/human/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
COLOR_PALETTE = ["#1A1A2E", "#E94560", "#9D4EDD", "#0F3460"] # Consistent with project branding

# Paths
INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'human')

def load_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing mapped responses at {INPUT_CSV}. Run map_responses.py first.")
    df = pd.read_csv(INPUT_CSV)
    # Ensure numeric types
    df['epsilon'] = df['epsilon'].astype(float)
    df['confidence_rating'] = pd.to_numeric(df['confidence_rating'], errors='coerce')
    df['response_correct'] = df['response_correct'].astype(bool)
    return df

def analyze_reliability(df):
    """Flag participants with <50% accuracy on clean images (eps=0.0)."""
    print("\n[1] PARTICIPANT RELIABILITY ANALYSIS")
    print("-" * 40)
    
    clean_df = df[df['epsilon'] == 0.0]
    reliability = clean_df.groupby('participant_id')['response_correct'].mean() * 100
    
    flagged = reliability[reliability < 50]
    
    print(f"Total Participants: {len(reliability)}")
    if not flagged.empty:
        print(f"WARNING: {len(flagged)} participants flagged for low accuracy on clean images (<50%):")
        for pid, acc in flagged.items():
            print(f"  - {pid}: {acc:.1f}% accuracy")
    else:
        print("✅ All participants exceed the 50% clean-image reliability threshold.")
        
    return flagged.index.tolist()

def analyze_non_monotonicity(df):
    """Check if accuracy strictly decreases with epsilon."""
    print("\n[2] ROBUSTNESS MONOTONICITY CHECK")
    print("-" * 40)
    
    # Calculate global accuracy per epsilon
    mono = df.groupby('epsilon')['response_correct'].mean().sort_index()
    
    violations = []
    prev_eps = None
    prev_acc = None
    
    for eps, acc in mono.items():
        if prev_acc is not None:
            if acc > prev_acc:
                violations.append((prev_eps, eps, prev_acc, acc))
        prev_eps = eps
        prev_acc = acc
        
    if violations:
        print("⚠️ NON-MONOTONICITY DETECTED:")
        for v in violations:
            print(f"  Accuracy increased from {v[2]*100:.1f}% to {v[3]*100:.1f}% "
                  f"when epsilon increased from {v[0]:.2f} to {v[1]:.2f}")
    else:
        print("✅ Performance is strictly monotonic (decreases as noise increases).")

def plot_class_heatmap(df):
    """Per-class accuracy heatmap across epsilon levels."""
    print("\n[3] GENERATING CLASS HEATMAP...")
    
    # Create pivot table: Rows=True Class, Cols=Epsilon
    pivot = df.pivot_table(index='true_class', columns='epsilon', values='response_correct', aggfunc='mean')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
    plt.title("Human Accuracy per Class across Epsilon Levels", fontsize=14, pad=20)
    plt.ylabel("Target Class")
    plt.xlabel("Epsilon (PGD Perturbation)")
    
    save_path = os.path.join(FIGURE_DIR, 'human_class_heatmap.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  - Saved to {save_path}")

def plot_calibration(df):
    """Confidence-Accuracy Calibration Analysis."""
    print("\n[4] GENERATING CALIBRATION PLOT...")
    
    # Bin confidence ratings
    bins = [0, 3.5, 6.5, 8.5, 10.5]
    labels = ['1-3 (Low)', '4-6 (Mid)', '7-8 (High)', '9-10 (Certain)']
    df['conf_bin'] = pd.cut(df['confidence_rating'], bins=bins, labels=labels)
    
    # Calculate accuracy per bin
    calibration = df.groupby('conf_bin')['response_correct'].agg(['mean', 'count', 'std'])
    calibration['error'] = calibration['std'] / np.sqrt(calibration['count'])
    
    plt.figure(figsize=(8, 6))
    plt.bar(calibration.index, calibration['mean'] * 100, color=COLOR_PALETTE[2], alpha=0.8)
    plt.errorbar(calibration.index, calibration['mean'] * 100, yerr=calibration['error']*100, fmt='none', color='black', capsize=5)
    
    # Perfect calibration line would follow a diagonal, but here we just show labels
    plt.title("Metacognitive Calibration: Confidence vs Actual Accuracy", fontsize=14)
    plt.ylabel("Actual Accuracy (%)")
    plt.xlabel("Subjective Confidence Rating")
    plt.ylim(0, 100)
    
    # Overlay count labels
    for i, row in enumerate(calibration.iterrows()):
        plt.text(i, 5, f"n={int(row[1]['count'])}", ha='center', color='white', fontweight='bold')

    save_path = os.path.join(FIGURE_DIR, 'human_calibration.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  - Saved to {save_path}")

def plot_device_comparison(df):
    """Compare performance across device types."""
    print("\n[5] GENERATING DEVICE COMPARISON...")
    
    # Clean up device names if necessary (e.g. "Smartphone", "Desktop")
    device_perf = df.groupby(['epsilon', 'device'])['response_correct'].mean().reset_index()
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=device_perf, x='epsilon', y='response_correct', hue='device', marker='o', palette=COLOR_PALETTE[:3])
    
    plt.title("Human Robustness: Smartphone vs Desktop Users", fontsize=14)
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.ylim(0, 1.0)
    
    save_path = os.path.join(FIGURE_DIR, 'device_comparison.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  - Saved to {save_path}")

def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. QC Check
    flagged_pids = analyze_reliability(df)
    
    # 2. Monotonicity check
    analyze_non_monotonicity(df)
    
    # 3. Visualization
    plot_class_heatmap(df)
    plot_calibration(df)
    plot_device_comparison(df)
    
    print("\n" + "="*40)
    print("ANALYSIS COMPLETE")
    print("="*40)

if __name__ == '__main__':
    main()
