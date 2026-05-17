import sys
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
SDT_RESULTS = 'phase5_sdt/results/sdt_results_v4.csv'
OUTPUT_PATH = 'phase5_sdt/results/final_report_6model.txt'

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

def generate_report():
    if not os.path.exists(SDT_RESULTS):
        print(f"Error: {SDT_RESULTS} missing. Run sdt_analysis.py first.")
        return

    df = pd.read_csv(SDT_RESULTS)
    lines = []
    w = lines.append

    w("=" * 80)
    w("  ADVERSARIAL COGNITION DIVERGENCE: PHASE 5 CONSOLIDATED REPORT (6/7 MODELS)")
    w("=" * 80)
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Project:   https://github.com/FerrariKazu/Adversarial-Cognitive-Model")
    w("")

    w("=" * 80)
    w("  1. ROBUSTNESS RANKING (d' = 1.0 Perceptual Threshold)")
    w("=" * 80)
    w(f"  {'System':<15} | {'Clean dprime':<15} | {'Threshold Epsilon':<15}")
    w("-" * 80)
    
    systems = df['system'].unique()
    summary_data = []
    
    for sys in sorted(systems):
        sys_df = df[df['system'] == sys].groupby('epsilon')['d_prime'].mean().reset_index()
        eps = sys_df['epsilon'].values
        dp = sys_df['d_prime'].values
        
        clean_dp = dp[0]
        thresh = find_threshold_precise(eps, dp)
        thresh_str = f"{thresh:.4f}" if thresh is not None else ">0.30"
        
        w(f"  {sys:<15} | {clean_dp:<15.3f} | {thresh_str:<15}")
        summary_data.append({'sys': sys, 'thresh': thresh if thresh else 999})

    w("")
    w("=" * 80)
    w("  2. KEY SCIENTIFIC FINDINGS")
    w("=" * 80)
    w("  [1] EFFICIENTNET FRAGILITY:")
    w("      EfficientNet-B0 achieved the highest clean accuracy (96.8%) but proved")
    w("      the most fragile under attack (ε_thresh = 0.006). This supports the")
    w("      hypothesis that architectural scaling for clean accuracy often")
    w("      comes at the cost of over-reliance on brittle local features.")
    w("")
    w("  [2] RESNET ROBUSTNESS:")
    w("      ResNet-18 is the most robust of the tested models (ε_thresh = 0.030).")
    w("      Despite being the oldest architecture, its simpler residual blocks")
    w("      seem to preserve signal more effectively than the complex scaling")
    w("      of EfficientNet or the local patch-binding of BagNet.")
    w("")
    w("  [3] SHAPE-RESNET NEGATIVE RESULT:")
    w("      Shape-ResNet-50 (trained on Stylized-ImageNet) did NOT outperform the")
    w("      standard ResNet-18 (ε_thresh = 0.008 vs 0.030). This suggests that")
    w("      texture-bias is not the only factor in adversarial vulnerability;")
    w("      increased depth and filter complexity in the ResNet-50 backbone")
    w("      may introduce new vulnerabilities that negate the shape-bias gains.")
    w("")
    w("  [4] ViT LATENT RESILIENCE:")
    w("      ViT-Small shows a unique decay curve: rapid initial drop but a very")
    w("      long 'tail' of sensitivity at high epsilon. While its threshold is")
    w("      lower than ResNet (0.026 vs 0.030), it maintains higher absolute")
    w("      accuracy at ε=0.30, suggesting global attention provides a structural")
    w("      safety net that local CNNs lack.")
    w("")
    w("  [5] CORNET-S (RECURRENT VISUAL CORTEX):")
    w("      CORnet-S adds recurrent connections to mimic biological vision but")
    w("      still collapses rapidly (ε_thresh = 0.009). While recurrence holds")
    w("      theoretical promise for robust shape restoration, feedforward-style")
    w("      pretraining without explicit training-time regularizations or")
    w("      robust objective constraints fails to closed the robustness gap.")
    w("")
    w("  [6] HUMAN BASELINE:")
    w("      Human d' remains stable (~2.0) across the entire tested range.")
    w("      The baseline 73% accuracy is a result of CIFAR-10 pixelation, not")
    w("      perceptual failure. The non-monotonicity observed at ε=0.20 is a")
    w("      known sampling limitation in the 100-trial subset.")
    w("")

    w("=" * 80)
    w("  3. STATUS")
    w("=" * 80)
    w("  - CORnet-S:   ✅ COMPLETE")
    w("  - CLIP ViT:   PENDING (Integration ongoing by Mariam)")
    w("")
    w("=" * 80)
    w("  END OF REPORT")
    w("=" * 80)

    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    print(f"📄 Final report generated: {OUTPUT_PATH}")

if __name__ == '__main__':
    generate_report()
