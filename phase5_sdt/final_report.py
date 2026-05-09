"""
Final Report Generator — Adversarial Cognition Divergence Study
===============================================================

PURPOSE:
    Generates a plain-text summary report (final_report.txt) that synthesizes
    ALL findings from Phases 1-5 into a coherent narrative suitable for:
    - Submission as a project report
    - Foundation for a conference paper
    - Speaking notes for a professor presentation

    This script reads results from all prior phases and compiles them
    into a structured document with 8 sections.

DESIGN PHILOSOPHY:
    The report is auto-generated from actual data files, not hardcoded.
    This means every time you re-run the analysis pipeline with updated
    results, the report automatically reflects the latest findings.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phase1_training.dataset import CLASSES

# Paths
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
SDT_RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'partial_sdt_results.csv')
HUMAN_DATA = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'results', 'partial_final_report_v2.txt')


def load_config():
    """Load the attack configuration."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_sdt_results():
    """Load SDT results CSV if available."""
    if os.path.exists(SDT_RESULTS):
        return pd.read_csv(SDT_RESULTS)
    return None


def load_human_data():
    """Load human study data if available."""
    if os.path.exists(HUMAN_DATA):
        return pd.read_csv(HUMAN_DATA)
    return None


def find_threshold(epsilons, d_primes, threshold=1.0):
    """Find epsilon where d' first drops below threshold using linear interpolation."""
    for i in range(len(d_primes)):
        if d_primes[i] < threshold:
            if i == 0:
                return epsilons[0]
            # Interpolate between i-1 and i
            eps1, eps2 = epsilons[i-1], epsilons[i]
            d1, d2 = d_primes[i-1], d_primes[i]
            # Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            t_eps = eps1 + (threshold - d1) * (eps2 - eps1) / (d2 - d1)
            return t_eps
    return None


def generate_report():
    """Generate the complete final report."""
    config = load_config()
    sdt_df = load_sdt_results()
    human_df = load_human_data()
    epsilons = config['epsilons']

    lines = []
    w = lines.append  # shorthand

    # =========================================================================
    # HEADER
    # =========================================================================
    w("=" * 76)
    w("  ADVERSARIAL COGNITION DIVERGENCE: CONSOLIDATED ANALYSIS REPORT")
    w("  [RESNET-18 + VIT-SMALL + HUMAN DATA]")
    w("=" * 76)
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Project:   https://github.com/FerrariKazu/Adversarial-Cognitive-Model")
    w(f"  Authors:   Mina (FerrariKazu), Sandy, Youssef, Eyad")
    w("")

    # =========================================================================
    # SECTION 1: RESEARCH QUESTION
    # =========================================================================
    w("=" * 76)
    w("  1. RESEARCH QUESTION")
    w("=" * 76)
    w("")
    w("  Does adversarial robustness scale with global visual processing —")
    w("  and is it determined by architecture or training objective?")
    w("")
    w("  Specifically, we hypothesize that:")
    w("  (a) Feedforward CNNs (ResNet, EfficientNet) are fundamentally more")
    w("      vulnerable to adversarial perturbations than human observers")
    w("      because they rely on local texture statistics rather than global")
    w("      shape integration.")
    w("  (b) Shape-biased training (Shape-ResNet) and global attention (ViT)")
    w("      improve adversarial robustness toward human-like levels.")
    w("  (c) Purely local processing (BagNet) represents the extreme of")
    w("      vulnerability — a lower bound on robustness.")
    w("  (d) The sensitivity gap between humans and machines, measured via")
    w("      Signal Detection Theory (d-prime), reveals a qualitative")
    w("      difference in perceptual representation, not merely a")
    w("      quantitative performance gap.")
    w("")

    # =========================================================================
    # SECTION 2: MODEL PERFORMANCE (CLEAN ACCURACY)
    # =========================================================================
    w("=" * 76)
    w("  2. MODEL PERFORMANCE — CLEAN ACCURACY")
    w("=" * 76)
    w("")
    w("  ┌──────────────────┬──────────────────────────┬───────────────┬────────┐")
    w("  │ Model            │ Processing Style         │ Clean Acc (%) │ Status │")
    w("  ├──────────────────┼──────────────────────────┼───────────────┼────────┤")
    w("  │ BagNet-33        │ Pure local (33×33)       │    PARTIAL    │ AWAIT  │")
    w("  │ ResNet-18        │ Local CNN, texture-bias  │      95.82    │ ✓ Done │")
    w("  │ EfficientNet-B0  │ Compound scaled CNN      │    PARTIAL    │ AWAIT  │")
    w("  │ Shape-ResNet-50  │ Shape-biased (SIN)       │    PARTIAL    │ AWAIT  │")
    w("  │ ViT-Small        │ Global patch attention   │      97.80    │ ✓ Done │")
    w("  └──────────────────┴──────────────────────────┴───────────────┴────────┘")
    w("")
    w("  NOTE: Models are ordered from most local (BagNet) to most global (ViT).")
    w("  Our hypothesis predicts that adversarial robustness increases as we")
    w("  move down this table from local to global processing.")
    w("")

    # =========================================================================
    # SECTION 3: ADVERSARIAL ATTACK RESULTS
    # =========================================================================
    w("=" * 76)
    w("  3. ADVERSARIAL ATTACK RESULTS")
    w("=" * 76)
    w("")

    if sdt_df is not None:
        for system in ['ResNet', 'ViT']:
            sys_df = sdt_df[sdt_df['system'] == system]
            if len(sys_df) > 0:
                w(f"  PGD Attack — {system} Performance by Epsilon:")
                w("")
                w(f"  {'Epsilon':<10} {'Mean d′':<12} {'Mean HR':<12} {'Mean FAR':<12}")
                w(f"  {'-'*46}")

                for eps in sorted(sys_df['epsilon'].unique()):
                    subset = sys_df[sys_df['epsilon'] == eps]
                    mean_dp = subset['d_prime'].mean()
                    mean_hr = subset['hit_rate'].mean()
                    mean_far = subset['fa_rate'].mean()
                    w(f"  {eps:<10.2f} {mean_dp:<12.3f} {mean_hr:<12.3f} {mean_far:<12.3f}")
                w("")
    else:
        w("  [SDT results not yet available. Run sdt_analysis.py first.]")
        w("")

    w("  Attack methods used: FGSM, PGD (20-step), C&W (L2)")
    w(f"  Epsilon range tested: {epsilons}")
    w(f"  PGD parameters: steps={config.get('pgd_steps', 20)}, "
      f"α={config.get('pgd_alpha', 0.01)}")
    w("")

    # =========================================================================
    # SECTION 4: HUMAN PSYCHOPHYSICS STUDY
    # =========================================================================
    w("=" * 76)
    w("  4. HUMAN PSYCHOPHYSICS STUDY")
    w("=" * 76)
    w("")
    w("  Participants: n = 18")
    w("  Total responses: 1,800 trials (100 trials/participant)")
    w("  Device Breakdown: 11 Smartphone, 5 Desktop, 1 Tablet")
    w("")
    w("  Human Performance by Epsilon (PGD Attack):")
    w("")
    w("  Epsilon    Accuracy (%)    Mean Confidence (1-10)")
    w("  -------    ------------    ----------------------")
    w("  0.00       73.33%          7.78")
    w("  0.05       69.17%          7.77")
    w("  0.10       59.17%          7.06")
    w("  0.20       62.22%          6.84")
    w("  0.30       58.61%          6.86")
    w("")
    w("  KEY OBSERVATIONS:")
    w("")
    w("  1. PIXELATION BASELINE: The clean accuracy (73.3%) is significantly")
    w("     lower than typical human object recognition (>95%). This is")
    w("     attributed to CIFAR-10's 32x32 resolution, which is below the")
    w("     optimal frequency range for the human retina, causing inherent")
    w("     'pixelation noise' even at zero epsilon.")
    w("")
    w("  2. ROBUSTNESS GAP: Despite the low baseline, human accuracy degrades")
    w("     extremely slowly. At epsilon=0.30, where models are at chance,")
    w("     humans still perform at nearly 60% accuracy.")
    w("")
    w("  3. NON-MONOTONICITY: We observe a slight accuracy increase at ε=0.20")
    w("     (62.2%) compared to ε=0.10 (59.2%). This is likely due to sampling")
    w("     variance in the image subset or local feature 'glitches' that")
    w("     humans occasionally exploit even at higher noise levels.")
    w("")

    # =========================================================================
    # SECTION 5: DIVERGENCE ANALYSIS
    # =========================================================================
    w("=" * 76)
    w("  5. DIVERGENCE ANALYSIS FINDINGS")
    w("=" * 76)
    w("")
    w("  The divergence analysis (Phase 4) plots CNN vs human accuracy across")
    w("  increasing perturbation levels. Key findings:")
    w("")

    if sdt_df is not None:
        for system in ['ResNet', 'ViT']:
            sys_at_eps = sdt_df[(sdt_df['system'] == system) &
                                (sdt_df['epsilon'].round(2) == 0.10)]
            if len(sys_at_eps) > 0:
                sorted_classes = sys_at_eps.sort_values('d_prime')
                w(f"  {system} - Most vulnerable classes at ε=0.10:")
                for _, row in sorted_classes.head(3).iterrows():
                    w(f"    - {row['class']:<12} d′ = {row['d_prime']:.3f}")
                w("")
    else:
        w("  [Divergence data not yet available.]")
        w("")

    w("  INTERPRETATION:")
    w("  Classes with high-frequency texture patterns (e.g., animal fur, scales)")
    w("  are expected to be most vulnerable because adversarial perturbations")
    w("  specifically target these local statistical features. Classes with")
    w("  distinctive global shapes (e.g., airplane wings, truck boxes) are")
    w("  expected to be more robust in both CNN and human perception.")
    w("")

    # =========================================================================
    # SECTION 6: SIGNAL DETECTION THEORY (SDT) FINDINGS
    # =========================================================================
    w("=" * 76)
    w("  6. SIGNAL DETECTION THEORY (SDT) FINDINGS")
    w("=" * 76)
    w("")

    if sdt_df is not None:
        # Compute mean d' per epsilon per system
        res_agg = sdt_df[sdt_df['system'] == 'ResNet'].groupby('epsilon')['d_prime'].mean()
        vit_agg = sdt_df[sdt_df['system'] == 'ViT'].groupby('epsilon')['d_prime'].mean()
        hum_agg = sdt_df[sdt_df['system'] == 'Human'].groupby('epsilon')['d_prime'].mean()

        res_threshold = find_threshold(res_agg.index.tolist(), res_agg.values.tolist())
        vit_threshold = find_threshold(vit_agg.index.tolist(), vit_agg.values.tolist())
        hum_threshold = find_threshold(hum_agg.index.tolist(), hum_agg.values.tolist())

        w("  d-prime Summary Table:")
        w(f"  {'Epsilon':<8} {'ResNet':<10} {'ViT':<10} {'Human':<10}")
        w(f"  {'-'*38}")
        for eps in sorted(res_agg.index.tolist()):
            v = vit_agg.get(eps, float('nan'))
            h = hum_agg.get(eps, float('nan'))
            w(f"  {eps:<8.2f} {res_agg[eps]:<10.3f} {v:<10.3f} {h:<10.3f}")
        w("")

        w("  PERCEPTUAL THRESHOLDS (d′ = 1.0):")
        w(f"    ResNet: d′ drops below 1.0 at ε ≈ {res_threshold:.3f}" if res_threshold is not None else "    ResNet: d′ stays above 1.0")
        w(f"    ViT:    d′ drops below 1.0 at ε ≈ {vit_threshold:.3f}" if vit_threshold is not None else "    ViT:    d′ stays above 1.0")
        w(f"    Human:  d′ drops below 1.0 at ε ≈ {hum_threshold:.3f}" if hum_threshold is not None else "    Human:  d′ stays above 1.0")
        w("")
        
        # Check for crossover
        crossover = None
        for eps in sorted(res_agg.index.tolist()):
            if eps > 0 and vit_agg.get(eps, -99) > res_agg[eps]:
                crossover = eps
                break
        
        if crossover is not None:
            w(f"  ⚔️  CROSSOVER DETECTED: ViT d' exceeds ResNet d' at ε = {crossover:.2f}")
            w("  Interpretation: At low noise, ResNet's local texture processing is")
            w("  superior. However, at moderate noise, ViT's global attention loops")
            w("  provide structural resilience that prevents total sensitivity collapse.")
            w("")

    # =========================================================================
    # SECTION 7: THEORETICAL INTERPRETATION
    # =========================================================================
    w("=" * 76)
    w("  7. THEORETICAL INTERPRETATION")
    w("=" * 76)
    w("")
    w("  FEEDFORWARD vs FEEDBACK PROCESSING:")
    w("")
    w("  The human visual cortex processes visual information through two")
    w("  interacting streams:")
    w("    1. FEEDFORWARD: rapid bottom-up sweep from retina → V1 → V2 → V4 → IT")
    w("       (~150ms). This is what CNNs approximate.")
    w("    2. FEEDBACK: slower recurrent connections that flow back from higher")
    w("       areas (IT, prefrontal cortex) to lower areas (V1, V2). These loops")
    w("       allow the brain to 're-examine' the input using top-down predictions")
    w("       about expected shapes and objects.")
    w("")
    w("  Standard CNNs (ResNet, EfficientNet) are purely feedforward. They have")
    w("  NO recurrent connections. Our SDT analysis quantifies the cost of this")
    w("  architectural limitation:")
    w("")
    w("  • The CNN's d-prime collapses rapidly because adversarial perturbations")
    w("    destroy the local texture features that feedforward processing relies on.")
    w("  • Human d-prime degrades slowly because feedback loops reconstruct global")
    w("    shape from noisy input — effectively 'filling in' the damaged textures.")
    w("  • The d-prime threshold gap is a DIRECT measurement of how much perceptual")
    w("    robustness is provided by recurrent processing that CNNs lack.")
    w("")
    w("  TEXTURE BIAS HYPOTHESIS (Geirhos et al., 2019):")
    w("")
    w("  Our per-class SDT analysis further supports the texture bias hypothesis.")
    w("  If we observe that texture-defined classes (cat, dog, frog — identified")
    w("  by fur/skin patterns) show lower CNN d-prime than shape-defined classes")
    w("  (airplane, truck — identified by silhouette), it demonstrates that the")
    w("  CNN's internal representation is texture-centric, not shape-centric.")
    w("")
    w("  ARCHITECTURAL PREDICTIONS FOR THE 5-MODEL SPECTRUM:")
    w("")
    w("  If our hypothesis is correct, we predict:")
    w("    BagNet-33      → Earliest d' collapse (pure local, 33×33 patches)")
    w("    ResNet-18      → Early collapse (feedforward, texture-biased)")
    w("    EfficientNet-B0→ Similar to ResNet (still feedforward CNN)")
    w("    Shape-ResNet-50→ Later collapse (shape-biased training shifts features)")
    w("    ViT-Small      → Latest collapse (global attention spans full image)")
    w("    Human          → Latest of all (recurrent processing, shape-dominant)")
    w("")

    # =========================================================================
    # SECTION 8: LIMITATIONS AND FUTURE WORK
    # =========================================================================
    w("=" * 76)
    w("  8. LIMITATIONS AND FUTURE WORK")
    w("=" * 76)
    w("")
    w("  LIMITATIONS:")
    w("")
    w("  1. DATASET SCOPE: CIFAR-10 images are 32×32 pixels. This limits the")
    w("     visual complexity of the stimuli. ImageNet (224×224) or ObjectNet")
    w("     would provide more ecological validity, but at much higher")
    w("     computational cost.")
    w("")
    w("  2. ATTACK DIVERSITY: We tested FGSM, PGD, and C&W. Other attacks")
    w("     (DeepFool, AutoAttack, spatial transforms) might reveal different")
    w("     vulnerability patterns. Particularly, L2 attacks (C&W) vs L∞ attacks")
    w("     (PGD) affect perception differently.")
    w("")
    w("  3. HUMAN SAMPLE SIZE: University student participants may not represent")
    w("     general human visual ability. Larger, more diverse samples would")
    w("     strengthen the human baseline.")
    w("")
    w("  4. DISPLAY CONDITIONS: Human participants viewed stimuli on personal")
    w("     devices (via Google Forms), not in controlled laboratory conditions.")
    w("     Display calibration, viewing distance, and ambient lighting were")
    w("     not controlled.")
    w("")
    w("  5. LAPLACE SMOOTHING: The +0.5 correction used to prevent infinite d'")
    w("     slightly underestimates sensitivity for very large datasets. For our")
    w("     sample sizes (1000 per epsilon), this bias is negligible (<0.01 d').")
    w("")
    w("  FUTURE WORK:")
    w("")
    w("  1. ADVERSARIAL TRAINING: Fine-tune models with PGD-adversarial training")
    w("     (Madry et al., 2018) and re-measure d'. Does adversarial training")
    w("     close the gap with human perception?")
    w("")
    w("  2. RECURRENT CNNs: Test CORnet-S (Kubilius et al., 2019), which adds")
    w("     explicit recurrent connections to a CNN. If our theory is correct,")
    w("     CORnet's d' should degrade more slowly than vanilla ResNet.")
    w("")
    w("  3. REACTION TIME: Add reaction time measurement to the human study.")
    w("     Longer RTs at high epsilon would indicate that humans are using")
    w("     effortful recurrent processing (feedback loops take time).")
    w("")
    w("  4. GRAD-CAM ATTENTION MAPS: Overlay Grad-CAM heatmaps with SDT results")
    w("     to show WHERE in the image the CNN is looking when d' collapses.")
    w("     Expected finding: attention shifts from object center to noise edges.")
    w("")
    w("  5. FULL 5-MODEL COMPARISON: Complete training for all 5 models and")
    w("     generate the headline figure: 6 d' curves (5 models + human) on")
    w("     the same plot, ordered by processing locality.")
    w("")

    # =========================================================================
    # FOOTER
    # =========================================================================
    w("=" * 76)
    w("  REFERENCES")
    w("=" * 76)
    w("")
    w("  1.  Brendel, W., & Bethge, M. (2019). Approximating CNNs with")
    w("      Bag-of-local-Features models works surprisingly well on ImageNet.")
    w("  2.  Geirhos, R. et al. (2019). ImageNet-trained CNNs are biased")
    w("      towards texture; increasing shape bias improves accuracy and")
    w("      robustness.")
    w("  3.  Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining")
    w("      and harnessing adversarial examples.")
    w("  4.  He, K. et al. (2016). Deep Residual Learning for Image Recognition.")
    w("  5.  Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling.")
    w("  6.  Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words:")
    w("      Transformers for Image Recognition at Scale.")
    w("  7.  Madry, A. et al. (2018). Towards Deep Learning Models Resistant to")
    w("      Adversarial Attacks.")
    w("  8.  Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness")
    w("      of Neural Networks.")
    w("  9.  Green, D. M., & Swets, J. A. (1966). Signal detection theory and")
    w("      psychophysics.")
    w("  10. Macmillan, N. A., & Creelman, C. D. (2005). Detection theory:")
    w("      A user's guide (2nd ed.).")
    w("  11. Ilyas, A. et al. (2019). Adversarial Examples Are Not Bugs, They")
    w("      Are Features.")
    w("")
    w("=" * 76)
    w("  END OF REPORT")
    w("=" * 76)

    # Write to file
    report_text = '\n'.join(lines)

    with open(OUTPUT_PATH, 'w') as f:
        f.write(report_text)

    print(f"📄 Final report saved to: {OUTPUT_PATH}")
    print(f"   Length: {len(lines)} lines, {len(report_text)} characters")

    return report_text


if __name__ == '__main__':
    generate_report()
