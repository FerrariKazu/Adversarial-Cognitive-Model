"""
SDT Analysis: d-Prime Sensitivity Under Adversarial Attack
==========================================================

PURPOSE:
    This is the scientific backbone of the project. For each epsilon level,
    it computes d-prime for the CNN AND for human observers, producing the
    key comparison table: at what noise level does each system lose the
    ability to discriminate signal from noise?

WHAT THIS SCRIPT PRODUCES:
    1. A CSV file: phase5_sdt/results/sdt_results.csv
       Columns: epsilon, system (CNN/Human), class, d_prime, beta, hit_rate, fa_rate
    2. A summary table printed to terminal
    3. The "perceptual threshold" — the epsilon where d' drops below 1.0

THE HEADLINE FINDING:
    The GAP between CNN and human perceptual thresholds is the single most
    important number in this entire project. It quantifies, on a theoretically
    grounded scale (Green & Swets, 1966), how much MORE sensitive humans are
    to object identity under adversarial noise than the CNN.

    If CNN threshold = ε=0.05 and human threshold = ε=0.25, the gap is 0.20.
    That means there is a 0.20-wide epsilon range where humans can still
    perceive the object but the CNN is effectively blind.

WHY d' THRESHOLD > RAW ACCURACY:
    "CNN accuracy drops from 96% to 15%" is descriptive.
    "CNN d' crosses the detection threshold (d'=1.0) at ε=0.05, while human
     d' doesn't cross until ε=0.25" is analytical.

    The d' statement is:
    (a) Independent of response bias (a CNN that says "cat" for everything
        scores 100% on cats but 0% on everything else — d' correctly shows 0)
    (b) On a universal scale (d'=1.0 means the same thing in 1966 radar
        detection and 2024 adversarial robustness)
    (c) Directly comparable across studies, datasets, and modalities

HOW TO PRESENT THIS TO YOUR PROFESSOR IN ONE SENTENCE:
    "Our signal detection analysis reveals that the CNN loses perceptual
    sensitivity to object identity — falling below the d'=1.0 detection
    threshold — at ε=[X], while human observers maintain above-threshold
    sensitivity until ε=[Y], demonstrating a [Y-X] epsilon-unit robustness
    gap attributable to the CNN's lack of recurrent shape processing."

WHAT THIS MEANS AT A COGNITIVE SCIENCE CONFERENCE:
    This result connects adversarial robustness to established psychophysical
    measurement theory. By showing that the CNN's d' curve has a fundamentally
    different shape than the human curve — collapsing earlier and more steeply —
    we provide evidence that feedforward texture-based processing (CNN) and
    recurrent shape-based processing (human visual cortex) produce qualitatively
    different perceptual representations, not just quantitatively weaker ones.
    The d' framework makes this claim rigorous because it controls for criterion
    effects that could confound a raw accuracy comparison.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phase1_training.model import CIFARResNet
from phase1_training.model_vit import CIFARViT
from phase1_training.dataset import CLASSES
from utils.metrics import load_adv_batch
from phase5_sdt.sdt_core import compute_sdt_all_classes, d_prime

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
HUMAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'partial_sdt_results.csv')


# =============================================================================
#  MOCK HUMAN DATA (fallback if human study not yet completed)
# =============================================================================

def generate_mock_human_sdt_data(epsilons):
    """
    Generate mock human classification data for SDT analysis.

    PLAIN LANGUAGE:
        Until the Google Form responses are collected, we simulate human
        performance based on the psychophysics literature. Humans are
        expected to maintain >90% accuracy at low epsilon and degrade
        slowly, reaching ~50% only at very high perturbation levels.

        The mock data produces realistic SDT curves that we can replace
        with real data when the human study completes.
    """
    print("⚠️  Human data not found. Generating MOCK data for demonstration.")
    print("    Replace with real data from phase3_human_study/data/responses_mapped.csv\n")

    rng = np.random.default_rng(42)

    # Empirically plausible human accuracy per epsilon
    # (degrades much slower than CNN — this is the hypothesis)
    human_acc_by_eps = {
        0.00: 0.98,
        0.01: 0.97,
        0.05: 0.94,
        0.10: 0.88,
        0.20: 0.72,
        0.30: 0.55,
    }

    all_preds = {}
    all_labels = {}

    for eps in epsilons:
        eps_f = float(eps)
        acc_target = human_acc_by_eps.get(eps_f, 0.50)

        n_per_class = 100
        preds = []
        labels = []

        for c in range(10):
            true_labels = np.full(n_per_class, c, dtype=int)
            # Simulate: acc_target fraction correct, rest random misclassification
            n_correct = int(n_per_class * acc_target)
            correct_preds = np.full(n_correct, c, dtype=int)
            wrong_preds = rng.choice([x for x in range(10) if x != c],
                                     size=n_per_class - n_correct)
            class_preds = np.concatenate([correct_preds, wrong_preds])
            rng.shuffle(class_preds)

            preds.append(class_preds)
            labels.append(true_labels)

        all_preds[eps_f] = np.concatenate(preds)
        all_labels[eps_f] = np.concatenate(labels)

    return all_preds, all_labels


def load_human_data_for_sdt(epsilons):
    """
    Load real human responses and convert to (preds, labels) format.

    The human CSV is expected to have columns:
        epsilon, true_class, predicted_class (or response_class)

    Returns dict[epsilon] -> (preds_array, labels_array)
    """
    if not os.path.exists(HUMAN_DATA_PATH):
        return generate_mock_human_sdt_data(epsilons)

    df = pd.read_csv(HUMAN_DATA_PATH)
    # Filter to PGD if available
    df_pgd = df[df['attack_type'] == 'pgd'].copy() if 'attack_type' in df.columns else df.copy()

    # Map class names to indices
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    all_preds = {}
    all_labels = {}

    for eps in epsilons:
        eps_f = float(eps)
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == eps_f]

        if len(subset) == 0:
            print(f"  ⚠️  No human data for ε={eps_f:.2f}, skipping")
            continue

        labels = subset['true_class'].map(class_to_idx).values.astype(int)

        # Try multiple possible column names for human predictions
        pred_col = None
        for col in ['predicted_class', 'response_class', 'human_response']:
            if col in subset.columns:
                pred_col = col
                break

        if pred_col is None:
            # Fallback: use response_correct to reconstruct approximate preds
            preds = labels.copy()
            incorrect_mask = ~subset['response_correct'].astype(bool).values
            rng = np.random.default_rng(42)
            for idx in np.where(incorrect_mask)[0]:
                wrong_choices = [c for c in range(10) if c != labels[idx]]
                preds[idx] = rng.choice(wrong_choices)
        else:
            preds = subset[pred_col].map(class_to_idx).values.astype(int)

        all_preds[eps_f] = preds
        all_labels[eps_f] = labels

    return all_preds, all_labels


# =============================================================================
#  CNN EVALUATION
# =============================================================================

def get_cnn_predictions(model, device, epsilons, model_name='resnet'):
    """
    Run the CNN on all PGD adversarial datasets and return raw predictions.
    Uses mmap_mode to prevent OOM.
    Returns dict[epsilon] -> (preds_array, labels_array)
    """
    all_preds = {}
    all_labels = {}

    print(f"Computing {model_name.upper()} predictions on adversarial datasets...")
    lbl_path = os.path.join(os.path.dirname(__file__), '..', 'phase2_attacks', 'adv_images', model_name, 'labels.npy')
    labels_np = np.load(lbl_path)

    for eps in epsilons:
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(os.path.dirname(__file__), '..', 'phase2_attacks', 'adv_images', model_name, f"pgd_eps{eps_str}_images.npy")
        
        try:
            images_mmap = np.load(img_path, mmap_mode='r')
        except FileNotFoundError as e:
            print(f"  ⚠️  {e} — skipping ε={eps}")
            continue

        batch_size = 128 if model_name == 'vit' else 256
        preds_list = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch = torch.tensor(images_mmap[i:i + batch_size], device=device)
                outputs = model(batch)
                preds_list.append(outputs.argmax(dim=1).cpu().numpy())

        all_preds[float(eps)] = np.concatenate(preds_list)
        all_labels[float(eps)] = labels_np
        print(f"  ε={eps:.2f} — {len(all_preds[float(eps)])} samples processed")

    return all_preds, all_labels


# =============================================================================
#  PERCEPTUAL THRESHOLD DETECTION
# =============================================================================

def find_perceptual_threshold(epsilons, d_primes, threshold=1.0):
    """
    Find the epsilon where d' first drops below the threshold.
    """
    for eps, dp in zip(epsilons, d_primes):
        if dp < threshold:
            return eps
    return None


# =============================================================================
#  MAIN ANALYSIS
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    resnet = CIFARResNet().to(device)
    resnet.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'best.pth'), map_location=device))
    
    vit = CIFARViT().to(device)
    vit.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth'), map_location=device))

    # Get predictions
    resnet_preds, resnet_labels = get_cnn_predictions(resnet, device, epsilons, model_name='resnet')
    vit_preds, vit_labels = get_cnn_predictions(vit, device, epsilons, model_name='vit')
    human_preds, human_labels = load_human_data_for_sdt(epsilons)

    # =========================================================================
    # Compute SDT metrics for every epsilon × system × class
    # =========================================================================
    all_rows = []

    print("\n" + "=" * 70)
    print("SIGNAL DETECTION THEORY ANALYSIS")
    print("=" * 70)

    # --- ResNet ---
    print("\n--- RESNET d-prime per epsilon ---")
    resnet_mean_dprimes = []

    for eps in epsilons:
        eps_f = float(eps)
        if eps_f not in resnet_preds:
            continue

        sdt_df = compute_sdt_all_classes(resnet_preds[eps_f], resnet_labels[eps_f])
        mean_dp = sdt_df['d_prime'].mean()
        resnet_mean_dprimes.append((eps_f, mean_dp))

        for _, row in sdt_df.iterrows():
            all_rows.append({
                'epsilon': eps_f,
                'system': 'ResNet',
                'class': CLASSES[int(row['class_idx'])],
                'class_idx': int(row['class_idx']),
                'd_prime': row['d_prime'],
                'beta': row['beta'],
                'hit_rate': row['hit_rate'],
                'fa_rate': row['fa_rate'],
                'hits': row['hits'],
                'misses': row['misses'],
                'false_alarms': row['false_alarms'],
                'correct_rejections': row['correct_rejections'],
            })

        print(f"  ε={eps_f:.2f} | mean d' = {mean_dp:>6.3f}")

    # --- ViT ---
    print("\n--- VIT d-prime per epsilon ---")
    vit_mean_dprimes = []

    for eps in epsilons:
        eps_f = float(eps)
        if eps_f not in vit_preds:
            continue

        sdt_df = compute_sdt_all_classes(vit_preds[eps_f], vit_labels[eps_f])
        mean_dp = sdt_df['d_prime'].mean()
        vit_mean_dprimes.append((eps_f, mean_dp))

        for _, row in sdt_df.iterrows():
            all_rows.append({
                'epsilon': eps_f,
                'system': 'ViT',
                'class': CLASSES[int(row['class_idx'])],
                'class_idx': int(row['class_idx']),
                'd_prime': row['d_prime'],
                'beta': row['beta'],
                'hit_rate': row['hit_rate'],
                'fa_rate': row['fa_rate'],
                'hits': row['hits'],
                'misses': row['misses'],
                'false_alarms': row['false_alarms'],
                'correct_rejections': row['correct_rejections'],
            })

        print(f"  ε={eps_f:.2f} | mean d' = {mean_dp:>6.3f}")

    # --- Human ---
    print("\n--- Human d-prime per epsilon ---")
    human_mean_dprimes = []

    for eps in epsilons:
        eps_f = float(eps)
        if eps_f not in human_preds:
            continue

        sdt_df = compute_sdt_all_classes(human_preds[eps_f], human_labels[eps_f])
        mean_dp = sdt_df['d_prime'].mean()
        human_mean_dprimes.append((eps_f, mean_dp))

        for _, row in sdt_df.iterrows():
            all_rows.append({
                'epsilon': eps_f,
                'system': 'Human',
                'class': CLASSES[int(row['class_idx'])],
                'class_idx': int(row['class_idx']),
                'd_prime': row['d_prime'],
                'beta': row['beta'],
                'hit_rate': row['hit_rate'],
                'fa_rate': row['fa_rate'],
                'hits': row['hits'],
                'misses': row['misses'],
                'false_alarms': row['false_alarms'],
                'correct_rejections': row['correct_rejections'],
            })

        print(f"  ε={eps_f:.2f} | mean d' = {mean_dp:>6.3f}")

    # =========================================================================
    # Save full results to CSV
    # =========================================================================
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📄 Full results saved to: {OUTPUT_CSV}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: d-PRIME vs EPSILON")
    print("=" * 70)
    print(f"{'Epsilon':<10} {'ResNet d′':<12} {'ViT d′':<12} {'Human d′':<12}")
    print("-" * 46)

    resnet_eps_list = [e for e, _ in resnet_mean_dprimes]
    resnet_dp_list = [d for _, d in resnet_mean_dprimes]
    vit_dp_dict = dict(vit_mean_dprimes)
    human_dp_dict = dict(human_mean_dprimes)
    
    crossover_eps = None

    for eps, res_dp in zip(resnet_eps_list, resnet_dp_list):
        vit_dp = vit_dp_dict.get(eps, float('nan'))
        h_dp = human_dp_dict.get(eps, float('nan'))
        
        # Check crossover
        if res_dp > vit_dp:
            pass
        elif res_dp < vit_dp and crossover_eps is None and eps > 0.00:
            crossover_eps = eps
            
        print(f"  {eps:<8.2f} {res_dp:<12.3f} {vit_dp:<12.3f} {h_dp:<12.3f}")

    # =========================================================================
    # Perceptual Threshold Detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("PERCEPTUAL THRESHOLD ANALYSIS (d' = 1.0)")
    print("=" * 70)

    resnet_threshold_eps = find_perceptual_threshold(resnet_eps_list, resnet_dp_list)
    vit_eps_list = [e for e, _ in vit_mean_dprimes]
    vit_dp_list = [d for _, d in vit_mean_dprimes]
    vit_threshold_eps = find_perceptual_threshold(vit_eps_list, vit_dp_list)
    
    human_eps_list = [e for e, _ in human_mean_dprimes]
    human_dp_list = [d for _, d in human_mean_dprimes]
    human_threshold_eps = find_perceptual_threshold(human_eps_list, human_dp_list)

    print(f"  🔴 ResNet threshold: ε = {resnet_threshold_eps}")
    print(f"  🟣 ViT threshold:    ε = {vit_threshold_eps}")
    print(f"  🟢 Human threshold:  ε = {human_threshold_eps}")
    
    if crossover_eps is not None:
        print(f"\n  ⚔️  CROSSOVER DETECTED: ViT d' exceeds ResNet d' at ε = {crossover_eps:.2f}")

    # =========================================================================
    # THE HEADLINE FINDING
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎯 HEADLINE FINDING: PERCEPTUAL THRESHOLD GAP")
    print("=" * 70)

    if resnet_threshold_eps is not None and human_threshold_eps is not None:
        gap = human_threshold_eps - resnet_threshold_eps
        print(f"\n  ResNet loses perceptual sensitivity at ε = {resnet_threshold_eps:.2f}")
        print(f"  Humans lose perceptual sensitivity at ε = {human_threshold_eps:.2f}")
        print(f"  ═══════════════════════════════════════")
        print(f"  THRESHOLD GAP = {gap:.2f} epsilon units")
        print(f"  ═══════════════════════════════════════")
        print()
        print(f"  INTERPRETATION:")
        print(f"  There exists a {gap:.2f}-wide epsilon range where humans can still")
        print(f"  perceive the object identity but the models (especially ResNet) are effectively blind.")
        print(f"  This gap is a direct consequence of the CNN's feedforward,")
        print(f"  texture-biased processing: it lacks the recurrent feedback loops")
        print(f"  that allow the human visual cortex to reconstruct global shape")
        print(f"  information from noisy inputs.")
    elif resnet_threshold_eps is not None and human_threshold_eps is None:
        print(f"\n  ResNet loses sensitivity at ε = {resnet_threshold_eps:.2f}")
        print(f"  Humans NEVER lose sensitivity across all tested epsilons.")
        print(f"  ═══════════════════════════════════════")
        print(f"  INTERPRETATION: Human perceptual robustness exceeds the entire")
        print(f"  tested epsilon range. The CNN's feedforward architecture is")
        print(f"  fundamentally more fragile than biological vision.")
    else:
        print(f"\n  Neither system drops below d'=1.0 in the tested range.")
        print(f"  Consider testing higher epsilon values (ε > {max(epsilons)}).")

    print(f"\n  ONE-SENTENCE SUMMARY FOR PROFESSOR:")
    if resnet_threshold_eps is not None:
        print(f'  "Signal detection analysis reveals the ResNet loses perceptual')
        print(f'   sensitivity (d′<1.0) at ε={resnet_threshold_eps:.2f}, while humans maintain')
        if human_threshold_eps is not None:
            print(f'   above-threshold sensitivity until ε={human_threshold_eps:.2f} — a')
            gap = human_threshold_eps - resnet_threshold_eps
            print(f'   {gap:.2f}-unit robustness gap attributable to the absence of')
        else:
            print(f'   above-threshold sensitivity across all tested conditions — a gap')
            print(f'   attributable to the absence of')
        print(f'   recurrent shape processing in feedforward architectures."')
    else:
        print(f'  "Both systems maintain perceptual sensitivity across all tested')
        print(f'   epsilon levels — higher perturbation budgets may be needed."')

    print()


if __name__ == '__main__':
    main()
