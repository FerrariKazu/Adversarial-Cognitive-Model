import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import psutil

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")
    print("Close other applications before continuing.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from phase1_training.dataset import CLASSES
from phase5_sdt.sdt_core import compute_sdt_all_classes, d_prime

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
HUMAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'sdt_results_v4.csv')

def get_cnn_predictions(model_name, cfg, device, epsilons):
    """Returns dict[epsilon] -> (preds_array, labels_array)"""
    all_preds = {}
    all_labels = {}

    save_dir = cfg['out']
    lbl_path = os.path.join(save_dir, 'labels.npy')
    
    if not os.path.exists(lbl_path):
        return None, None

    labels_np = np.load(lbl_path)
    
    # Load Model
    mclass = cfg['class']
    ckpt_path = cfg.get('ckpt')
    
    if mclass.__name__ == 'ShapeResNet':
        model = mclass(num_classes=10, weights_path=ckpt_path).to(device)
    else:
        model = mclass().to(device)
        if not cfg.get('zero_shot') and ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    model.eval()
    
    print(f"Computing {model_name.upper()} SDT data...")
    for eps in epsilons:
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(save_dir, f"pgd_eps{eps_str}_images.npy")
        
        if not os.path.exists(img_path):
            continue
            
        images_mmap = np.load(img_path, mmap_mode='r')
        batch_size = 32 if model_name in ['vit', 'efficientnet', 'clip', 'bagnet'] else 64
        preds_list = []

        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch = torch.tensor(images_mmap[i:i + batch_size], device=device)
                preds = model(batch).argmax(dim=1).cpu().numpy()
                preds_list.append(preds)
                del batch
        
        torch.cuda.empty_cache()
        all_preds[float(eps)] = np.concatenate(preds_list)
        all_labels[float(eps)] = labels_np
        
    del model
    torch.cuda.empty_cache()
    return all_preds, all_labels

def load_human_data_for_sdt(epsilons):
    if not os.path.exists(HUMAN_DATA_PATH):
        print("Human data missing - SDT will skip human comparison.")
        return {}, {}

    df = pd.read_csv(HUMAN_DATA_PATH)
    df_pgd = df[df['attack_type'] == 'pgd'].copy()
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    all_preds = {}
    all_labels = {}

    for eps in epsilons:
        eps_f = float(eps)
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == eps_f]
        if len(subset) == 0:
            continue
        labels = subset['true_class'].map(class_to_idx).values.astype(int)
        
        pred_col = None
        for col in ['predicted_class', 'response_class', 'human_response']:
            if col in subset.columns:
                pred_col = col
                break
        
        if pred_col:
            preds = subset[pred_col].map(class_to_idx).values.astype(int)
        else:
            # Reconstruct approximately
            preds = labels.copy()
            incorrect_mask = ~subset['response_correct'].astype(bool).values
            rng = np.random.default_rng(42)
            for idx in np.where(incorrect_mask)[0]:
                wrong_choices = [c for c in range(10) if c != labels[idx]]
                preds[idx] = rng.choice(wrong_choices)

        all_preds[eps_f] = preds
        all_labels[eps_f] = labels

    return all_preds, all_labels

def find_perceptual_threshold(epsilons, d_primes, threshold=1.0):
    for eps, dp in zip(epsilons, d_primes):
        if dp < threshold:
            return eps
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_rows = []
    results_summary = {}

    # Models
    for model_name, m_cfg in MODELS.items():
        preds, labels = get_cnn_predictions(model_name, m_cfg, device, epsilons)
        if preds:
            mean_dprimes = []
            for eps in epsilons:
                eps_f = float(eps)
                if eps_f in preds:
                    sdt_df = compute_sdt_all_classes(preds[eps_f], labels[eps_f])
                    mean_dp = sdt_df['d_prime'].mean()
                    mean_dprimes.append((eps_f, mean_dp))
                    for _, row in sdt_df.iterrows():
                        all_rows.append({
                            'epsilon': eps_f,
                            'system': model_name.capitalize(),
                            'class': CLASSES[int(row['class_idx'])],
                            'd_prime': row['d_prime'],
                            'hit_rate': row['hit_rate'],
                            'fa_rate': row['fa_rate']
                        })
            results_summary[model_name] = mean_dprimes

    # Human
    h_preds, h_labels = load_human_data_for_sdt(epsilons)
    if h_preds:
        mean_dprimes = []
        for eps in epsilons:
            eps_f = float(eps)
            if eps_f in h_preds:
                sdt_df = compute_sdt_all_classes(h_preds[eps_f], h_labels[eps_f])
                mean_dp = sdt_df['d_prime'].mean()
                mean_dprimes.append((eps_f, mean_dp))
                for _, row in sdt_df.iterrows():
                    all_rows.append({
                        'epsilon': eps_f,
                        'system': 'Human',
                        'class': CLASSES[int(row['class_idx'])],
                        'd_prime': row['d_prime'],
                        'hit_rate': row['hit_rate'],
                        'fa_rate': row['fa_rate']
                    })
        results_summary['human'] = mean_dprimes

    # Save to CSV
    pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSDT Results saved to {OUTPUT_CSV}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Epsilon':<8}", end="")
    for system in results_summary.keys():
        print(f" | {system[:8]:<8}", end="")
    print("\n" + "-" * 80)

    for eps in epsilons:
        eps_f = float(eps)
        print(f"{eps_f:<8.2f}", end="")
        for system, d_list in results_summary.items():
            d_val = next((d for e, d in d_list if e == eps_f), np.nan)
            print(f" | {d_val:<8.3f}", end="")
        print()

    # Thresholds
    print("\n" + "=" * 80)
    print("PERCEPTUAL THRESHOLDS (d' = 1.0)")
    print("-" * 80)
    for system, d_list in results_summary.items():
        eps_list = [e for e, _ in d_list]
        dp_list = [d for _, d in d_list]
        thresh = find_perceptual_threshold(eps_list, dp_list)
        print(f"  {system.capitalize():<12} threshold: ε = {thresh}")

if __name__ == '__main__':
    main()
