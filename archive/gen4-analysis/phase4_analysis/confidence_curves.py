import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from utils.metrics import accuracy

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
HUMAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'combined')

# Exact colors from request
COLORS = {
    'resnet': '#E94560',
    'vit': '#7C3AED',
    'efficientnet': '#0F3460',
    'shaperesnet': '#16A34A',
    'human': '#22C55E',
    'bagnet': '#558B2F' # Adding BagNet for completeness
}

def get_model_metrics(model_name, cfg, device, epsilons):
    """Computes mean confidence and accuracy for a model across epsilons."""
    print(f"\nAnalyzing {model_name.upper()} confidence metrics...")
    
    save_dir = cfg['out']
    lbl_path = os.path.join(save_dir, 'labels.npy')
    
    if not os.path.exists(lbl_path):
        print(f"  Skipping {model_name}: labels.npy not found.")
        return None, None

    labels_np = np.load(lbl_path)
    labels_tensor = torch.tensor(labels_np, device=device)
    
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
    
    conf_scores = []
    acc_scores = []
    
    for eps in epsilons:
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(save_dir, f"pgd_eps{eps_str}_images.npy")
        
        if not os.path.exists(img_path):
            print(f"  Missing images for ε={eps:.2f}")
            conf_scores.append(np.nan)
            acc_scores.append(np.nan)
            continue
            
        images_mmap = np.load(img_path, mmap_mode='r')
        batch_size = 32 if model_name in ['vit', 'efficientnet', 'clip', 'bagnet'] else 64
        
        eps_confs = []
        eps_correct = 0
        
        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch_imgs = torch.tensor(images_mmap[i:i+batch_size], device=device)
                batch_lbls = labels_tensor[i:i+batch_size]
                
                logits = model(batch_imgs)
                probs = F.softmax(logits, dim=1)
                
                # Confidence = mean max softmax score
                max_probs, preds = probs.max(dim=1)
                eps_confs.extend(max_probs.cpu().numpy())
                
                # Accuracy
                eps_correct += (preds == batch_lbls).sum().item()
                
                del batch_imgs, logits, probs, max_probs, preds
                
        torch.cuda.empty_cache()
        
        mean_conf = np.mean(eps_confs)
        acc = (eps_correct / len(labels_np)) * 100.0
        
        conf_scores.append(mean_conf)
        acc_scores.append(acc)
        print(f"  ε={eps:.2f} | Conf: {mean_conf:.3f} | Acc: {acc:.2f}%")
        
    del model
    torch.cuda.empty_cache()
    return np.array(conf_scores), np.array(acc_scores)

def get_human_metrics(df, epsilons):
    """Extract confidence and accuracy from human responses."""
    print("\nAnalyzing Human confidence metrics...")
    df_pgd = df[df['attack_type'] == 'pgd'].copy()
    df_pgd['conf_norm'] = df_pgd['confidence_rating'] / 10.0
    
    hum_conf = []
    hum_acc = []
    
    for eps in epsilons:
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == float(eps)]
        if len(subset) == 0:
            hum_conf.append(np.nan)
            hum_acc.append(np.nan)
            continue
            
        hum_conf.append(subset['conf_norm'].mean())
        hum_acc.append(subset['response_correct'].mean() * 100.0)
        
    return np.array(hum_conf), np.array(hum_acc)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Human Data
    if os.path.exists(HUMAN_DATA_PATH):
        df_human = pd.read_csv(HUMAN_DATA_PATH)
        hum_conf, hum_acc = get_human_metrics(df_human, epsilons)
    else:
        hum_conf = hum_acc = np.full(len(epsilons), np.nan)

    results_conf = {}
    results_acc = {}

    # Filter to models requested by user (plus BagNet which we just finished)
    target_models = ['resnet', 'vit', 'efficientnet', 'shaperesnet', 'bagnet']
    
    for model_name in target_models:
        if model_name in MODELS:
            conf, acc = get_model_metrics(model_name, MODELS[model_name], device, epsilons)
            if conf is not None:
                results_conf[model_name] = conf
                results_acc[model_name] = acc

    # --- Plot 1: Confidence vs Epsilon ---
    plt.figure(figsize=(10, 6), dpi=150)
    for model_name, conf in results_conf.items():
        plt.plot(epsilons, conf, marker='o', lw=3, color=COLORS.get(model_name, 'gray'), label=model_name.capitalize())
    
    if not np.isnan(hum_conf).all():
        plt.plot(epsilons, hum_conf, marker='s', lw=4, color=COLORS['human'], label='Human Perception')

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance-level Confidence')
    plt.title('Subjective Confidence Collapse: Models vs Human Metacognition', fontsize=14)
    plt.xlabel('Perturbation Budget (Epsilon)')
    plt.ylabel('Normalized Confidence (0-1)')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_collapse.png'), bbox_inches='tight')
    plt.close()

    # --- Plot 2: Confidence-Accuracy Gap ---
    plt.figure(figsize=(10, 6), dpi=150)
    max_gap = -1
    headline_model = ""
    headline_eps = 0

    for model_name, conf in results_conf.items():
        acc_norm = results_acc[model_name] / 100.0
        gap = conf - acc_norm
        plt.plot(epsilons, gap, marker='o', lw=3, color=COLORS.get(model_name, 'gray'), label=model_name.capitalize())
        
        # Track max gap for headline
        local_max_idx = np.nanargmax(gap)
        if gap[local_max_idx] > max_gap:
            max_gap = gap[local_max_idx]
            headline_model = model_name
            headline_eps = epsilons[local_max_idx]
            
        # Annotate peak
        plt.annotate(f"Peak: {gap[local_max_idx]:.2f}", 
                     (epsilons[local_max_idx], gap[local_max_idx]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Human gap (usually near zero or negative if conservative)
    if not np.isnan(hum_conf).all():
        hum_gap = hum_conf - (hum_acc / 100.0)
        plt.plot(epsilons, hum_gap, marker='s', lw=4, color=COLORS['human'], label='Human Gap')

    plt.title('The "Overconfidence Gap": Confidence - Accuracy', fontsize=14)
    plt.xlabel('Perturbation Budget (Epsilon)')
    plt.ylabel('Gap (Confidence > Accuracy)')
    plt.axhline(y=0, color='black', lw=1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # --- COGNITIVE EXPLANATION ---
    # The confidence-accuracy gap maps directly to the human concept of being "fooled" 
    # vs just "confused". When a model's accuracy drops but its confidence remains high, 
    # it is making incorrect predictions with high certainty (Type II error in metacognition). 
    # A human who is "confused" by noise would report low confidence. A model that is 
    # "fooled" by adversarial features reports high confidence in a wrong class.
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_accuracy_gap.png'), bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("HEADLINE FINDING")
    print("="*50)
    print(f"Model with highest Overconfidence Gap: {headline_model.upper()}")
    print(f"Peak Gap: {max_gap:.3f} at ε = {headline_eps:.2f}")
    print("="*50)
    print(f"\nFigures saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
