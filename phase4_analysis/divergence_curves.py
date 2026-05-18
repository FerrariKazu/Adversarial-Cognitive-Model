import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import psutil

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")
    print("Close other applications before continuing.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from utils.metrics import accuracy, confidence_from_logits

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
HUMAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Color Palette for 7 models + Human
COLORS = {
    'resnet': '#E94560',       # Red
    'efficientnet': '#0F3460', # Dark Blue
    'vit': '#9D4EDD',          # Violet
    'shaperesnet': '#F97306',  # Orange
    'cornets': '#00ADB5',      # Cyan
    'clip': '#533483',         # Indigo
    'bagnet': '#558B2F',       # Green
    'rhan-clean': '#F59E0B',   # Amber
    'rhan-adv': '#DC2626',     # Bold Red
    'human': '#2E8B57'         # Sea Green
}

def get_cnn_performance(model_name, cfg, device, epsilons):
    """Run model on all PGD datasets and compute metrics."""
    cnn_acc = []
    cnn_conf = []
    
    print(f"\nEvaluating {model_name.upper()} on adversarial datasets...")
    
    save_dir = cfg['out']
    lbl_path = os.path.join(save_dir, 'labels.npy')
    
    if not os.path.exists(lbl_path):
        print(f"  Skipping {model_name}: labels.npy not found.")
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
    
    for eps in epsilons:
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(save_dir, f"pgd_eps{eps_str}_images.npy")
        
        if not os.path.exists(img_path):
            print(f"  Missing images for ε={eps:.2f}")
            cnn_acc.append(np.nan)
            cnn_conf.append(np.nan)
            continue
            
        images_mmap = np.load(img_path, mmap_mode='r')
        
        batch_size = 32 if model_name in ['vit', 'efficientnet', 'clip', 'bagnet'] else 64
        all_preds = []
        all_confs = []
        
        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch_imgs = torch.tensor(images_mmap[i:i+batch_size], device=device)
                outputs = model(batch_imgs)
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                confs = confidence_from_logits(outputs)
                
                all_preds.append(preds)
                all_confs.append(confs)
                
                del batch_imgs
                del outputs
                del preds
                del confs
                
        torch.cuda.empty_cache()
        all_preds = np.concatenate(all_preds)
        all_confs = np.concatenate(all_confs)
        
        acc = accuracy(all_preds, labels_np)
        mean_conf = np.mean(all_confs)
        
        cnn_acc.append(acc)
        cnn_conf.append(mean_conf)
        print(f"  CNN PGD ε={eps:.2f} | Acc: {acc:>5.1f}% | Conf: {mean_conf:.3f}")
        
    del model
    torch.cuda.empty_cache()
    return np.array(cnn_acc), np.array(cnn_conf)

def get_human_performance(df, epsilons):
    """Extract metrics from human responses dataframe."""
    human_acc = []
    human_conf = []
    
    df_pgd = df[df['attack_type'] == 'pgd'].copy()
    df_pgd['conf_norm'] = df_pgd['confidence_rating'] / 10.0
    
    print("\nEvaluating Human responses...")
    for eps in epsilons:
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == float(eps)]
        if len(subset) == 0:
            human_acc.append(np.nan)
            human_conf.append(np.nan)
            continue
            
        acc = subset['response_correct'].mean() * 100.0
        conf = subset['conf_norm'].mean()
        
        human_acc.append(acc)
        human_conf.append(conf)
        print(f"  Human PGD ε={eps:.2f} | Acc: {acc:>5.1f}% | Conf: {conf:.3f} (n={len(subset)})")
        
    return np.array(human_acc), np.array(human_conf)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_dir = os.path.join(OUTPUT_DIR, 'combined')
    os.makedirs(combined_dir, exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load human data
    if os.path.exists(HUMAN_DATA_PATH):
        df_human = pd.read_csv(HUMAN_DATA_PATH)
        hum_acc, hum_conf = get_human_performance(df_human, epsilons)
    else:
        print("Human data missing - using placeholder.")
        hum_acc = hum_conf = np.full(len(epsilons), np.nan)

    results_acc = {}
    results_conf = {}

    for model_name, m_cfg in MODELS.items():
        if model_name == 'rhan': continue # Skip auto-eval for rhan
        acc, conf = get_cnn_performance(model_name, m_cfg, device, epsilons)
        if acc is not None:
            results_acc[model_name] = acc
            results_conf[model_name] = conf

    # Inject RHAN hardcoded PGD-100 results
    results_acc['rhan-clean'] = np.array([89.06, 69.73, 9.96, 0.39, 0.00, 0.00])
    results_acc['rhan-adv'] = np.array([88.28, 80.86, 39.45, 7.42, 0.20, 0.00])
    
    # We don't have conf for RHAN, so we just add nan to avoid plotting
    results_conf['rhan-clean'] = np.full(len(epsilons), np.nan)
    results_conf['rhan-adv'] = np.full(len(epsilons), np.nan)

    eps_ticks = np.array(epsilons, dtype=float)

    # Plot Accuracy
    plt.figure(figsize=(12, 7), dpi=150)
    for model_name, acc in results_acc.items():
        linestyle = '--' if model_name == 'rhan-adv' else '-'
        plt.plot(eps_ticks, acc, marker='o', lw=3, ls=linestyle, color=COLORS.get(model_name, 'gray'), label=model_name.capitalize())
    
    if not np.isnan(hum_acc).all():
        plt.plot(eps_ticks, hum_acc, marker='s', lw=4, color=COLORS['human'], label='Human Perception', zorder=10)

    # Add RHAN annotation at eps=0.05
    plt.annotate('RHAN-adv: 39.45% — 14× ResNet, 4.6× ViT', 
                 xy=(0.05, 39.45), xytext=(0.06, 45),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                 fontsize=10, fontweight='bold', color='#DC2626')

    plt.title('Adversarial Cognition Divergence — 7 Systems + Human', fontsize=16, pad=15)
    plt.xlabel('Perturbation Budget (Epsilon)', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(combined_dir, 'FINAL_divergence_with_rhan.png'), bbox_inches='tight')
    plt.close()

    # Plot Confidence
    plt.figure(figsize=(12, 7), dpi=150)
    for model_name, conf in results_conf.items():
        plt.plot(eps_ticks, conf, marker='o', lw=3, color=COLORS.get(model_name, 'gray'), label=model_name.capitalize())
    
    if not np.isnan(hum_conf).all():
        plt.plot(eps_ticks, hum_conf, marker='s', lw=4, color=COLORS['human'], label='Human Confidence', zorder=10)

    plt.title('Confidence Degradation — 7 Systems + Human', fontsize=16, pad=15)
    plt.xlabel('Perturbation Budget (Epsilon)', fontsize=12)
    plt.ylabel('Normalized Confidence Score (0-1)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(combined_dir, 'full_confidence_curve.png'), bbox_inches='tight')
    plt.close()

    print(f"\nSaved combined plots to {combined_dir}")

if __name__ == '__main__':
    main()
