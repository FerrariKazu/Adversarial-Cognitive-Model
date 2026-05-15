import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from phase1_training.dataset import CLASSES
from utils.metrics import per_class_accuracy

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
HUMAN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'phase3_human_study', 'data', 'responses_mapped.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

def get_cnn_class_matrix(model_name, m_cfg, device, epsilons):
    """Returns a 10x6 matrix of CNN accuracy."""
    matrix = np.zeros((10, len(epsilons)))
    
    save_dir = m_cfg['out']
    lbl_path = os.path.join(save_dir, 'labels.npy')
    
    if not os.path.exists(lbl_path):
        print(f"  Skipping {model_name}: labels.npy not found.")
        return None

    labels_np = np.load(lbl_path)
    
    # Load Model
    mclass = m_cfg['class']
    ckpt_path = m_cfg.get('ckpt')
    
    if mclass.__name__ == 'ShapeResNet':
        model = mclass(num_classes=10, weights_path=ckpt_path).to(device)
    else:
        model = mclass().to(device)
        if not m_cfg.get('zero_shot') and ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    model.eval()
    
    for j, eps in enumerate(epsilons):
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(save_dir, f"pgd_eps{eps_str}_images.npy")
        
        if not os.path.exists(img_path):
            matrix[:, j] = np.nan
            continue
            
        images_mmap = np.load(img_path, mmap_mode='r')
        
        batch_size = 32 if model_name in ['vit', 'efficientnet', 'clip', 'bagnet'] else 64
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch_imgs = torch.tensor(images_mmap[i:i+batch_size], device=device)
                preds = model(batch_imgs).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                del batch_imgs
                
        torch.cuda.empty_cache()
        all_preds = np.concatenate(all_preds)
        accs = per_class_accuracy(all_preds, labels_np, num_classes=10)
        matrix[:, j] = accs
        
    del model
    torch.cuda.empty_cache()
    return matrix

def get_human_class_matrix(df, epsilons):
    """Returns a 10x6 matrix of Human accuracy."""
    matrix = np.zeros((10, len(epsilons)))
    matrix.fill(np.nan)
    
    df_pgd = df[df['attack_type'] == 'pgd']
    
    for j, eps in enumerate(epsilons):
        subset = df_pgd[df_pgd['epsilon'].astype(float).round(2) == float(eps)]
        if len(subset) == 0:
            continue
            
        for c_idx, class_name in enumerate(CLASSES):
            class_subset = subset[subset['true_class'] == class_name]
            if len(class_subset) > 0:
                acc = class_subset['response_correct'].mean() * 100.0
                matrix[c_idx, j] = acc
                
    return matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Specific model to run heatmap for')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(HUMAN_DATA_PATH):
        df_human = pd.read_csv(HUMAN_DATA_PATH)
        hum_mat = get_human_class_matrix(df_human, epsilons)
    else:
        hum_mat = np.full((10, len(epsilons)), np.nan)
    
    models_to_run = [args.model] if args.model else MODELS.keys()
    
    for model_name in models_to_run:
        if model_name not in MODELS:
            continue
            
        m_cfg = MODELS[model_name]
        print(f"\n--- Running Class Heatmap for {model_name.upper()} ---")
        
        cnn_mat = get_cnn_class_matrix(model_name, m_cfg, device, epsilons)
        if cnn_mat is None:
            continue
        
        delta_mat = cnn_mat - hum_mat
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=150)
        eps_labels = [f"ε={e}" for e in epsilons]
        
        sns.heatmap(cnn_mat, ax=axes[0], cmap="RdYlGn", vmin=0, vmax=100,
                    xticklabels=eps_labels, yticklabels=CLASSES, annot=True, fmt=".0f", cbar=False)
        axes[0].set_title(f"{model_name.upper()} Accuracy (%)", fontsize=14, pad=15)
        
        sns.heatmap(hum_mat, ax=axes[1], cmap="RdYlGn", vmin=0, vmax=100,
                    xticklabels=eps_labels, yticklabels=[], annot=True, fmt=".0f", cbar=False)
        axes[1].set_title("Human Accuracy (%)", fontsize=14, pad=15)
        
        sns.heatmap(delta_mat, ax=axes[2], cmap="coolwarm_r", vmin=-100, vmax=100,
                    xticklabels=eps_labels, yticklabels=[], annot=True, fmt=".0f")
        axes[2].set_title(f"Delta ({model_name.upper()} - Human)", fontsize=14, pad=15)
        
        plt.suptitle(f"Per-Class Robustness: {model_name.upper()}", fontsize=18, y=1.05)
        plt.tight_layout()
        
        model_out_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        out_path = os.path.join(model_out_dir, 'class_heatmap.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to: {out_path}")

if __name__ == '__main__':
    main()
