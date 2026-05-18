import sys
import os
import yaml
import numpy as np
import torch
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from utils.metrics import accuracy

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    epsilons = config['epsilons']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg = MODELS['clip']
    save_dir = cfg['out']
    lbl_path = os.path.join(save_dir, 'labels.npy')
    labels_np = np.load(lbl_path)
    
    model = cfg['class']().to(device)
    model.eval()
    
    results = {}
    
    print("Evaluating CLIP Adversarial Arrays...")
    for eps in epsilons:
        eps_str = f"{float(eps):.2f}"
        img_path = os.path.join(save_dir, f"pgd_eps{eps_str}_images.npy")
        images_mmap = np.load(img_path, mmap_mode='r')
        
        batch_size = 32
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(labels_np), batch_size):
                batch_imgs = torch.tensor(images_mmap[i:i+batch_size], device=device)
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                del batch_imgs, outputs
                
        torch.cuda.empty_cache()
        all_preds = np.concatenate(all_preds)
        acc = accuracy(all_preds, labels_np)
        results[float(eps)] = acc
        
    print("\n===========================================================================")
    print("PGD ACCURACY COLLAPSE — CLIP ViT-B/32")
    print("===========================================================================")
    print(f"{'Epsilon':<10} | {'CLIP':<5}")
    print("-------------------------")
    for eps in epsilons:
        print(f"{float(eps):<10.2f} | {results[float(eps)]:.2f}%")
        
if __name__ == '__main__':
    main()
