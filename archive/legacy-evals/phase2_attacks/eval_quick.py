import torch
import numpy as np
import sys
import os
import yaml
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

# Import model registry logic (using the dict from generate_adv_all_models.py but simplified for eval)
from phase2_attacks.generate_adv_all_models import MODELS

def evaluate_model(model_name, cfg, device, epsilons):
    print(f"\nEvaluating {model_name.upper()}...")
    
    # Model Loading Logic (matching generate_adv_all_models.py)
    mclass = cfg['class']
    ckpt_path = cfg.get('ckpt')
    
    if mclass.__name__ == 'ShapeResNet':
        print(f"  Debug: Instantiating {mclass.__name__}")
        print(f"  Debug: Targeted checkpoint: {ckpt_path}")
        # ShapeResNet handles its own specialized loading logic in __init__
        model = mclass(num_classes=10, weights_path=ckpt_path).to(device)
    else:
        model = mclass().to(device)
        print(f"  Debug: Instantiating {mclass.__name__}")
        if cfg.get('zero_shot'):
            print(f"  Using zero-shot model.")
        elif ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"  Loaded checkpoint from: {ckpt_path}")
        else:
            print(f"  Warning: Checkpoint NOT found at {ckpt_path}")
    
    model.eval()
    print(f"  Debug: model.eval() called: {not model.training}")

    save_dir = cfg['out']
    labels_path = os.path.join(save_dir, 'labels.npy')
    
    if not os.path.exists(labels_path):
        error_msg = f"ERROR: BagNet analysis files (labels.npy) MISSING in {save_dir}. " \
                    f"Please run: python3 phase2_attacks/generate_adv_all_models.py --model bagnet"
        print(f"  {error_msg}")
        return [float('nan')] * len(epsilons)
    
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"  WARNING: BagNet checkpoint MISSING at {ckpt_path}. Accuracy will be random (~10%).")
        print(f"  Fix: python3 phase1_training/train.py --model bagnet")
        
    labels = np.load(labels_path)
    labels_tensor = torch.tensor(labels, device=device)
    
    accuracies = []
    
    for eps in epsilons:
        images_path = os.path.join(save_dir, f"pgd_eps{eps:.2f}_images.npy")
        
        if not os.path.exists(images_path):
            print(f"  Missing images for eps {eps:.2f}")
            accuracies.append(float('nan'))
            continue

        # Memory-safe loading
        adv_images_np = np.load(images_path, mmap_mode='r')
        
        correct = 0
        total = len(labels)
        batch_size = 32 if model_name in ['vit', 'efficientnet', 'clip', 'bagnet'] else 64
        
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_images = torch.tensor(adv_images_np[i:i+batch_size], device=device)
                batch_labels = labels_tensor[i:i+batch_size]
                
                outputs = model(batch_images)
                preds = outputs.argmax(dim=1)
                batch_correct = (preds == batch_labels).sum().item()
                correct += batch_correct
                
                if i == 0:
                    print(f"  Debug: First batch accuracy: {100.0 * batch_correct / batch_images.size(0):.2f}%")
                
                del batch_images
                del batch_labels
                del outputs
                del preds
                
        torch.cuda.empty_cache()
        acc = 100.0 * correct / total
        accuracies.append(acc)
        print(f"  PGD | {eps:.2f} | {acc:.2f}%")
        
    del model
    del labels_tensor
    torch.cuda.empty_cache()
    
    return accuracies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, help='Comma-separated list of models to eval (default: all available)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config to get epsilons
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'attack_config.yaml')
    with open(config_path, 'r') as f:
        attack_config = yaml.safe_load(f)
    epsilons = attack_config['epsilons']
    
    if args.models:
        selected_models = [m.strip() for m in args.models.split(',')]
    else:
        # Filter to models that actually have attack directories
        selected_models = []
        for m_name, m_cfg in MODELS.items():
            if os.path.exists(m_cfg['out']):
                selected_models.append(m_name)
    
    results = {}
    for name in selected_models:
        if name not in MODELS:
            print(f"Unknown model: {name}")
            continue
        try:
            accs = evaluate_model(name, MODELS[name], device, epsilons)
            results[name] = accs
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            results[name] = [float('nan')] * len(epsilons)
            
    # Print the comparison table
    print("\n" + "=" * 100)
    print("PGD ACCURACY COLLAPSE COMPARISON (v4.0 7-MODEL SPECTRUM)")
    print("=" * 100)
    
    header = f"{'Epsilon':<10}"
    for name in selected_models:
        header += f" | {name[:12]:<12}"
    print(header)
    print("-" * len(header))
    
    for i, eps in enumerate(epsilons):
        row = f"{eps:<10.2f}"
        for name in selected_models:
            acc = results.get(name, [float('nan')]*len(epsilons))[i]
            row += f" | {acc:<12.2f}"
        print(row)
    print("-" * len(header))

if __name__ == '__main__':
    main()
