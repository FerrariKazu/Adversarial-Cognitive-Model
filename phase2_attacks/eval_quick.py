import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))
from model import CIFARResNet
from model_vit import CIFARViT
from model_efficientnet import CIFAREfficientNet

def evaluate_model(model_name, dir_name, model_class, ckpt_path, device, epsilons):
    print(f"\nEvaluating {model_name}...")
    model = model_class().to(device)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    labels_path = os.path.join(os.path.dirname(__file__), 'adv_images', dir_name, 'labels.npy')
    labels = np.load(labels_path)
    labels_tensor = torch.tensor(labels, device=device)
    
    accuracies = []
    
    for eps in epsilons:
        images_path = os.path.join(os.path.dirname(__file__), 'adv_images', dir_name, f"pgd_eps{eps:.2f}_images.npy")
        
        # Memory-safe loading
        adv_images_np = np.load(images_path, mmap_mode='r')
        
        correct = 0
        total = len(labels)
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_images = torch.tensor(adv_images_np[i:i+batch_size], device=device)
                batch_labels = labels_tensor[i:i+batch_size]
                
                outputs = model(batch_images)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                
                # Explicit deletion for memory safety
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    
    models_to_eval = [
        ('ResNet-18', 'resnet', CIFARResNet, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'best.pth')),
        ('ViT-Small', 'vit', CIFARViT, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth')),
        ('EfficientNet', 'efficientnet', CIFAREfficientNet, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'efficientnet_best.pth'))
    ]
    
    results = {}
    for name, dir_name, mclass, ckpt in models_to_eval:
        try:
            accs = evaluate_model(name, dir_name, mclass, ckpt, device, epsilons)
            results[name] = accs
        except FileNotFoundError as e:
            print(f"Skipping {name} due to missing files: {e}")
            results[name] = [float('nan')] * len(epsilons)
            
    # Print the requested table
    print("\n" + "=" * 60)
    print("PGD ACCURACY COLLAPSE COMPARISON (3 MODELS)")
    print("=" * 60)
    print(f"{'Epsilon':<10} | {'ResNet-18':<12} | {'ViT-Small':<12} | {'EfficientNet-B0':<15}")
    print("-" * 60)
    
    for i, eps in enumerate(epsilons):
        r_acc = results['ResNet-18'][i]
        v_acc = results['ViT-Small'][i]
        e_acc = results['EfficientNet'][i]
        print(f"{eps:<10.2f} | {r_acc:<12.2f} | {v_acc:<12.2f} | {e_acc:<15.2f}")

if __name__ == '__main__':
    main()
