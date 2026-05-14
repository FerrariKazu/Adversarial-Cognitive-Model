import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))
from model import CIFARResNet
from model_vit import CIFARViT
from model_efficientnet import CIFAREfficientNet
from model_shaperesnet import ShapeResNet

def evaluate_model(model_name, dir_name, model_class, ckpt_path, device, epsilons):
    print(f"\nEvaluating {model_name}...")
    
    if model_class == ShapeResNet:
        # ShapeResNet handles its own specialized loading logic in __init__
        model = model_class(num_classes=10, weights_path=ckpt_path).to(device)
    else:
        model = model_class().to(device)
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"  Warning: Checkpoint NOT found at {ckpt_path}")
    
    model.eval()

    labels_path = os.path.join(os.path.dirname(__file__), 'adv_images', dir_name, 'labels.npy')
    if not os.path.exists(labels_path):
        print(f"  Error: Labels not found at {labels_path}")
        return [float('nan')] * len(epsilons)
        
    labels = np.load(labels_path)
    labels_tensor = torch.tensor(labels, device=device)
    
    accuracies = []
    
    for eps in epsilons:
        images_path = os.path.join(os.path.dirname(__file__), 'adv_images', dir_name, f"pgd_eps{eps:.2f}_images.npy")
        
        if not os.path.exists(images_path):
            print(f"  Missing images for eps {eps:.2f}")
            accuracies.append(float('nan'))
            continue

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
        # ('ResNet-18', 'resnet', CIFARResNet, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'best.pth')),
        # ('ViT-Small', 'vit', CIFARViT, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'vit_small_best.pth')),
        # ('EfficientNet', 'efficientnet', CIFAREfficientNet, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'efficientnet_best.pth')),
        ('ShapeResNet', 'shaperesnet', ShapeResNet, os.path.join(os.path.dirname(__file__), '..', 'phase1_training', 'checkpoints', 'shaperesnet50_best_v2.pth'))
    ]
    
    results = {}
    for name, dir_name, mclass, ckpt in models_to_eval:
        try:
            accs = evaluate_model(name, dir_name, mclass, ckpt, device, epsilons)
            results[name] = accs
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            results[name] = [float('nan')] * len(epsilons)
            
    # Print the requested table dynamically
    print("\n" + "=" * 75)
    print("PGD ACCURACY COLLAPSE COMPARISON")
    print("=" * 75)
    
    header = f"{'Epsilon':<10}"
    for name, _, _, _ in models_to_eval:
        header += f" | {name:<12}"
    print(header)
    print("-" * len(header))
    
    for i, eps in enumerate(epsilons):
        row = f"{eps:<10.2f}"
        for name, _, _, _ in models_to_eval:
            acc = results.get(name, [float('nan')]*len(epsilons))[i]
            row += f" | {acc:<12.2f}"
        print(row)

if __name__ == '__main__':
    main()
