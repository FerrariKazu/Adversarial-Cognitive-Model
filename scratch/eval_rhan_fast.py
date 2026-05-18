import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan import RHAN
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

def eval_pgd_fast(model, loader, eps_val, device, cifar_min, cifar_max, steps=100, max_samples=512):
    if eps_val == 0:
        correct = total = 0
        confs = []
        with torch.no_grad():
            for images, labels in loader:
                if total >= max_samples:
                    break
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, predicted = probs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                confs.extend(max_probs.cpu().numpy())
        return 100. * correct / total, np.mean(confs)

    # Disable all parameter gradients to avoid huge backward graph overhead and OOM/thrashing!
    for p in model.parameters():
        p.requires_grad = False

    a = max(eps_val / 10, 0.001)
    correct = total = 0
    confs = []
    for images, labels in loader:
        if total >= max_samples:
            break
        images, labels = images.to(device), labels.to(device)
        adv_images, predicted = pgd_attack(
            model, images, labels,
            epsilon=eps_val, alpha=a, steps=steps,
            device=device, clip_min=cifar_min, clip_max=cifar_max,
        )
        with torch.no_grad():
            outputs = model(adv_images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = probs.max(1)
            confs.extend(max_probs.cpu().numpy())
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100. * correct / total, np.mean(confs)

def main():
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR bounds
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    # Load data
    _, testloader_raw = get_dataloaders(batch_size=256, num_workers=4, model_name='resnet')
    from torch.utils.data import DataLoader
    testloader = DataLoader(
        testloader_raw.dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=False,
    )

    # Load clean model
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    clean_ckpt = os.path.join(ckpt_dir, 'rhan_v2_best.pth')
    clean_model = RHAN(num_classes=10, head_type='linear').to(device)
    clean_model.load_state_dict(torch.load(clean_ckpt, map_location=device))
    clean_model.eval()

    # Load adv model
    adv_ckpt = os.path.join(ckpt_dir, 'rhan_adv_best.pth')
    adv_model = RHAN(num_classes=10, head_type='linear').to(device)
    adv_model.load_state_dict(torch.load(adv_ckpt, map_location=device))
    adv_model.eval()

    # NOTE: DO NOT use torch.compile here to avoid slow compilation of PGD graph
    
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]

    print("\nEvaluating RHAN-clean (v2) Fast...")
    for eps_val in epsilons:
        acc, conf = eval_pgd_fast(clean_model, testloader, eps_val, device, cifar_min, cifar_max)
        print(f"  ε={eps_val:.2f} → Acc: {acc:.2f}%, Conf: {conf:.4f}")

    print("\nEvaluating RHAN-adv Fast...")
    for eps_val in epsilons:
        acc, conf = eval_pgd_fast(adv_model, testloader, eps_val, device, cifar_min, cifar_max)
        print(f"  ε={eps_val:.2f} → Acc: {acc:.2f}%, Conf: {conf:.4f}")

if __name__ == '__main__':
    main()
