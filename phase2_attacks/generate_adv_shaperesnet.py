"""
Adversarial Attack Generation — Shape-ResNet-50
================================================
Generates adversarial examples using FGSM, PGD, and C&W attacks.
Output saved to phase2_attacks/adv_images/shaperesnet/
"""
import sys
import os
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model_shaperesnet import ShapeResNet
from fgsm import fgsm_attack
from pgd import pgd_attack
from cw import cw_attack
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), 'adv_images', 'shaperesnet')
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT = os.path.join(os.path.dirname(__file__),
             '..', 'phase1_training', 'checkpoints', 'shaperesnet50_best_v2.pth')

EPSILON_LEVELS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]
BATCH_SIZE = 16  # small for 2GB VRAM

def get_testloader():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(root='../data', train=False,
                                 download=False, transform=transform)
    return DataLoader(test_set, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=2)

def load_model(device):
    model = ShapeResNet(num_classes=10, weights_path=None).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {CHECKPOINT}")
    return model

def evaluate_accuracy(all_preds, all_labels):
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    return 100. * correct / len(all_labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    loader = get_testloader()

    # Save clean labels once
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    np.save(os.path.join(OUT_DIR, 'labels.npy'), np.array(all_labels))
    print(f"Labels saved: {len(all_labels)} samples")

    # FGSM attacks
    print("\n--- FGSM Attacks ---")
    for eps in EPSILON_LEVELS:
        all_images, all_preds = [], []
        for images, labels in tqdm(loader, desc=f"FGSM eps={eps:.2f}"):
            images, labels = images.to(device), labels.to(device)
            if eps == 0.0:
                adv_images = images
            else:
                adv_images, _ = fgsm_attack(model, images, labels, eps, device)
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_images.append(adv_images.cpu().numpy())
            all_preds.extend(preds)

        all_images = np.concatenate(all_images, axis=0)
        acc = evaluate_accuracy(all_preds, all_labels)
        fname = f"fgsm_eps{eps:.2f}_images.npy"
        np.save(os.path.join(OUT_DIR, fname), all_images)
        print(f"  FGSM eps={eps:.2f} → Accuracy: {acc:.2f}% — saved {fname}")

    # PGD attacks
    print("\n--- PGD Attacks ---")
    for eps in EPSILON_LEVELS:
        if eps == 0.0:
            continue
        all_images, all_preds = [], []
        for images, labels in tqdm(loader, desc=f"PGD eps={eps:.2f}"):
            images, labels = images.to(device), labels.to(device)
            adv_images, _ = pgd_attack(model, images, labels, eps,
                                    alpha=eps/4, steps=20, device=device)
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_images.append(adv_images.cpu().numpy())
            all_preds.extend(preds)

        all_images = np.concatenate(all_images, axis=0)
        acc = evaluate_accuracy(all_preds, all_labels)
        fname = f"pgd_eps{eps:.2f}_images.npy"
        np.save(os.path.join(OUT_DIR, fname), all_images)
        print(f"  PGD  eps={eps:.2f} → Accuracy: {acc:.2f}% — saved {fname}")

    # C&W attack (runs on subset — too slow for full test set on MX330)
    print("\n--- C&W Attack (500 samples) ---")
    cw_images, cw_preds, cw_labels = [], [], []
    count = 0
    for images, labels in loader:
        if count >= 500:
            break
        images, labels = images.to(device), labels.to(device)
        adv_images = cw_attack(model, images, labels, device)
        outputs = model(adv_images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        cw_images.append(adv_images.cpu().numpy())
        cw_preds.extend(preds)
        cw_labels.extend(labels.cpu().numpy())
        count += len(images)

    cw_images = np.concatenate(cw_images, axis=0)
    acc = evaluate_accuracy(cw_preds, cw_labels)
    np.save(os.path.join(OUT_DIR, 'cw_images.npy'), cw_images)
    print(f"  C&W → Accuracy: {acc:.2f}% — saved cw_images.npy")

    print(f"\nAll attacks complete. Files saved to {OUT_DIR}")
    print("Summary of output files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} — {size/1e6:.1f} MB")

if __name__ == '__main__':
    main()
