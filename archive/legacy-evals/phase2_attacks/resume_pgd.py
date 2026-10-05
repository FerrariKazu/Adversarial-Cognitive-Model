import sys, os, numpy as np, torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model_shaperesnet import ShapeResNet
from pgd import pgd_attack
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

OUT_DIR = 'adv_images/shaperesnet'
CHECKPOINT = '../phase1_training/checkpoints/shaperesnet50_best_v2.pth'
EPSILON_LEVELS = [0.02, 0.05, 0.10, 0.20]  # skip 0.01 — already done
BATCH_SIZE = 16
num_workers = 0  # set to 0 to avoid I/O multiprocessing errors

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
                      shuffle=False, num_workers=num_workers)

def load_model(device):
    model = ShapeResNet(num_classes=10, weights_path=None).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded ✓")
    return model

def evaluate_accuracy(all_preds, all_labels):
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    return 100. * correct / len(all_labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    loader = get_testloader()

    all_labels = np.load(os.path.join(OUT_DIR, 'labels.npy')).tolist()
    print(f"Labels loaded: {len(all_labels)} samples")

    # Delete corrupted pgd_eps0.02 file
    corrupt = os.path.join(OUT_DIR, 'pgd_eps0.02_images.npy')
    if os.path.exists(corrupt) and os.path.getsize(corrupt) < 100_000_000:
        os.remove(corrupt)
        print("Removed corrupted pgd_eps0.02 file")

    for eps in EPSILON_LEVELS:
        fname = f"pgd_eps{eps:.2f}_images.npy"
        fpath = os.path.join(OUT_DIR, fname)

        # Skip if already saved correctly
        if os.path.exists(fpath) and os.path.getsize(fpath) > 100_000_000:
            print(f"Skipping {fname} — already complete")
            continue

        print(f"\nRunning PGD eps={eps:.2f}...")
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
        np.save(fpath, all_images)
        print(f"  PGD eps={eps:.2f} → Accuracy: {acc:.2f}% — saved {fname}")

    print("\nPGD attacks complete!")

if __name__ == '__main__':
    main()
