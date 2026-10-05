"""
CLIP Adversarial Attack Generation
Standalone script — uses CLIPAttackWrapper for gradient flow.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

EPSILONS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]
OUT_DIR = os.path.join(os.path.dirname(__file__), 'adv_images', 'clip')
os.makedirs(OUT_DIR, exist_ok=True)

class CLIPAttackWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, _ = clip.load('ViT-B/32', device=device)
        self.model.float()
        prompts = [
            'a photo of an airplane', 'a photo of an automobile',
            'a photo of a bird', 'a photo of a cat', 'a photo of a deer',
            'a photo of a dog', 'a photo of a frog', 'a photo of a horse',
            'a photo of a ship', 'a photo of a truck',
        ]
        text_tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features.detach()

    def forward(self, images):
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.model.encode_image(images)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        return 100.0 * image_features @ self.text_features.T

def pgd_clip(model, images, labels, epsilon, device, steps=10, alpha=None):
    if alpha is None:
        alpha = epsilon / 4 if epsilon > 0 else 0
    images = images.to(device)
    labels = labels.to(device)
    if epsilon == 0:
        with torch.no_grad():
            return images.clone(), model(images).argmax(dim=1)
    adv = images.clone().detach()
    adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
    adv = adv.clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(adv), labels)
        model.zero_grad()
        loss.backward()
        adv = adv.detach() + alpha * adv.grad.sign()
        adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
        adv = adv.clamp(0, 1).detach()
    with torch.no_grad():
        preds = model(adv).argmax(dim=1)
    return adv, preds

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    model = CLIPAttackWrapper(device)
    model.train()

    dataset = datasets.CIFAR10(root='./data', train=False, download=True,
        transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Save labels once
    all_labels = torch.tensor(dataset.targets)
    np.save(os.path.join(OUT_DIR, 'labels.npy'), all_labels.numpy())
    print(f"Labels saved: {len(all_labels)}")

    results = {}
    for eps in EPSILONS:
        print(f"\nAttacking epsilon = {eps:.2f}")
        all_adv, all_preds, all_true = [], [], []
        for i, (images, labels) in enumerate(loader):
            adv, preds = pgd_clip(model, images, labels, eps, device)
            all_adv.append(adv.cpu())
            all_preds.append(preds.cpu())
            all_true.append(labels)
            if (i+1) % 10 == 0:
                print(f"  Batch {i+1}/{len(loader)}")
        all_adv = torch.cat(all_adv).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_true = torch.cat(all_true).numpy()
        acc = (all_preds == all_true).mean() * 100
        results[eps] = acc
        fname = f"adv_eps{eps:.2f}.npy".replace('.', '_') + '.npy'
        fname = f"adv_eps{str(eps).replace('.','_')}.npy"
        np.save(os.path.join(OUT_DIR, fname), all_adv)
        print(f"  Accuracy at eps={eps:.2f}: {acc:.2f}%")

    print("\n=== CLIP ADVERSARIAL ACCURACY TABLE ===")
    for eps, acc in results.items():
        print(f"  eps={eps:.2f}: {acc:.2f}%")

if __name__ == '__main__':
    main()
