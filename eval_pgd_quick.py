#!/usr/bin/env python3
"""Quick PGD-20 eval for RHAN-v5 selfalign checkpoint.
Uses torchvision CIFAR-10 (already cached) instead of HF datasets."""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))
from model_rhan_v5 import RHANv5

def main():
    device = torch.device('cuda')
    print(f"Device: {device}", flush=True)

    # Load model
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_selfalign_best.pth')
    model = RHANv5().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded: {ckpt_path}", flush=True)

    # Wrap for logits only
    class W(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out
    wrapper = W(model)

    # Load CIFAR-10 test set via torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test set: {len(testset)} images", flush=True)

    # Clean accuracy
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in testloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = wrapper(imgs)
            correct += logits.argmax(1).eq(lbls).sum().item()
            total += lbls.size(0)
    clean_acc = 100. * correct / total
    print(f"\nClean accuracy: {clean_acc:.2f}% ({correct}/{total})", flush=True)

    # PGD-20 attack (eps=8/255)
    epsilon = 8/255
    alpha = 2/255
    steps = 20
    
    # Collect all test images
    all_imgs = []
    all_lbls = []
    for imgs, lbls in testloader:
        all_imgs.append(imgs)
        all_lbls.append(lbls)
    x_test = torch.cat(all_imgs, dim=0).to(device)
    y_test = torch.cat(all_lbls, dim=0).to(device)
    print(f"Running PGD-20 on {x_test.size(0)} images (eps={epsilon:.4f})...", flush=True)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    correct_adv = 0
    total_adv = 0
    t0 = time.time()

    # Process in batches to avoid OOM
    bs = 128
    for i in range(0, x_test.size(0), bs):
        x_batch = x_test[i:i+bs]
        y_batch = y_test[i:i+bs]
        B = x_batch.size(0)

        # PGD random start
        delta = torch.empty_like(x_batch).uniform_(-epsilon, epsilon)
        x_adv = (x_batch + delta).clamp(cifar_min, cifar_max).detach()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = wrapper(x_adv)
            loss = F.cross_entropy(logits, y_batch)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_batch, -epsilon, epsilon)
            x_adv = torch.clamp(x_batch + delta, cifar_min, cifar_max).detach()

        with torch.no_grad():
            preds = wrapper(x_adv).argmax(1)
            correct_adv += (preds == y_batch).sum().item()
            total_adv += B

        if (i // bs) % 40 == 0:
            elapsed = time.time() - t0
            print(f"  [{total_adv}/{x_test.size(0)}] adv_acc so far: {100.*correct_adv/total_adv:.2f}%  ({elapsed:.0f}s)", flush=True)

    adv_acc = 100. * correct_adv / total_adv
    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"PGD-20 accuracy (eps=8/255): {adv_acc:.2f}% ({correct_adv}/{total_adv})", flush=True)
    print(f"Clean accuracy: {clean_acc:.2f}%", flush=True)
    print(f"Robustness gap: {clean_acc - adv_acc:.2f} pp", flush=True)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)", flush=True)
    print(f"{'='*60}", flush=True)

    # Per-class breakdown
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print(f"\nPer-class PGD-20 accuracy:", flush=True)
    with torch.no_grad():
        for c in range(10):
            mask = y_test == c
            if mask.sum() == 0:
                continue
            # Get adversarial predictions for this class
            idx = mask.nonzero(as_tuple=True)[0]
            x_c = x_test[idx]
            y_c = y_test[idx]
            
            # Re-run PGD on this subset
            delta = torch.empty_like(x_c).uniform_(-epsilon, epsilon)
            x_adv_c = (x_c + delta).clamp(cifar_min, cifar_max).detach()
            for _ in range(steps):
                x_adv_c.requires_grad_(True)
                logits = wrapper(x_adv_c)
                loss = F.cross_entropy(logits, y_c)
                grad = torch.autograd.grad(loss, x_adv_c)[0]
                x_adv_c = x_adv_c.detach() + alpha * grad.sign()
                delta = torch.clamp(x_adv_c - x_c, -epsilon, epsilon)
                x_adv_c = torch.clamp(x_c + delta, cifar_min, cifar_max).detach()
            
            preds_c = wrapper(x_adv_c).argmax(1)
            acc_c = (preds_c == y_c).float().mean().item()
            print(f"  {classes[c]:>12s}: {acc_c*100:.1f}%", flush=True)

    print(f"\nDone.", flush=True)

main()
