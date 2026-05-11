import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model_shaperesnet import ShapeResNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='../data', train=True,
                                  download=True, transform=train_transforms)
    test_set  = datasets.CIFAR10(root='../data', train=False,
                                  download=True, transform=test_transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS     = 15
    LR         = 0.0001
    BATCH_SIZE = 32
    os.makedirs("checkpoints", exist_ok=True)

    model = ShapeResNet(num_classes=10,
                        weights_path='resnet50_trained_on_SIN.model').to(device)

    # Freeze all layers except FC for first 5 epochs
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    print("Phase 1: Training FC layer only (epochs 1-5)...")

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    # Optimizer created ONCE outside the loop — fixes momentum reset bug
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze all layers at epoch 6 — only update requires_grad, not optimizer
        if epoch == 6:
            print("Phase 2: Unfreezing all layers...")
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        running_loss = correct = total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {running_loss/(batch_idx+1):.3f} "
                      f"Acc: {100.*correct/total:.1f}%")

        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch:02d}/{EPOCHS} — "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       'checkpoints/shaperesnet50_best_v2.pth')
            print(f"  *** New best: {best_acc:.2f}% — checkpoint saved ***")

    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print("Checkpoint: checkpoints/shaperesnet50_best_v2.pth")

if __name__ == '__main__':
    main()
