# pip install git+https://github.com/dicarlolab/CORnet.git
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_cornets import CIFARCORnet
from dataset_vit import get_dataloaders_vit

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = CIFARCORnet().to(device)
    
    # Get dataloaders for CORnet-S (224x224, CIFAR-10 normalization) using dataset_vit
    trainloader, testloader = get_dataloaders_vit(
        batch_size=64, 
        num_workers=4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer, lr=0.0001, weight_decay=0.01
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # CosineAnnealingLR, T_max=30, epochs=30
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # TensorBoard setup
    writer = SummaryWriter('../runs/cornets_cifar10')
    os.makedirs('checkpoints', exist_ok=True)
    
    best_acc = 0.0
    
    print("Starting training for CORnet-S...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Loop
        model.train()
        train_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(trainloader.dataset)
        
        # Evaluation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        test_acc = 100. * correct / total
        current_lr = scheduler.get_last_lr()[0]
        
        # TensorBoard Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Checkpointing
        if test_acc > best_acc:
            torch.save(model.state_dict(), 'checkpoints/cornets_best.pth')
            best_acc = test_acc
            
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.5f} | Time: {elapsed:.1f}s", flush=True)
              
    writer.close()
    print(f"Training complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
