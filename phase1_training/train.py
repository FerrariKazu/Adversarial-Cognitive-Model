import os
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import CIFARResNet
from dataset import get_dataloaders

# -----------------------------------------------------------------------------
# Reproducibility Setup
# -----------------------------------------------------------------------------
# 1. WHAT: Fixing the random seeds for all libraries that generate randomness.
# 2. WHY: Neural network training is highly stochastic (weight initialization, 
#         data shuffling, augmentations). Setting seeds ensures that if you run 
#         this script twice, you get the exact same accuracy curve. Essential for 
#         scientific research to isolate variables.
# 3. OBSERVE: Subsequent runs yield identical identical results.
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Load Configuration
    with open('../config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    # -------------------------------------------------------------------------
    # Device and Model Setup
    # -------------------------------------------------------------------------
    # 1. WHAT: Checking for CUDA (GPU) and initializing the model.
    # 2. WHY: We must explicitly move the model to the GPU for hardware acceleration.
    # 3. OBSERVE: Should print "Using device: cuda".
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CIFARResNet().to(device)
    trainloader, testloader = get_dataloaders(
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
        data_dir='../data'
    )
    
    # -------------------------------------------------------------------------
    # Loss, Optimizer, and Scheduler
    # -------------------------------------------------------------------------
    # 1. WHAT: Setting up CrossEntropyLoss, SGD Optimizer, and CosineAnnealingLR.
    # 2. WHY: 
    #    - SGD with Momentum is standard for ResNet on CIFAR and usually generalizes 
    #      better than Adam.
    #    - CosineAnnealingLR smoothly decays the learning rate following a cosine 
    #      curve, which helps the model settle into local minima at the end of training.
    # 3. OBSERVE: Learning rate will start at `lr` and slowly decrease to 0 by `epochs`.
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                          momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # TensorBoard setup
    writer = SummaryWriter('../runs/phase1_training')
    os.makedirs('checkpoints', exist_ok=True)
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # ---------------------------------------------------------------------
        # Training Loop
        # ---------------------------------------------------------------------
        # 1. WHAT: The core optimization step. Iterates over batches, calculates
        #          loss, backpropagates gradients, and updates weights.
        # 2. WHY: This is how the neural network learns. `optimizer.zero_grad()` 
        #         prevents gradient accumulation between batches.
        # 3. OBSERVE: Train loss should decrease over the epoch.
        # ---------------------------------------------------------------------
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
        
        # ---------------------------------------------------------------------
        # Evaluation Loop
        # ---------------------------------------------------------------------
        # 1. WHAT: Tests the model on unseen data.
        # 2. WHY: We need to monitor test accuracy to prevent overfitting. 
        #         `torch.no_grad()` saves memory/compute since we don't need gradients.
        # 3. OBSERVE: Test accuracy should gradually climb.
        # ---------------------------------------------------------------------
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
        
        # ---------------------------------------------------------------------
        # Checkpointing
        # ---------------------------------------------------------------------
        # 1. WHAT: Saves the model parameters if it achieves a new high score.
        # 2. WHY: We want the absolute best version of the model for Phase 2 
        #         (Attacks), not necessarily the one from the final epoch which 
        #         might have overfit slightly.
        # 3. OBSERVE: Will create/overwrite `checkpoints/best.pth`.
        # ---------------------------------------------------------------------
        if test_acc > best_acc:
            torch.save(model.state_dict(), 'checkpoints/best.pth')
            best_acc = test_acc
            
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss: {train_loss:.4f} | Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.5f} | Time: {elapsed:.1f}s")
              
    writer.close()
    print(f"Training complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
