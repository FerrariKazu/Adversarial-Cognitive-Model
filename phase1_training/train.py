import os
import time
import yaml
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders

# -----------------------------------------------------------------------------
# Dynamic Model Router
# -----------------------------------------------------------------------------
def get_model(name: str) -> nn.Module:
    """Routes the --model flag to the correct architecture class."""
    name = name.lower()
    if name == "efficientnet":
        from model_efficientnet import CIFAREfficientNet
        return CIFAREfficientNet(num_classes=10)
    if name == "resnet":
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 10)
        return m
    if name == "cornets":
        from model_cornets import CIFARCORnet 
        return CIFARCORnet(num_classes=10)
        
    raise ValueError(f"Unknown model: {name!r}.")


# -----------------------------------------------------------------------------
# Reproducibility Setup
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # -------------------------------------------------------------------------
    # Terminal Argument Parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Adversarial Cognitive Model Training")
    parser.add_argument('--model', type=str, required=True, help="Model to train: resnet, efficientnet, cornets")
    args = parser.parse_args()

    # Load Configuration
    with open('../config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    # -------------------------------------------------------------------------
    # Device and Model Setup
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Initializing model: {args.model}")
    
    # Dynamically load the model based on the terminal flag
    model = get_model(args.model).to(device)
    
    # Pass the model_name to get_dataloaders so it applies the correct Resize transforms
    trainloader, testloader = get_dataloaders(
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
        data_dir='../data',
        model_name=args.model
    )
    
    # -------------------------------------------------------------------------
    # Loss, Optimizer, and Scheduler
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
        # Checkpointing (Dynamically named so it doesn't overwrite teammates)
        # ---------------------------------------------------------------------
        save_path = f'checkpoints/{args.model}_best.pth'
        
        if test_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = test_acc
            
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss: {train_loss:.4f} | Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.5f} | Time: {elapsed:.1f}s")
              
    writer.close()
    print(f"Training complete. Best Accuracy for {args.model}: {best_acc:.2f}%")
    print(f"Checkpoint saved to: checkpoints/{args.model}_best.pth")

if __name__ == '__main__':
    main()