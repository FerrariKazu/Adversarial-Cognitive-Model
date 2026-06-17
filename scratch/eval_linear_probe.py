import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase1_training'))

from phase1_training.model_rhan_v9 import RHANv9
from phase1_training.dataset import get_dataloaders

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../checkpoints/rhan_v9_sail.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    model = RHANv9().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # Freeze the encoder
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Loaded Phase 1 representation checkpoint: {ckpt_path}")

    # Data
    batch_size = 256
    trainloader_raw, testloader_raw = get_dataloaders(batch_size=batch_size, num_workers=4, model_name='resnet')
    trainloader = DataLoader(trainloader_raw.dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testloader_raw.dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Linear probe
    # Get feature dimension
    with torch.no_grad():
        dummy_x = torch.randn(1, 3, 32, 32).to(device)
        dummy_f, _ = model.get_feature_vector(dummy_x)
        feat_dim = dummy_f.shape[1]
        print(f"Feature dimension: {feat_dim}")

    probe = nn.Linear(feat_dim, 10).to(device)
    
    optimizer = optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    print(f"\nTraining Linear Probe for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        probe.train()
        correct = total = train_loss = 0
        
        for imgs, lbls in trainloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            with torch.no_grad():
                features, _ = model.get_feature_vector(imgs)
                
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            correct += logits.argmax(1).eq(lbls).sum().item()
            total += imgs.size(0)
            
        train_acc = 100. * correct / total
        
        # Eval
        probe.eval()
        correct_test = total_test = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                features, _ = model.get_feature_vector(imgs)
                logits = probe(features)
                correct_test += logits.argmax(1).eq(lbls).sum().item()
                total_test += imgs.size(0)
                
        test_acc = 100. * correct_test / total_test
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    print("\nVerification target: >60% test accuracy")

if __name__ == '__main__':
    main()
