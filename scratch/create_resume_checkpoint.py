import os
import sys
import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_v6 import RHANv6

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = 'checkpoints'
    best_path = os.path.join(ckpt_dir, 'rhan_v6_best.pth')
    checkpoint_path = os.path.join(ckpt_dir, 'rhan_v6_checkpoint.pth')
    
    if not os.path.exists(best_path):
        print(f"ERROR: best checkpoint not found at {best_path}")
        return
        
    print(f"Loading weights from {best_path}...")
    model = RHANv6(head_type='cosine').to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scaler = GradScaler('cuda')
    
    # We set epoch = 18. The resume logic will do start_epoch = epoch + 1 = 19
    # which is Epoch 20 (since index 0 is Epoch 1).
    checkpoint_data = {
        'epoch': 18,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': {},
        'phase_name': 'A',
        'epsilon': 0.031,
        'best_test_acc': 82.74,
        'loss_history': {
            'l_adv': [],
            'l_clean': [],
            'l_align': [],
            'l_freq': [],
            'l_ponder': [],
            'l_total': [],
            'train_acc': [],
            'test_acc': []
        }
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Successfully constructed and saved resume checkpoint to: {checkpoint_path}")

if __name__ == '__main__':
    main()
