import torch

checkpoint_path = 'checkpoints/rhan_v5_best.pth'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

try:
    state = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint loaded successfully.")
    
    # Check keys related to frequency weights
    keys_of_interest = ['freq_weight_low', 'freq_weight_high']
    for k in keys_of_interest:
        if k in state:
            val = state[k].item()
            sig_val = torch.sigmoid(state[k]).item()
            print(f"  {k}: raw = {val:.4f}, sigmoid = {sig_val:.4f}")
        else:
            print(f"  Key '{k}' not found in state dict.")
            
except Exception as e:
    print(f"Error loading checkpoint: {e}")
