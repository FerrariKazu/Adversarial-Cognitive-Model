import os
import torch

rolling_path = 'checkpoints_tier2/rhan_stl10_large_pseudolabel_rolling.pth'
best_path = 'checkpoints_tier2/rhan_stl10_large_pseudolabel_best.pth'

if os.path.exists(rolling_path):
    try:
        data = torch.load(rolling_path, map_location='cpu')
        epoch = data.get('epoch', None)
        best_acc = data.get('best_acc', None)
        print(f"Rolling checkpoint: {rolling_path}")
        print(f" -> epoch: {epoch}")
        print(f" -> best_acc: {best_acc}%")
    except Exception as e:
        print(f"Error loading rolling checkpoint: {e}")
else:
    print(f"Rolling checkpoint not found at {rolling_path}")

if os.path.exists(best_path):
    try:
        data = torch.load(best_path, map_location='cpu')
        # Some best checkpoints only save raw state dict. Let's check if it's a dict of state_dict or raw.
        if isinstance(data, dict) and ('epoch' in data or 'model' in data):
            epoch = data.get('epoch', None)
            best_acc = data.get('best_acc', None)
            print(f"Best checkpoint (wrapped): {best_path}")
            print(f" -> epoch: {epoch}")
            print(f" -> best_acc: {best_acc}%")
        else:
            print(f"Best checkpoint (raw state dict): {best_path}")
            print(f" -> Keys: {list(data.keys())[:5]}... (total keys: {len(data.keys())})")
    except Exception as e:
        print(f"Error loading best checkpoint: {e}")
else:
    print(f"Best checkpoint not found at {best_path}")
