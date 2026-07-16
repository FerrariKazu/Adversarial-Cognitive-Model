import os
import torch
from huggingface_hub import hf_hub_download

def main():
    token = None
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.strip().startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip('"').strip("'")
    
    if not token:
        print("No HF_TOKEN found in .env.")
        return

    try:
        print("Downloading rhan_stl10_v10_rolling.pth from FerrariKazu/rhan-checkpoints...")
        path = hf_hub_download(
            repo_id='FerrariKazu/rhan-checkpoints',
            filename='rhan_stl10_v10_rolling.pth',
            repo_type='dataset',
            token=token
        )
        data = torch.load(path, map_location='cpu')
        epoch = data.get('epoch', None)
        best_acc = data.get('best_acc', None)
        print(f"-> rolling checkpoint: epoch {epoch}, best_acc {best_acc}%")
    except Exception as e:
        print(f"Error checking checkpoint: {e}")

if __name__ == '__main__':
    main()
