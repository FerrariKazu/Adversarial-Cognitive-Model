import os
import torch
from huggingface_hub import HfApi, hf_hub_download

# Parse HF token from .env
token = None
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"').strip("'")

print(f"HF_TOKEN detected: {bool(token)}")

try:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id='FerrariKazu/rhan-checkpoints', repo_type='dataset')
    print("Files in HF repo:")
    for f in files:
        print(f" - {f}")
        
    # Check epoch for rhan_stl10_large_pseudolabel_rolling.pth
    if 'rhan_stl10_large_pseudolabel_rolling.pth' in files:
        print("\nDownloading and checking rhan_stl10_large_pseudolabel_rolling.pth...")
        path = hf_hub_download(
            repo_id='FerrariKazu/rhan-checkpoints',
            filename='rhan_stl10_large_pseudolabel_rolling.pth',
            repo_type='dataset',
            token=token
        )
        data = torch.load(path, map_location='cpu')
        epoch = data.get('epoch', None)
        best_acc = data.get('best_acc', None)
        print(f"-> rolling checkpoint epoch: {epoch}, best_acc: {best_acc}%")
        
    # Check epoch for rhan_stl10_large_pseudolabel_best.pth
    if 'rhan_stl10_large_pseudolabel_best.pth' in files:
        print("\nDownloading and checking rhan_stl10_large_pseudolabel_best.pth...")
        path = hf_hub_download(
            repo_id='FerrariKazu/rhan-checkpoints',
            filename='rhan_stl10_large_pseudolabel_best.pth',
            repo_type='dataset',
            token=token
        )
        data = torch.load(path, map_location='cpu')
        epoch = data.get('epoch', None)
        best_acc = data.get('best_acc', None)
        print(f"-> best checkpoint epoch: {epoch}, best_acc: {best_acc}%")
except Exception as e:
    print(f"Error checking HF repo: {e}")
