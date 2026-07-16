import os
import torch
from huggingface_hub import HfApi, hf_hub_download

def check_checkpoint(repo_id, filename, token):
    try:
        print(f"\nChecking {filename} in {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type='dataset',
            token=token
        )
        data = torch.load(path, map_location='cpu')
        epoch = data.get('epoch', None)
        best_acc = data.get('best_acc', None)
        print(f"-> Found: epoch {epoch}, best_acc {best_acc}%")
    except Exception as e:
        print(f"-> Could not read {filename}: {e}")

def main():
    token = None
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.strip().startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip('"').strip("'")
    
    print(f"HF_TOKEN detected: {bool(token)}")
    if not token:
        print("No HF_TOKEN found in .env. Exiting.")
        return

    api = HfApi(token=token)
    
    # 1. Check rolling repo
    print("\n--- Checking rolling repo ---")
    try:
        files = api.list_repo_files(repo_id='FerrariKazu/rhan-checkpoints-rolling', repo_type='dataset')
        print("Files in rhan-checkpoints-rolling:")
        for f in files:
            print(f"  - {f}")
        if 'rhan_stl10_v10_rolling.pth' in files:
            check_checkpoint('FerrariKazu/rhan-checkpoints-rolling', 'rhan_stl10_v10_rolling.pth', token)
    except Exception as e:
        print(f"Could not access rhan-checkpoints-rolling: {e}")

    # 2. Check main repo
    print("\n--- Checking main repo ---")
    try:
        files = api.list_repo_files(repo_id='FerrariKazu/rhan-checkpoints', repo_type='dataset')
        print("Files in rhan-checkpoints:")
        for f in files:
            print(f"  - {f}")
        for fn in ['rhan_stl10_v10_best.pth', 'rhan_stl10_large_pseudolabel_best.pth', 'rhan_stl10_large_pseudolabel_rolling.pth']:
            if fn in files:
                check_checkpoint('FerrariKazu/rhan-checkpoints', fn, token)
    except Exception as e:
        print(f"Could not access rhan-checkpoints: {e}")

if __name__ == '__main__':
    main()
