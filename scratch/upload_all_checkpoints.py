import os
import sys
from huggingface_hub import HfApi

# 1. Parse HF token from .env
token = None
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"').strip("'")

if not token:
    print("Error: HF_TOKEN not found in .env. Please configure it first.")
    sys.exit(1)

print(f"HF_TOKEN successfully loaded.")

# 2. Files to upload
files_to_upload = [
    # (local_path, path_in_repo)
    ("checkpoints_tier2/rhan_stl10_large_pseudolabel_best.pth", "rhan_stl10_large_pseudolabel_best.pth"),
    ("checkpoints_tier2/rhan_stl10_large_pseudolabel_rolling.pth", "rhan_stl10_large_pseudolabel_rolling.pth"),
    ("checkpoints_tier2/rhan_stl10_large_video_tdv.pth", "rhan_stl10_large_video_tdv.pth"),
    ("checkpoints_tier2/rhan_stl10_pseudolabel_best.pth", "rhan_stl10_pseudolabel_best.pth"),
]

api = HfApi(token=token)
repo_id = "FerrariKazu/rhan-checkpoints"

print(f"\nStarting upload of checkpoints to {repo_id}...")
for local_path, repo_path in files_to_upload:
    if os.path.exists(local_path):
        print(f"Uploading {local_path} -> {repo_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            print(f"  ✓ Upload success.")
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
    else:
        print(f"Skipping {local_path} (file not found locally).")

print("\nAll uploads complete!")
