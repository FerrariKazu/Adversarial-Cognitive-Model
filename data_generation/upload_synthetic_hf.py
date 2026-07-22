#!/usr/bin/env python3
"""
Sprint 2 Phase B: Upload Filtered Synthetic Shards to HuggingFace
=================================================================
Uploads filtered WebDataset shards from `./data/synthetic_stl10_filtered`
to HuggingFace dataset repository: `FerrariKazu/stl10-synthetic`

Usage:
  python3 data_generation/upload_synthetic_hf.py \
    --input-dir ./data/synthetic_stl10_filtered \
    --repo-id FerrariKazu/stl10-synthetic
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo

def get_huggingface_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), '../.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
    return token

def main():
    parser = argparse.ArgumentParser(description="Upload Filtered Synthetic Shards to HuggingFace")
    parser.add_argument('--input-dir', type=str, default='./data/synthetic_stl10_filtered',
                        help='Directory containing filtered .tar shards and report')
    parser.add_argument('--repo-id', type=str, default='FerrariKazu/stl10-synthetic',
                        help='HuggingFace dataset repository ID')
    parser.add_argument('--private', action='store_true',
                        help='Make repo private if creating new')
    args = parser.parse_args()

    token = get_huggingface_token()
    if not token:
        print("Error: HF_TOKEN not found in environment or .env file.")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"--> Initializing upload to HuggingFace ({args.repo-id})...", flush=True)

    try:
        create_repo(repo_id=args.repo-id, repo_type="dataset", token=token, private=args.private, exist_ok=True)
        print(f"    ✓ Dataset repository '{args.repo-id}' ready.", flush=True)
    except Exception as e:
        print(f"    Notice: {e}", flush=True)

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)

    files_to_upload = [f for f in os.listdir(args.input_dir) if f.endswith('.tar') or f.endswith('.json')]
    print(f"--> Uploading {len(files_to_upload)} files from {args.input_dir}...", flush=True)

    for idx, fname in enumerate(files_to_upload, 1):
        fpath = os.path.join(args.input_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  [{idx}/{len(files_to_upload)}] Uploading {fname} ({size_mb:.1f} MB)...", flush=True)
        
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=args.repo-id,
            repo_type="dataset",
            token=token
        )

    print("\n============================================================", flush=True)
    print(f"  ✓ Upload Complete! Dataset available at:", flush=True)
    print(f"    https://huggingface.co/datasets/{args.repo-id}", flush=True)
    print("============================================================", flush=True)

if __name__ == '__main__':
    main()
