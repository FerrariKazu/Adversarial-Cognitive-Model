import os
from huggingface_hub import HfApi

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
    commits = api.list_repo_commits(
        repo_id='FerrariKazu/rhan-checkpoints',
        repo_type='dataset'
    )
    print("\nRecent commits in FerrariKazu/rhan-checkpoints:")
    for i, commit in enumerate(commits[:10]):
        print(f"Commit {i}:")
        print(f"  ID: {commit.commit_id}")
        print(f"  Created: {commit.created_at}")
        print(f"  Title: {commit.title}")
        print(f"  Message: {commit.message}")
        print("-" * 50)
except Exception as e:
    print(f"Error listing commits: {e}")
