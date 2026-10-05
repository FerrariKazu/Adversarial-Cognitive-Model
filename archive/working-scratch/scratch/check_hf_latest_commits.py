import os
from huggingface_hub import HfApi

def print_commits(api, repo_id):
    try:
        commits = api.list_repo_commits(
            repo_id=repo_id,
            repo_type='dataset'
        )
        print(f"\nRecent commits in {repo_id}:")
        for i, commit in enumerate(commits[:3]):
            print(f"Commit {i}:")
            print(f"  ID: {commit.commit_id}")
            print(f"  Created: {commit.created_at}")
            print(f"  Title: {commit.title}")
            print("-" * 50)
    except Exception as e:
        print(f"Could not retrieve commits for {repo_id}: {e}")

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

    api = HfApi(token=token)
    print_commits(api, 'FerrariKazu/rhan-checkpoints-rolling')
    print_commits(api, 'FerrariKazu/rhan-checkpoints')

if __name__ == '__main__':
    main()
