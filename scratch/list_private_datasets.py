import os
from huggingface_hub import HfApi

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
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"Authenticated as: {username}")
        
        datasets = api.list_datasets(author=username)
        print("\nYour Private Datasets:")
        count = 0
        for ds in datasets:
            if ds.private:
                count += 1
                # Try to get size
                try:
                    repo_info = api.repo_info(repo_id=ds.id, repo_type="dataset")
                    print(f"  - {ds.id}")
                    print(f"    Private: {repo_info.private}")
                    print(f"    Last Modified: {repo_info.lastModified}")
                except Exception:
                    print(f"  - {ds.id} (metadata restricted)")
        if count == 0:
            print("  No private datasets found.")
            
    except Exception as e:
        print(f"Error listing datasets: {e}")

if __name__ == '__main__':
    main()
