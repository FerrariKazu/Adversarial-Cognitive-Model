import os
from huggingface_hub import HfApi

local_path = "checkpoints/rhan_stl10_pseudolabel_best.pth"
if not os.path.exists(local_path):
    print(f"Error: {local_path} not found. Please make sure you are in the project root directory.")
    exit(1)

token = input("Please enter your Hugging Face write token: ").strip()
if not token:
    print("Error: A token is required to upload files to a private/public dataset repository.")
    exit(1)

api = HfApi(token=token)
try:
    print("Uploading checkpoints/rhan_stl10_pseudolabel_best.pth to FerrariKazu/rhan-checkpoints...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="rhan_stl10_pseudolabel_best.pth",
        repo_id="FerrariKazu/rhan-checkpoints",
        repo_type="dataset"
    )
    print("\nSuccess! Upload completed. Colab will now be able to fetch it.")
except Exception as e:
    print(f"\nUpload failed: {e}")
