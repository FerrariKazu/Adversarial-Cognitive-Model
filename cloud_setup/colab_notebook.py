#!/usr/bin/env python3
"""
Google Colab Setup & Execution Script for Experiments 2 & 3
============================================================
This script contains the Python cell commands for mounting Google Drive,
syncing checkpoints, downloading datasets, and running video TDV training.
"""

# %% [markdown]
# # Google Colab Persistent Training Template
# This notebook is optimized to run training for the **Adversarial Cognitive Model (RHAN)**.
# It ensures **no progress is lost** when Colab disconnects by:
# 1. Storing the code repository, custom files, and checkpoints directly in your **Google Drive** (`/content/drive/MyDrive`).
# 2. Storing the large dataset (UCF-101, ~13GB) on the fast local VM scratch space (`/content/data`) to prevent FUSE latency and avoid using up Google Drive space limits.
# 3. Utilizing our custom automatic resume checkpoints (`_resume.pth`) which restore model parameters, epoch progress, learning rates, and optimizer/scheduler states automatically on reconnect.

# %% [markdown]
# ## Step 1: Mount Google Drive
# Run this cell to mount your Google Drive. This enables persistent storage of your repository and checkpoints.

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab environment. Skipping Drive mount.")

# %% [markdown]
# ## Step 2: Setup & Sync Repository in Google Drive
# This cell clones the repository directly into your Google Drive (if it's not already cloned) and pulls any new commits. 
# This ensures all code, checkpoints, and logs are saved permanently.

# %%
import os
import subprocess

# Define the workspace directory inside Google Drive
drive_workspace = "/content/drive/MyDrive/Adversarial-Cognitive-Model"

if os.path.exists("/content/drive"):
    if not os.path.exists(drive_workspace):
        print("Cloning repository directly into Google Drive for persistence...")
        os.chdir("/content/drive/MyDrive")
        subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git", shell=True, check=True)
    else:
        print("Repository already exists on Google Drive. Pulling latest commits...")
        os.chdir(drive_workspace)
        subprocess.run("git pull origin main", shell=True, check=True)
    
    # Change current working directory to the repository folder
    os.chdir(drive_workspace)
    print(f"Working directory set to: {os.getcwd()}")
else:
    print("Google Drive not mounted. Using local /content workspace (temporary).")
    local_workspace = "/content/Adversarial-Cognitive-Model"
    if not os.path.exists(local_workspace):
        os.chdir("/content")
        subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git", shell=True, check=True)
    os.chdir(local_workspace)
    print(f"Working directory set to: {os.getcwd()}")

# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# %% [markdown]
# ## Step 3: Environment Installation
# Installs dependencies. Since we are running directly from our persistent workspace, these python packages will be installed on the Colab container.

# %%
def run_command(cmd, shell=True):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True, text=True)
    return result.returncode

# 1. Upgrade packaging utilities first to restore legacy distutils support on Python 3.12+
run_command("pip install --upgrade pip setuptools wheel")

# 2. Install autoattack from source (directly from GitHub) to resolve Python 3.12 compatibility
run_command("pip install git+https://github.com/fra31/auto-attack.git")

# 3. Install remaining dependencies (without --quiet to see exact error logs if they fail)
run_command("pip install opencv-python gdown datasets")
run_command("pip install git+https://github.com/openai/CLIP.git")
run_command("pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git")
run_command("pip install git+https://github.com/dicarlolab/CORnet.git")

# %% [markdown]
# ## Step 4: Download & Setup UCF-101 Video Dataset
# Downloads the UCF-101 dataset to the fast local container disk (`/content/data`) so training has fast file access without FUSE lag or hitting Google Drive storage limits.

# %%
# Store dataset on local scratch disk (fastest, does not consume GDrive storage)
local_data_dir = "/content/data"
os.makedirs(local_data_dir, exist_ok=True)

ucf_dir = os.path.join(local_data_dir, "ucf101")
if not os.path.exists(ucf_dir):
    print("UCF-101 Video Dataset not found on local VM scratch space. Downloading (13GB)...")
    # Official CRC unrar flow
    run_command(f"wget --no-check-certificate -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O {local_data_dir}/UCF101.rar")
    print("Extracting UCF-101 dataset (this may take a few minutes)...")
    run_command(f"unrar x {local_data_dir}/UCF101.rar {local_data_dir}/ > /dev/null")
    if os.path.exists(f"{local_data_dir}/UCF-101"):
        os.rename(f"{local_data_dir}/UCF-101", ucf_dir)
    # Clean up rar file to save local disk space
    if os.path.exists(f"{local_data_dir}/UCF101.rar"):
        os.remove(f"{local_data_dir}/UCF101.rar")
    print("UCF-101 Dataset successfully downloaded, extracted, and cleaned up.")
else:
    print("UCF-101 Dataset already present locally on VM scratch space.")

# %% [markdown]
# ## Step 5: Launch Training (Experiment 2 or 3)
# Set your target GPU parameters. The training script automatically detects if a `_resume.pth` file exists in the `checkpoints/` folder and resumes training from the exact epoch if it was interrupted.
# 
# Adjust `--batch-size` and `--accum-steps` based on the GPU allocated:
# - **A100 (40GB VRAM)**: `--batch-size 256 --accum-steps 2` (or `--batch-size 512 --accum-steps 1`)
# - **V100 (16GB VRAM) / T4 (15GB VRAM) / L4 (24GB VRAM)**: `--batch-size 128 --accum-steps 4`
# 
# *(Note: Both batch size combinations maintain the target effective batch size of 512).*

# %%
# RUN EXPERIMENT 2: Base Model (frozen stem)
# -------------------------------------------
# Phase A: Video TDV Pretraining
print("Starting/Resuming Experiment 2 Phase A...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase tdv "
    "--model-size base "
    "--data-root /content/data "
    "--batch-size 128 "
    "--accum-steps 4"
)

# Phase B: TRADES Fine-tuning
print("Starting/Resuming Experiment 2 Phase B...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase trades "
    "--model-size base "
    "--data-root /content/data "
    "--batch-size 128 "
    "--accum-steps 4"
)

# %%
# RUN EXPERIMENT 3: Scaled Large Model
# ------------------------------------
# Phase A: Large Model Video TDV Pretraining
print("Starting/Resuming Experiment 3 Phase A...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase tdv "
    "--model-size large "
    "--data-root /content/data "
    "--batch-size 128 "
    "--accum-steps 4"
)

# Phase B: Large Model TRADES Fine-tuning
print("Starting/Resuming Experiment 3 Phase B...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase trades "
    "--model-size large "
    "--data-root /content/data "
    "--batch-size 128 "
    "--accum-steps 4"
)

# %% [markdown]
# ## Pro Tip: Prevent Colab Inactivity Disconnect
# Colab frequently disconnects if you do not interact with the window. 
# Open the browser console (`Ctrl+Shift+I` or `Cmd+Option+I` on Mac), go to the **Console** tab, paste the following JavaScript code, and press Enter:
# 
# ```javascript
# function KeepAlive() {
#     console.log("Clicking Connect button...");
#     document.querySelector("colab-connect-button").click();
# }
# setInterval(KeepAlive, 60000);
# ```
