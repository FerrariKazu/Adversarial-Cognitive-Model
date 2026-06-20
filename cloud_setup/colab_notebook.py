#!/usr/bin/env python3
"""
Google Colab A100 Setup & Execution Script for Experiments 2 & 3
================================================================
This script contains the Python cell commands for mounting Google Drive,
syncing checkpoints, downloading datasets, and running video TDV training.
"""

# %% [markdown]
# # Cell 1: Mount Google Drive
# Mounts Google Drive to preserve checkpoints between disconnects.

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab environment. Skipping Drive mount.")

# %% [markdown]
# # Cell 2: Setup Symlinks for Checkpoints
# Create a symlink from Google Drive's folder to the local workspace checkpoint folder.

# %%
import os
import subprocess

# Local checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# Google Drive target folder
drive_ckpt_dir = "/content/drive/MyDrive/rhan_checkpoints"

if os.path.exists("/content/drive"):
    os.makedirs(drive_ckpt_dir, exist_ok=True)
    # Check if checkpoints is already a symlink, if not link it
    if not os.path.islink("checkpoints"):
        print("Backing up local checkpoints folder and linking to Google Drive...")
        subprocess.run("rm -rf checkpoints && ln -s /content/drive/MyDrive/rhan_checkpoints checkpoints", shell=True)
        print("Symlink created: checkpoints -> /content/drive/MyDrive/rhan_checkpoints")
    else:
        print("Checkpoints directory is already symlinked to Google Drive.")
else:
    print("Google Drive directory not found. Using local checkpoints folder.")

# %% [markdown]
# # Cell 3: Environment Installation
# Installs AutoAttack, CLIP, OpenCV and other requirements.

# %%
def run_command(cmd, shell=True):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True, text=True)
    return result.returncode

run_command("pip install --quiet autoattack opencv-python gdown")
run_command("pip install --quiet git+https://github.com/openai/CLIP.git")
run_command("pip install --quiet git+https://github.com/wielandbrendel/bag-of-local-features-models.git")
run_command("pip install --quiet git+https://github.com/dicarlolab/CORnet.git")

# %% [markdown]
# # Cell 4: Download UCF-101 Video Dataset
# Downloads the 13GB UCF-101 video dataset directly to Colab scratch space.
# Uses gdown or direct link for maximum speed.

# %%
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

ucf_dir = os.path.join(data_dir, "ucf101")
if not os.path.exists(ucf_dir):
    print("Downloading UCF-101 Video Dataset (13GB)...")
    # Recommended Google Drive shared link or official CRC source
    # Official CRC unrar flow:
    run_command("wget -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O ./data/UCF101.rar")
    print("Extracting UCF-101 dataset (this may take a few minutes)...")
    run_command("unrar x ./data/UCF101.rar ./data/ > /dev/null")
    # Rename directory if needed to match expected path './data/ucf101'
    if os.path.exists("./data/UCF-101"):
        os.rename("./data/UCF-101", ucf_dir)
    print("UCF-101 Dataset successfully downloaded and extracted.")
else:
    print("UCF-101 Dataset already present.")

# %% [markdown]
# # Cell 5: Launch Experiment 2 (Real Video TDV on Base Model)
# Phase A: Video Temporal Difference Vision (TDV) pretraining (stem frozen).
# Phase B: TRADES fine-tuning using both real STL-10 labels and video temporal consistency.

# %%
print("Launching Experiment 2 Phase A: Base Model TDV Pretraining...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase tdv "
    "--model-size base "
    "--data-root ./data "
    "--batch-size 512"
)

print("Launching Experiment 2 Phase B: Base Model TRADES Fine-tuning...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase trades "
    "--model-size base "
    "--data-root ./data "
    "--batch-size 512"
)

# %% [markdown]
# # Cell 6: Launch Experiment 3 (Scaled Large Model ViT-B scale)
# Phase A: Large Model Video TDV pretraining (~55.6M parameters).
# Phase B: Large Model TRADES Fine-tuning.

# %%
print("Launching Experiment 3 Phase A: Large Model TDV Pretraining...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase tdv "
    "--model-size large "
    "--data-root ./data "
    "--batch-size 512"
)

print("Launching Experiment 3 Phase B: Large Model TRADES Fine-tuning...")
run_command(
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase trades "
    "--model-size large "
    "--data-root ./data "
    "--batch-size 512"
)
