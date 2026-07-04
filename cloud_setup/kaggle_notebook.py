#!/usr/bin/env python3
"""
Kaggle 2xT4 Setup & Execution Script for Experiment 3 (Large Model TRADES)
===========================================================================
This file can be run directly as a python script or imported into a Kaggle Notebook.
It uses '#' and '%%' cell markers to allow easy import into Jupyter cells.
"""

# %% [markdown]
# # Step 1: Environment Setup & Dependencies
# Installs necessary package dependencies and injects the Hugging Face Token.

# %%
import os
import sys
import subprocess
import shutil

# Fetch HF_TOKEN and inject it into the environment so the training process inherits it
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF_TOKEN successfully loaded and injected into environment.")
else:
    print("WARNING: HF_TOKEN not found in environment or secrets. Hugging Face sync will be disabled.")

def run_command(cmd, shell=True):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True, text=True)
    return result.returncode

print("Installing Python packages...")
run_command("pip install --quiet autoattack")
run_command("pip install --quiet git+https://github.com/openai/CLIP.git")
run_command("pip install --quiet git+https://github.com/wielandbrendel/bag-of-local-features-models.git")
run_command("pip install --quiet git+https://github.com/dicarlolab/CORnet.git")
run_command("pip install --quiet opencv-python datasets huggingface_hub")

# %% [markdown]
# # Step 2: Clone and Sync Repository
# Clones the repository to Kaggle's writable VM scratch disk (`/kaggle/working`)
# and pulls the latest commits.

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
REPO_URL = f'https://github.com/FerrariKazu/{REPO_NAME}.git'

os.chdir('/kaggle/working')

if not os.path.exists(f'/kaggle/working/{REPO_NAME}'):
    print('Cloning repository...')
    subprocess.run(f'git clone {REPO_URL}', shell=True, check=True)

os.chdir(f'/kaggle/working/{REPO_NAME}')
print('Syncing repository to latest commit...')
subprocess.run('git fetch origin main && git reset --hard origin/main', shell=True, check=True)

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)
print(f"Working directory successfully set to: {os.getcwd()}")

# %% [markdown]
# # Step 3: Setup UCF-101 Video Dataset
# Searches for the UCF-101 dataset inside Kaggle inputs and symlinks it.
# If not found, downloads and extracts the official UCF-101 archive.

# %%
print('Locating UCF-101 dataset...')
ucf_src_path = None
kaggle_input_dir = '/kaggle/input'
if os.path.exists(kaggle_input_dir):
    for item in os.listdir(kaggle_input_dir):
        item_path = os.path.join(kaggle_input_dir, item)
        if os.path.isdir(item_path):
            try:
                contents = os.listdir(item_path)
                if 'UCF-101' in contents or 'ucf101' in contents:
                    ucf_src_path = os.path.join(item_path, 'UCF-101' if 'UCF-101' in contents else 'ucf101')
                    break
                if any(cat in contents for cat in ['ApplyEyeMakeup', 'Archery', 'Basketball']):
                    ucf_src_path = item_path
                    break
            except Exception:
                pass

os.makedirs('data', exist_ok=True)
if ucf_src_path:
    print(f"Found mounted UCF-101 dataset in inputs at: {ucf_src_path}")
    target_link = 'data/ucf101'
    if os.path.exists(target_link):
        if os.path.islink(target_link):
            os.unlink(target_link)
        else:
            shutil.rmtree(target_link)
    os.symlink(ucf_src_path, target_link)
    print("Successfully symlinked dataset to data/ucf101.")
else:
    # Direct download fallback
    if not os.path.exists('data/ucf101'):
        print("UCF-101 dataset not found in inputs. Downloading archive...")
        download_cmd = "wget --no-check-certificate -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O data/UCF101.rar"
        try:
            subprocess.run(download_cmd, shell=True, check=True)
            print("Extracting UCF-101 dataset...")
            extract_cmd = "unrar x data/UCF101.rar data/ > /dev/null"
            subprocess.run(extract_cmd, shell=True, check=True)
            if os.path.exists('data/UCF-101'):
                os.rename('data/UCF-101', 'data/ucf101')
            print("Dataset downloaded and extracted successfully.")
        except Exception as e:
            print(f"Download/Extraction failed: {e}")
        finally:
            if os.path.exists('data/UCF101.rar'):
                os.remove('data/UCF101.rar')
    else:
        print("UCF-101 dataset already present at data/ucf101.")

# %% [markdown]
# # Step 4: Run training (Experiment 3 Phase B: Large Model TRADES)
# Uses Kaggle's dual-T4 multi-GPU capability. `nn.DataParallel` splits the batch
# size across both GPUs, so a batch size of 256 will allocate 128 samples per T4 GPU,
# fully saturating the VRAM without OOM.

# %%
print("Launching Experiment 3 Phase B...")
# Target batch size 256 + 2 gradient accumulation steps = 512 effective batch size.
cmd = (
    "python3 phase1_training/train_rhan_video_tdv.py "
    "--phase trades "
    "--model-size large "
    "--data-root ./data "
    "--batch-size 256 "
    "--accum-steps 2"
)

try:
    # Set PYTHONPATH to include project root
    os.environ["PYTHONPATH"] = f"/kaggle/working/{REPO_NAME}:{os.environ.get('PYTHONPATH', '')}"
    run_command(cmd)
    print("Training phase completed successfully.")
except Exception as e:
    print(f"Training run failed: {e}")
