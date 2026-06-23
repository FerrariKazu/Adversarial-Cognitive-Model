#!/usr/bin/env python3
"""
Kaggle Setup & Execution Script for Video TDV Pretraining (Experiment 3)
========================================================================
This script handles cloning the repository on Kaggle, installing dependencies,
locating and copying the pre-trained checkpoints from Kaggle datasets,
and starting the training process in the background.
"""

import os
import shutil
import subprocess

# 1. Setup Repository
REPO_NAME = 'Adversarial-Cognitive-Model'
REPO_URL = f'https://github.com/FerrariKazu/{REPO_NAME}.git'

# Make sure we start in Kaggle's writable working directory
os.chdir('/kaggle/working')

if not os.path.exists(f'/kaggle/working/{REPO_NAME}'):
    print('Cloning repository...')
    # Clone directly into the Kaggle working directory
    subprocess.run(f'git clone {REPO_URL}', shell=True, check=True)

os.chdir(f'/kaggle/working/{REPO_NAME}')

# Pull latest commits to stay synced
print('Pulling latest commits from main...')
subprocess.run('git pull origin main', shell=True, check=True)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 2. Install Dependencies (Resolving P100 sm_60 compatibility)
print('Installing requirements...')

# Check if pre-installed PyTorch supports P100 (sm_60) using a separate subprocess
need_torch_reinstall = False
try:
    check_cmd = "python3 -c \"import torch; cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; print('REINSTALL' if cap == (6, 0) and 'sm_60' not in torch.cuda.get_arch_list() else 'OK')\""
    res = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    if 'REINSTALL' in res.stdout:
        print("WARNING: Pre-installed PyTorch does not support P100 (sm_60) architecture.")
        need_torch_reinstall = True
except Exception:
    pass

if need_torch_reinstall:
    print("Reinstalling PyTorch with CUDA 11.8 wheels to restore sm_60 support...")
    try:
        subprocess.run('pip install --force-reinstall -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to reinstall PyTorch using CUDA 11.8 wheel.")
        raise e

# Install other required non-torch dependencies forcing official PyPI index and trusted hosts
try:
    subprocess.run('pip install -q datasets huggingface_hub autoattack opencv-python --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org', shell=True, check=True)
except subprocess.CalledProcessError as e:
    print("\n" + "=" * 80)
    print("WARNING: Dependency installation failed. This might be due to a Kaggle network mirror issue.")
    print("Since train_rhan_video_tdv.py only uses PyTorch/Torchvision for pretraining, we will proceed anyway.")
    print("=" * 80 + "\n")

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        reqs = f.readlines()
    filtered_reqs = []
    for r in reqs:
        r_clean = r.strip()
        # Skip package index commands and PyTorch packages to avoid overwriting P100-compatible PyTorch
        if not r_clean or r_clean.startswith('--') or (any(pkg in r_clean for pkg in ['torch', 'torchvision', 'torchaudio']) and not any(pkg in r_clean for pkg in ['torchattacks'])):
            continue
        filtered_reqs.append(r_clean)
    
    temp_reqs_path = 'kaggle_requirements.txt'
    with open(temp_reqs_path, 'w') as f:
        f.write('\n'.join(filtered_reqs))
    
    try:
        subprocess.run(f'pip install -q -r {temp_reqs_path} --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org', shell=True, check=True)
    except subprocess.CalledProcessError:
        print("WARNING: requirements.txt installation failed. Proceeding anyway...")

# 3. Setup UCF-101 Video Dataset
print('Setting up UCF-101 dataset...')
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
        print("UCF-101 dataset not found in inputs. Downloading official archive (~6.5GB)...")
        download_cmd = "wget --no-check-certificate -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O data/UCF101.rar"
        try:
            subprocess.run(download_cmd, shell=True, check=True)
            print("Extracting UCF-101 dataset (this may take a few minutes)...")
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
                print("Cleaned up download archive to save space.")
    else:
        print("UCF-101 dataset already present at data/ucf101.")

# 4. Handle Checkpoint from Kaggle Dataset
expected_ckpt = 'rhan_stl10_tdv_trades.pth'
dst_ckpt = os.path.join(checkpoint_dir, expected_ckpt)

# Search robustly for the checkpoint under Kaggle inputs
src_ckpt = None
if os.path.exists(kaggle_input_dir):
    print("Searching for checkpoint in Kaggle inputs...")
    for root, dirs, files in os.walk(kaggle_input_dir):
        if expected_ckpt in files:
            src_ckpt = os.path.join(root, expected_ckpt)
            break

if src_ckpt:
    print(f'Found checkpoint at: {src_ckpt}')
    print(f'Copying checkpoint to: {dst_ckpt}')
    # Kaggle input is read-only, so we copy instead of move
    shutil.copy(src_ckpt, dst_ckpt)
    
    # Also copy to large/base names in case they are referenced directly by the script
    shutil.copy(src_ckpt, os.path.join(checkpoint_dir, 'rhan_stl10_large_video_tdv.pth'))
    shutil.copy(src_ckpt, os.path.join(checkpoint_dir, 'rhan_stl10_base_video_tdv.pth'))
    print("Checkpoint mapped to standard locations successfully.")
else:
    # Fallback check if it was uploaded to /kaggle/working/ directly
    fallback_src = os.path.join('/kaggle/working/', expected_ckpt)
    if os.path.exists(fallback_src):
        print(f'Moving checkpoint from {fallback_src} to {dst_ckpt}')
        shutil.move(fallback_src, dst_ckpt)
        shutil.copy(dst_ckpt, os.path.join(checkpoint_dir, 'rhan_stl10_large_video_tdv.pth'))
        shutil.copy(dst_ckpt, os.path.join(checkpoint_dir, 'rhan_stl10_base_video_tdv.pth'))
    else:
        print(f'Warning: {expected_ckpt} not found.')
        print("Please ensure you have added the 'RHAN-Cloud' dataset to this Kaggle notebook.")

# 4. Run Training in Background
print('Starting training in background (nohup)...')
print('Monitoring script: train_rhan_video_tdv.py')
print('You can monitor progress by running: !tail -n 20 -f training_log.out')

# Command for video pretraining phase
training_cmd = """nohup python3 phase1_training/train_rhan_video_tdv.py \
    --phase tdv \
    --model-size large \
    --data-root ./data \
    --batch-size 128 \
    --accum-steps 4 > training_log.out 2>&1 &"""

os.system(training_cmd)
