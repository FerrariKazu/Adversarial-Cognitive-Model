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

# 2. Install Dependencies
print('Installing requirements...')
subprocess.run('pip install -q datasets huggingface_hub autoattack opencv-python', shell=True, check=True)
if os.path.exists('requirements.txt'):
    subprocess.run('pip install -q -r requirements.txt', shell=True, check=True)

# 3. Handle Checkpoint from Kaggle Dataset
expected_ckpt = 'rhan_stl10_tdv_trades.pth'
dst_ckpt = os.path.join(checkpoint_dir, expected_ckpt)

# Search robustly for the checkpoint under Kaggle inputs
src_ckpt = None
kaggle_input_dir = '/kaggle/input'
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
    --batch-size 512 > training_log.out 2>&1 &"""

os.system(training_cmd)
