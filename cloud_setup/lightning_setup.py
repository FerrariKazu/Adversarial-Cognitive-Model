#!/usr/bin/env python3
"""
Lightning.ai T4 Setup & Execution Script
========================================
This script automates the setup, requirements installation, dataset download,
and execution of Phase A (TDV pretraining) followed by Phase B (TRADES fine-tuning)
on a Lightning.ai GPU instance equipped with a 16GB T4 GPU.
"""

import os
import shutil
import subprocess

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Setup Repository
REPO_NAME = 'Adversarial-Cognitive-Model'
REPO_URL = f'https://github.com/FerrariKazu/{REPO_NAME}.git'

# Ensure we are operating inside the repository folder
if os.path.exists(REPO_NAME):
    os.chdir(REPO_NAME)
    print(f"Switched to repository folder: {os.getcwd()}")
    print("Pulling latest commits from main...")
    run_cmd("git pull origin main")
else:
    if os.path.basename(os.getcwd()) == REPO_NAME:
        print("Already inside cloned repository folder. Pulling latest commits...")
        run_cmd("git pull origin main")
    else:
        print(f"Cloning repository {REPO_NAME}...")
        run_cmd(f"git clone {REPO_URL}")
        os.chdir(REPO_NAME)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 2. Install Dependencies (Preserving Lightning's optimized PyTorch)
print('Installing requirements...')
# Install required non-torch dependencies first
run_cmd("pip install -q datasets huggingface_hub autoattack opencv-python --index-url https://pypi.org/simple")
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        reqs = f.readlines()
    filtered = []
    for r in reqs:
        r_clean = r.strip()
        # Skip package index commands and PyTorch packages to avoid overwriting optimized PyTorch
        if not r_clean or r_clean.startswith('--') or (any(pkg in r_clean for pkg in ['torch', 'torchvision', 'torchaudio']) and not any(pkg in r_clean for pkg in ['torchattacks'])):
            continue
        filtered.append(r_clean)
    
    temp_reqs_path = 'lightning_requirements.txt'
    with open(temp_reqs_path, 'w') as f:
        f.write('\n'.join(filtered))
        
    print(f"Installing filtered requirements: {filtered}")
    run_cmd(f"pip install -q -r {temp_reqs_path} --index-url https://pypi.org/simple")

# 3. Setup UCF-101 Video Dataset
print('Setting up UCF-101 dataset...')
os.makedirs('data', exist_ok=True)
if not os.path.exists('data/ucf101'):
    print("Downloading UCF-101 video dataset archive (~6.5GB)...")
    run_cmd("wget --no-check-certificate -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O data/UCF101.rar")
    print("Extracting UCF-101 dataset...")
    run_cmd("unrar x data/UCF101.rar data/ > /dev/null")
    if os.path.exists('data/UCF-101'):
        os.rename('data/UCF-101', 'data/ucf101')
    if os.path.exists('data/UCF101.rar'):
        os.remove('data/UCF101.rar')
    print("UCF-101 dataset successfully prepared.")
else:
    print("UCF-101 dataset already present at data/ucf101.")

# 4. Handle Checkpoints
expected_ckpt = 'rhan_stl10_tdv_trades.pth'
dst_ckpt = os.path.join(checkpoint_dir, expected_ckpt)

# Look for uploaded checkpoints in local or parent directories
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
src_parent_ckpt = os.path.join(parent_dir, expected_ckpt)
src_local_ckpt = os.path.join(os.getcwd(), expected_ckpt)

if os.path.exists(src_parent_ckpt):
    print(f"Found checkpoint in parent directory. Moving to {dst_ckpt}...")
    shutil.move(src_parent_ckpt, dst_ckpt)
elif os.path.exists(src_local_ckpt):
    print(f"Found checkpoint in local directory. Moving to {dst_ckpt}...")
    shutil.move(src_local_ckpt, dst_ckpt)

if os.path.exists(dst_ckpt):
    shutil.copy(dst_ckpt, os.path.join(checkpoint_dir, 'rhan_stl10_large_video_tdv.pth'))
    shutil.copy(dst_ckpt, os.path.join(checkpoint_dir, 'rhan_stl10_base_video_tdv.pth'))
    print("Checkpoint mapped to standard locations.")
else:
    print("Warning: rhan_stl10_tdv_trades.pth not found. The script will train from scratch.")

# 5. Run Pretraining (Phase A)
print("\n" + "="*80)
print("STARTING PHASE A: Video TDV Pretraining...")
print("="*80)
# T4 has 16GB VRAM, so we use batch size 128 and accum steps 4 to prevent CUDA OOM on the Large model
run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase tdv --model-size large --data-root ./data --batch-size 128 --accum-steps 4")

# 6. Run Fine-Tuning (Phase B)
print("\n" + "="*80)
print("STARTING PHASE B: TRADES Fine-Tuning...")
print("="*80)
run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase trades --model-size large --data-root ./data --batch-size 128 --accum-steps 4")

print("\nSetup and training execution completed successfully!")
