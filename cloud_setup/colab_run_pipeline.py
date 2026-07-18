#!/usr/bin/env python3
"""
Google Colab Automation Pipeline
================================
Automates package installation, dataset downloading, Google Drive mounting, 
checkpoint symlinking, and sequential execution of Phase 0, 1, and 2.
Designed to be robust and auto-resume if the session disconnects.
"""

import os
import sys
import argparse
import subprocess
import shutil

# Disable Hugging Face Hub progress bars to keep output silent and clean
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def run_cmd(cmd):
    print(f"\n[RUNNING]: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )
    
    while True:
        char = process.stdout.read(1)
        if not char:
            break
        sys.stdout.write(char)
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd}")

def install_dependencies():
    print("\n>>> Installing Python Dependencies...")
    run_cmd("pip install --quiet --upgrade pip setuptools wheel")
    run_cmd("pip install --quiet git+https://github.com/fra31/auto-attack.git")
    run_cmd("pip install --quiet opencv-python gdown datasets")
    run_cmd("pip install --quiet git+https://github.com/openai/CLIP.git")
    run_cmd("pip install --quiet git+https://github.com/wielandbrendel/bag-of-local-features-models.git")
    run_cmd("pip install --quiet git+https://github.com/dicarlolab/CORnet.git")
    print(">>> Python environment successfully configured.")

def download_and_setup_dataset():
    local_data_dir = "/content/data"
    os.makedirs(local_data_dir, exist_ok=True)
    ucf_dir = os.path.join(local_data_dir, "ucf101")
    
    if not os.path.exists(ucf_dir):
        print("\n>>> UCF-101 dataset not found. Downloading (13GB)...")
        rar_path = os.path.join(local_data_dir, "UCF101.rar")
        run_cmd(f"wget -q --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O {rar_path}")
        
        print("\n>>> Extracting dataset (this will take a few minutes)...")
        run_cmd(f"unrar x -o+ {rar_path} {local_data_dir}/")
        
        extracted_dir = os.path.join(local_data_dir, "UCF-101")
        if os.path.exists(extracted_dir):
            os.rename(extracted_dir, ucf_dir)
            
        if os.path.exists(rar_path):
            os.remove(rar_path)
        print(">>> UCF-101 dataset successfully loaded and extracted.")
    else:
        print("\n>>> UCF-101 dataset already present on scratch disk.")

def setup_checkpoints_dir():
    if '__file__' in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = "/content/Adversarial-Cognitive-Model"
    local_ckpt_dir = os.path.join(repo_root, "checkpoints")
    
    print(">>> Checkpoints will be saved locally on the VM.")
    if os.path.islink(local_ckpt_dir):
        print(f">>> Removing existing Google Drive symlink for checkpoints: {local_ckpt_dir}")
        os.unlink(local_ckpt_dir)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    
    return local_ckpt_dir

def main():
    parser = argparse.ArgumentParser(description="Colab Pipeline")
    args, _ = parser.parse_known_args()

    target_workspace = "/content/Adversarial-Cognitive-Model"
    
    # 0. Clone repository if running in Google Colab and workspace is missing
    if os.path.exists("/content"):
        if not os.path.exists(target_workspace):
            print(">>> Cloning repository to fast local VM scratch space...")
            os.chdir("/content")
            subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git", shell=True, check=True)
        else:
            print(">>> Repository exists locally. Pulling latest commits from GitHub...")
            os.chdir(target_workspace)
            subprocess.run("git pull origin main", shell=True)

    if '__file__' in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = target_workspace
    os.chdir(repo_root)
    
    # Set Python path to include repo root
    os.environ["PYTHONPATH"] = repo_root + ":" + os.environ.get("PYTHONPATH", "")

    # Inject Hugging Face Token from Colab secrets securely if available
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            os.environ["HF_TOKEN"] = token
            print(">>> HF_TOKEN successfully loaded from Colab Secrets.")
        else:
            print(">>> WARNING: HF_TOKEN not found in Colab Secrets. Check your Secrets tab (key icon).")
    except Exception:
        pass

    # 2. Configure checkpoints directory
    ckpt_dir = setup_checkpoints_dir()
    
    # 3. Install packages (STL-10 downloads automatically during training, so we skip UCF-101 download)
    install_dependencies()
    
    # 4. Run RHAN-v10 Curriculum Training
    print("\n>>> Starting RHAN-v10 Curriculum Training (60 Epochs)...")
    # Using the pre-registered curriculum args without --force-restart to allow auto-resume
    run_cmd("python3 phase1_training/train_rhan_v10.py "
            "--target-ckpt checkpoints_tier2/rhan_stl10_large_pseudolabel_best.pth "
            "--batch-size 8 "
            "--accum-steps 32")
    
    # 5. Run RHAN-v10 Evaluation and Diagnostic Generation
    print("\n>>> Starting RHAN-v10 Evaluation and Scientific Claim Verification...")
    print("\n>>> Evaluating BEST checkpoint...")
    run_cmd("python3 phase1_training/eval_rhan_v10.py "
            "--checkpoint checkpoints/rhan_stl10_v10_best.pth")
    
    print("\n>>> Evaluating FINAL ROLLING checkpoint (60th Epoch)...")
    run_cmd("python3 phase1_training/eval_rhan_v10.py "
            "--checkpoint checkpoints/rhan_stl10_v10_rolling.pth")
    
    print("\n>>> RHAN-V10 PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
