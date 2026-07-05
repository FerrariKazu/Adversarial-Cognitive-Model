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
import subprocess
import shutil

def run_cmd(cmd):
    print(f"\n[RUNNING]: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Stream output in real-time
    for line in process.stdout:
        print(line, end="")
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
    
    # Check if Google Drive is mounted
    gdrive_mount = "/content/drive"
    gdrive_ckpt_dir = "/content/drive/MyDrive/Adversarial-Cognitive-Model/checkpoints"
    
    if os.path.exists(gdrive_mount):
        print(">>> Google Drive detected. Setting up persistent storage...")
        os.makedirs(gdrive_ckpt_dir, exist_ok=True)
        
        # If local checkpoints dir is a real directory and not a symlink, migrate files
        if os.path.exists(local_ckpt_dir) and not os.path.islink(local_ckpt_dir):
            print(">>> Migrating existing local checkpoints to Google Drive...")
            for f in os.listdir(local_ckpt_dir):
                src = os.path.join(local_ckpt_dir, f)
                dst = os.path.join(gdrive_ckpt_dir, f)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            shutil.rmtree(local_ckpt_dir)
            
        # Create symlink
        if not os.path.exists(local_ckpt_dir):
            os.symlink(gdrive_ckpt_dir, local_ckpt_dir)
            print(f">>> Symlinked {local_ckpt_dir} -> {gdrive_ckpt_dir}")
    else:
        print(">>> WARNING: Google Drive not mounted. Checkpoints will be saved locally on the VM.")
        os.makedirs(local_ckpt_dir, exist_ok=True)
        
    return local_ckpt_dir

def main():
    target_workspace = "/content/Adversarial-Cognitive-Model"
    
    # 0. Clone or pull repository if running in Google Colab
    if os.path.exists("/content"):
        if not os.path.exists(target_workspace):
            print(">>> Cloning repository to fast local VM scratch space...")
            os.chdir("/content")
            subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git", shell=True, check=True)
        else:
            print(">>> Repository exists locally. Pulling latest commits...")
            os.chdir(target_workspace)
            subprocess.run("git fetch origin main && git reset --hard origin/main", shell=True, check=True)

    if '__file__' in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = target_workspace
    os.chdir(repo_root)
    
    # Set Python path to include repo root
    os.environ["PYTHONPATH"] = repo_root + ":" + os.environ.get("PYTHONPATH", "")
    
    # 1. Mount Google Drive if not done
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            print(">>> Mounting Google Drive...")
            drive.mount('/content/drive')
    except ImportError:
        pass

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

    # 2. Configure symlinks
    ckpt_dir = setup_checkpoints_dir()
    
    # 3. Install packages & download datasets
    install_dependencies()
    download_and_setup_dataset()
    
    # 4. Define target paths
    tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_pretrained.pth')
    labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_labeled.pth')
    
    # 5. Phase 0: Video TDV Pretraining
    if not os.path.exists(tdv_ckpt):
        print("\n>>> Starting Phase 0 (Video TDV Pretraining)...")
        run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase tdv --model-size large --data-root /content/data --batch-size 128")
    else:
        print("\n>>> Phase 0 checkpoint already exists. Skipping pretraining.")
        
    # 6. Phase 1: Labeled Classifier Head Calibration
    if not os.path.exists(labeled_ckpt):
        print("\n>>> Starting Phase 1 (Labeled Classifier Head Calibration)...")
        run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase label --model-size large --data-root /content/data")
    else:
        print("\n>>> Phase 1 checkpoint already exists. Skipping calibration.")
        
    # 7. Phase 2: TRADES Adversarial Fine-Tuning
    print("\n>>> Starting Phase 2 (TRADES Fine-Tuning)...")
    run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase trades --model-size large --data-root /content/data --batch-size 128 --accum-steps 4")
    
    print("\n>>> PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
