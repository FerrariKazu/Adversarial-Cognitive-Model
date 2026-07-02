#!/usr/bin/env python3
"""
Google Colab Automation Pipeline
================================
Automates Google Drive mounting, checkpoint symlinking, and sequential execution
of Phase 0 (Pretraining), Phase 1 (Calibration), and Phase 2 (TRADES).
Designed to be robust and auto-resume if the session disconnects.
"""

import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"\n[RUNNING]: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Stream output in real-time
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd}")

def setup_checkpoints_dir():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_ckpt_dir = os.path.join(repo_root, "checkpoints")
    
    # Check if Google Drive is mounted
    gdrive_mount = "/content/drive"
    gdrive_ckpt_dir = "/content/drive/MyDrive/checkpoints"
    
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
                    subprocess.run(f"cp '{src}' '{dst}'", shell=True)
            subprocess.run(f"rm -rf '{local_ckpt_dir}'", shell=True)
            
        # Create symlink
        if not os.path.exists(local_ckpt_dir):
            subprocess.run(f"ln -s '{gdrive_ckpt_dir}' '{local_ckpt_dir}'", shell=True)
            print(f">>> Symlinked {local_ckpt_dir} -> {gdrive_ckpt_dir}")
    else:
        print(">>> WARNING: Google Drive not mounted. Checkpoints will be saved locally on the VM.")
        os.makedirs(local_ckpt_dir, exist_ok=True)
        
    return local_ckpt_dir

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
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

    # 2. Configure symlinks
    ckpt_dir = setup_checkpoints_dir()
    
    # 3. Define target paths
    tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_pretrained.pth')
    labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_labeled.pth')
    final_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv.pth')
    
    # 4. Phase 0: Video TDV Pretraining
    if not os.path.exists(tdv_ckpt):
        print("\n>>> Starting Phase 0 (Video TDV Pretraining)...")
        run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase tdv --model-size large --data-root /content/data --batch-size 128")
    else:
        print("\n>>> Phase 0 checkpoint already exists. Skipping pretraining.")
        
    # 5. Phase 1: Labeled Classifier Head Calibration
    if not os.path.exists(labeled_ckpt):
        print("\n>>> Starting Phase 1 (Labeled Classifier Head Calibration)...")
        run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase label --model-size large --data-root /content/data")
    else:
        print("\n>>> Phase 1 checkpoint already exists. Skipping calibration.")
        
    # 6. Phase 2: TRADES Adversarial Fine-Tuning
    print("\n>>> Starting Phase 2 (TRADES Fine-Tuning)...")
    # Will automatically resume if interrupted using the built-in resume checkpoint
    run_cmd("python3 phase1_training/train_rhan_video_tdv.py --phase trades --model-size large --data-root /content/data --batch-size 256")
    
    print("\n>>> PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
