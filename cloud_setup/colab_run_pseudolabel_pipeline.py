#!/usr/bin/env python3
"""
Google Colab Automation Pipeline for Pseudo-Label Training
==========================================================
Automates package installation, Google Drive persistent checkpoint mounting,
Hugging Face token injection, and sequential execution of the Large model + Pseudo-label training.
Designed to auto-resume seamlessly across disconnects and different accounts.
"""

import os
import sys
import argparse
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
    run_cmd("pip install --quiet opencv-python datasets huggingface_hub")
    print(">>> Python environment successfully configured.")

def setup_checkpoints_dir(use_drive=True):
    if '__file__' in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = "/content/Adversarial-Cognitive-Model"
    local_ckpt_dir = os.path.join(repo_root, "checkpoints")

    # Check if Google Drive is mounted and allowed
    gdrive_mount = "/content/drive"
    gdrive_ckpt_dir = "/content/drive/MyDrive/Adversarial-Cognitive-Model/checkpoints"

    if use_drive and os.path.exists(gdrive_mount):
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
        print(">>> Checkpoints will be saved locally on the VM.")
        os.makedirs(local_ckpt_dir, exist_ok=True)

    return local_ckpt_dir

def setup_data_dir(use_drive=True):
    local_data_dir = "/content/data"
    gdrive_mount = "/content/drive"
    gdrive_data_dir = "/content/drive/MyDrive/Adversarial-Cognitive-Model/data"

    if use_drive and os.path.exists(gdrive_mount):
        print(">>> Google Drive detected. Setting up persistent dataset storage...")
        os.makedirs(gdrive_data_dir, exist_ok=True)

        # If local data dir exists and is not a symlink, migrate files
        if os.path.exists(local_data_dir) and not os.path.islink(local_data_dir):
            print(">>> Migrating existing local dataset to Google Drive...")
            shutil.copytree(local_data_dir, gdrive_data_dir, dirs_exist_ok=True)
            shutil.rmtree(local_data_dir)

        # Create symlink
        if not os.path.exists(local_data_dir):
            os.symlink(gdrive_data_dir, local_data_dir)
            print(f">>> Symlinked {local_data_dir} -> {gdrive_data_dir}")
    else:
        os.makedirs(local_data_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Colab Pipeline for Pseudo-Label Training")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--unlabeled-batch-size', type=int, default=256)
    parser.add_argument('--accum-steps', type=int, default=16)
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--no-drive', action='store_true', help="Disable Google Drive mounting and checkpoint syncing")
    args, _ = parser.parse_known_args()

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

    # 1. Mount Google Drive if allowed and not done
    if not args.no_drive:
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
    use_drive = not args.no_drive
    setup_checkpoints_dir(use_drive=use_drive)
    setup_data_dir(use_drive=use_drive)

    # 3. Install dependencies
    install_dependencies()

    # 4. Launch Pseudo-Label Training
    print("\n>>> Launching Large Model + Pseudo-Label Training...")
    train_cmd = (
        f"python3 phase1_training/train_rhan_large_pseudolabel.py "
        f"--data-root /content/data "
        f"--batch-size {args.batch_size} "
        f"--unlabeled-batch-size {args.unlabeled_batch_size} "
        f"--accum-steps {args.accum_steps} "
        f"--confidence-threshold {args.confidence_threshold}"
    )
    if args.no_drive:
        train_cmd += " --no-drive"
    run_cmd(train_cmd)

    print("\n>>> PSEUDO-LABEL PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
