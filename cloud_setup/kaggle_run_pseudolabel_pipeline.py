#!/usr/bin/env python3
"""
Kaggle Automation Pipeline for Pseudo-Label Training
=====================================================
Automates package installation, GitHub synchronization,
Hugging Face token injection from Kaggle Secrets, and training execution.
Optimized for Kaggle's dual-T4 GPUs (batch size 32, gradient accumulation 8).
"""

# %%
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

def main():
    # Detect available GPUs to adjust batch size dynamically and avoid OOM on single-GPU session
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    default_bs = 32 if num_gpus >= 2 else 16
    default_accum = 8 if num_gpus >= 2 else 16
    
    print(f"\n>>> Hardware Detection: Found {num_gpus} GPU(s).")
    print(f"    Setting default batch-size={default_bs}, accum-steps={default_accum} "
          f"(effective batch size = {default_bs * default_accum}).")

    parser = argparse.ArgumentParser(description="Kaggle Pipeline for Pseudo-Label Training")
    parser.add_argument('--batch-size', type=int, default=default_bs, help=f'Batch size across active GPUs (default: {default_bs})')
    parser.add_argument('--unlabeled-batch-size', type=int, default=256)
    parser.add_argument('--accum-steps', type=int, default=default_accum, help=f'Gradient accumulation steps (default: {default_accum})')
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    args, _ = parser.parse_known_args()

    target_workspace = "/kaggle/working/Adversarial-Cognitive-Model"

    # 0. Clone or pull repository
    if not os.path.exists(target_workspace):
        print(">>> Cloning repository to Kaggle scratch directory...")
        os.chdir("/kaggle/working")
        subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git", shell=True, check=True)
    else:
        print(">>> Repository exists. Pulling latest commits...")
        os.chdir(target_workspace)
        subprocess.run("git fetch origin main && git reset --hard origin/main", shell=True, check=True)

    if '__file__' in globals():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = target_workspace
    os.chdir(repo_root)

    # Set Python path to include repo root
    os.environ["PYTHONPATH"] = repo_root + ":" + os.environ.get("PYTHONPATH", "")

    # Inject Hugging Face Token from Kaggle secrets securely if available
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print(">>> HF_TOKEN successfully loaded from Kaggle Secrets.")
        else:
            print(">>> WARNING: HF_TOKEN not found in Kaggle Secrets.")
    except Exception:
        pass

    # 1. Install dependencies
    install_dependencies()

    # 2. Launch Pseudo-Label Training (supports nn.DataParallel across dual-T4 GPUs automatically)
    print("\n>>> Launching Large Model + Pseudo-Label Training...")
    train_cmd = (
        f"python3 phase1_training/train_rhan_large_pseudolabel.py "
        f"--data-root ./data "
        f"--batch-size {args.batch_size} "
        f"--unlabeled-batch-size {args.unlabeled_batch_size} "
        f"--accum-steps {args.accum_steps} "
        f"--confidence-threshold {args.confidence_threshold}"
    )
    run_cmd(train_cmd)

    print("\n>>> PSEUDO-LABEL PIPELINE EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
