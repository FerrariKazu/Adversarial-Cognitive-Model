#!/usr/bin/env python3
"""
Google Colab Setup & Execution Script for Experiments 2 & 3
============================================================
This script contains the Python cell commands for mounting Google Drive,
syncing checkpoints, downloading datasets, and running video TDV training.
"""

# %% [markdown]
# # Google Colab Persistent Training Template
# This notebook is optimized to run training for the **Adversarial Cognitive Model (RHAN)**.
# It ensures **no progress is lost** when Colab disconnects by:
# 1. Storing the code repository, custom files, and checkpoints directly in your **Google Drive** (`/content/drive/MyDrive`).
# 2. Storing the large dataset (UCF-101, ~13GB) on the fast local VM scratch space (`/content/data`) to prevent FUSE latency and avoid using up Google Drive space limits.
# 3. Utilizing our custom automatic resume checkpoints (`_resume.pth`) which restore model parameters, epoch progress, learning rates, and optimizer/scheduler states automatically on reconnect.

# %% [markdown]
# ## Step 1: Mount Google Drive
# Run this cell to mount your Google Drive. This enables persistent storage of your repository and checkpoints.

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab environment. Skipping Drive mount.")

# %% [markdown]
# ## Step 2: Setup & Sync Repository in Google Drive
# 
# > [!IMPORTANT]
# > **To access your checkpoints from another Google Account:**
# > 1. Log in to your **new** Google account in your browser.
# > 2. Open the shared Google Drive folder link: [Shared Folder](https://drive.google.com/drive/folders/1ufdzhOMslipYUoe3yPrRfi2xozIcK4XI).
# > 3. Click the drop-down menu next to the folder name at the top of the page, select **"Organize"** -> **"Add shortcut"**, choose **"My Drive"**, and confirm.
# > 4. Once the shortcut is added, this cell will automatically find it, sync updates, and write all checkpoints directly back to it!

# %%
import os
import subprocess
import shutil

# Define the local workspace directory (fast local VM scratch space)
local_workspace = "/content/Adversarial-Cognitive-Model"
drive_checkpoint_dir = "/content/drive/MyDrive/Adversarial-Cognitive-Model/checkpoints"

# 1. Clone or pull repository on the fast local VM disk
if not os.path.exists(local_workspace):
    print("Cloning repository to fast local VM scratch space...")
    os.chdir("/content")
    subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git", shell=True, check=True)
else:
    print("Repository exists locally. Pulling latest commits...")
    os.chdir(local_workspace)
    subprocess.run("git fetch origin main && git reset --hard origin/main", shell=True, check=True)

# Change current working directory to the local repository folder
os.chdir(local_workspace)

# 2. Setup Google Drive checkpoints persistence via symbolic link
if os.path.exists("/content/drive"):
    print("Google Drive detected. Setting up persistent checkpoints symlink...")
    # Resolve the correct drive path (check for shortcuts/shared folders if standard path doesn't exist)
    drive_base = "/content/drive/MyDrive/Adversarial-Cognitive-Model"
    if not os.path.exists(drive_base):
        for item in os.listdir("/content/drive/MyDrive"):
            full_path = os.path.join("/content/drive/MyDrive", item)
            if os.path.isdir(full_path):
                if os.path.exists(os.path.join(full_path, "phase1_training")) or "Adversarial-Cognitive-Model" in item:
                    drive_base = full_path
                    break
    
    drive_checkpoint_dir = os.path.join(drive_base, "checkpoints")
    os.makedirs(drive_checkpoint_dir, exist_ok=True)
    
    # Symlink the local checkpoints folder to Google Drive
    if os.path.exists("checkpoints"):
        if os.path.islink("checkpoints"):
            os.unlink("checkpoints")
        elif os.path.isdir("checkpoints"):
            # Move any existing local checkpoints to the persistent Drive directory
            for ckpt in os.listdir("checkpoints"):
                shutil.move(os.path.join("checkpoints", ckpt), os.path.join(drive_checkpoint_dir, ckpt))
            shutil.rmtree("checkpoints")
            
    os.symlink(drive_checkpoint_dir, "checkpoints")
    print(f"Checkpoints directory successfully symlinked to Google Drive: {drive_checkpoint_dir}")
else:
    print("Google Drive not mounted. Checkpoints will be saved locally (temporary).")
    os.makedirs("checkpoints", exist_ok=True)

print(f"Working directory successfully set to: {os.getcwd()}")

# %% [markdown]
# ## Step 3: Environment Installation
# Installs dependencies. Since we are running directly from our persistent workspace, these python packages will be installed on the Colab container.

# %%
def run_command(cmd, shell=True):
    import sys
    import re
    print(f"Executing: {cmd}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Matches any line that contains a percentage figure (wget, pip, etc.)
    progress_re = re.compile(r'\d+%')
    last_was_progress = False
    file_count = 0
    
    process = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.rstrip('\n')
            
            # Collapse wget progress lines
            if progress_re.search(line):
                # Parse wget lines into a compact summary, fall back to raw line
                m = re.search(r'(\d+)%\s+(\S+)\s+(\S+)\s*$', line)
                summary = (
                    f"  Downloading... {m.group(1)}%  |  {m.group(2)}/s  |  ETA {m.group(3)}   "
                    if m else f"  {line.strip():<80}"
                )
                sys.stdout.write(f'\r{summary}')
                sys.stdout.flush()
                last_was_progress = True
                
            # Collapse unrar file extraction logs into a single dynamic counter to prevent tab freeze
            elif line.strip().startswith("Extracting "):
                file_count += 1
                if file_count % 10 == 0:  # Update UI every 10 files to prevent printing overhead
                    sys.stdout.write(f'\r  Extracting dataset... {file_count} files extracted')
                    sys.stdout.flush()
                last_was_progress = True
                
            else:
                if last_was_progress:
                    sys.stdout.write('\n')   # seal the progress line before moving on
                    last_was_progress = False
                sys.stdout.write(output)
                sys.stdout.flush()
                
    if last_was_progress:
        sys.stdout.write('\n')
        sys.stdout.flush()
        
    rc = process.poll()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

# 1. Upgrade packaging utilities first to restore legacy distutils support on Python 3.12+
run_command("pip install --upgrade pip setuptools wheel")

# 2. Install autoattack from source (directly from GitHub) to resolve Python 3.12 compatibility
run_command("pip install git+https://github.com/fra31/auto-attack.git")

# 3. Install remaining dependencies (without --quiet to see exact error logs if they fail)
run_command("pip install opencv-python gdown datasets")
run_command("pip install git+https://github.com/openai/CLIP.git")
run_command("pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git")
run_command("pip install git+https://github.com/dicarlolab/CORnet.git")

# %% [markdown]
# ## Step 4: Download & Setup UCF-101 Video Dataset
# Downloads the UCF-101 dataset to the fast local container disk (`/content/data`) so training has fast file access without FUSE lag or hitting Google Drive storage limits.

# %%
## Store dataset on local scratch disk (fastest, does not consume GDrive storage)
local_data_dir = "/content/data"
os.makedirs(local_data_dir, exist_ok=True)

ucf_dir = os.path.join(local_data_dir, "ucf101")
if not os.path.exists(ucf_dir):
    print("UCF-101 Video Dataset not found on local VM scratch space. Downloading (13GB)...")
    print("Downloading in quiet mode (no progress logs shown to prevent browser tab crash)...")
    run_command(
        f"wget -q --no-check-certificate "
        f"https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O {local_data_dir}/UCF101.rar"
    )
    print("Download complete. Extracting dataset (this will take a few minutes)...")
    # -o+ forces unrar to overwrite existing files without prompting, keeping execution non-interactive
    run_command(f"unrar x -o+ {local_data_dir}/UCF101.rar {local_data_dir}/")
    if os.path.exists(f"{local_data_dir}/UCF-101"):
        os.rename(f"{local_data_dir}/UCF-101", ucf_dir)
    # Clean up rar file to save local disk space
    if os.path.exists(f"{local_data_dir}/UCF101.rar"):
        os.remove(f"{local_data_dir}/UCF101.rar")
    print("UCF-101 Dataset successfully downloaded, extracted, and cleaned up.")
else:
    print("UCF-101 Dataset already present locally on VM scratch space.")

# %% [markdown]
# ## Step 5: Launch Training (Experiment 2 or 3)
# Set your target GPU parameters. The training script automatically detects if a `_resume.pth` file exists in the `checkpoints/` folder and resumes training from the exact epoch if it was interrupted.
# 
# Adjust `--batch-size` and `--accum-steps` based on the GPU allocated:
# - **A100 (40GB VRAM)**: `--batch-size 256 --accum-steps 2` (or `--batch-size 512 --accum-steps 1`)
# - **V100 (16GB VRAM)**: `--batch-size 128 --accum-steps 4`
# - **T4 (15GB VRAM)**: `--batch-size 64 --accum-steps 8` (or `--batch-size 128 --accum-steps 4`)
# 
# *(Note: Both batch size combinations maintain the target effective batch size of 512).*

# %%
def run_interactive_command(cmd):
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None:
            # IPython system runner prints stdout/stderr directly in real-time
            ipy.system(cmd)
            return
    except Exception:
        pass
    # Fallback to standard subprocess
    subprocess.run(cmd, shell=True, check=True)

# %%
# %%
# RUNTIME CONFIGURATION
# ---------------------
MODEL_SIZE = "large"      # 'base' or 'large'
BATCH_SIZE_TDV = 128
BATCH_SIZE_TRADES = 256  # Set to 256 for A100/V100, 64/128 for lower end GPUs
ACCUM_STEPS_TRADES = 2   # 2 for batch_size 256, 4 for batch_size 128, 8 for batch_size 64

# Determine checkpoint paths
ckpt_dir = "checkpoints"
if MODEL_SIZE == 'large':
    tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_pretrained.pth')
    labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv_labeled.pth')
else:
    tdv_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_base_video_tdv_pretrained.pth')
    labeled_ckpt = os.path.join(ckpt_dir, 'rhan_stl10_base_video_tdv_labeled.pth')

# 1. Run Phase 0 (Video TDV Pretraining) if not complete
if not os.path.exists(tdv_ckpt):
    print(f"Starting Phase 0 (Video TDV Pretraining) for {MODEL_SIZE} model...")
    run_interactive_command(
        f"python3 phase1_training/train_rhan_video_tdv.py "
        f"--phase tdv "
        f"--model-size {MODEL_SIZE} "
        f"--data-root /content/data "
        f"--batch-size {BATCH_SIZE_TDV}"
    )
else:
    print(">>> Phase 0 (Pretraining) checkpoint found. Skipping.")

# 2. Run Phase 1 (Labeled Classifier Head Calibration) if not complete
if not os.path.exists(labeled_ckpt):
    print(f"Starting Phase 1 (Classifier Head Calibration) for {MODEL_SIZE} model...")
    run_interactive_command(
        f"python3 phase1_training/train_rhan_video_tdv.py "
        f"--phase label "
        f"--model-size {MODEL_SIZE} "
        f"--data-root /content/data"
    )
else:
    print(">>> Phase 1 (Head Calibration) checkpoint found. Skipping.")

# 3. Run Phase 2 (TRADES Adversarial Fine-Tuning)
print(f"Starting/Resuming Phase 2 (TRADES Adversarial Fine-Tuning) for {MODEL_SIZE} model...")
run_interactive_command(
    f"python3 phase1_training/train_rhan_video_tdv.py "
    f"--phase trades "
    f"--model-size {MODEL_SIZE} "
    f"--data-root /content/data "
    f"--batch-size {BATCH_SIZE_TRADES} "
    f"--accum-steps {ACCUM_STEPS_TRADES}"
)


# %% [markdown]
# ## Pro Tip: Prevent Colab Inactivity Disconnect
# Colab frequently disconnects if you do not interact with the window. 
# Open the browser console (`Ctrl+Shift+I` or `Cmd+Option+I` on Mac), go to the **Console** tab, paste the following JavaScript code, and press Enter:
# 
# ```javascript
# function KeepAlive() {
#     console.log("Clicking Connect button...");
#     document.querySelector("colab-connect-button").click();
# }
# setInterval(KeepAlive, 60000);
# ```
