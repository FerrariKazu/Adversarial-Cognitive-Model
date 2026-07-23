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

## 1. Upgrade packaging utilities first to restore legacy distutils support on Python 3.12+
run_command("pip install --upgrade pip setuptools wheel")

# 2. Install autoattack and other necessary libraries for evaluation
run_command("pip install git+https://github.com/fra31/auto-attack.git")
run_command("pip install opencv-python scipy datasets")

# %% [markdown]
# ## Step 4: Run STL-10 Empirical Epsilon Sweep Evaluation
# Run the evaluation script across all 5 checkpoints. This will evaluate clean accuracy and robustness (using PGD-50) on 500 random test samples to calculate the $d'(\varepsilon)$ sensitivity metrics and d'=1.0 crossing thresholds.
# 
# STL-10 dataset will be automatically downloaded by torchvision during runtime.

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

# RUNTIME CONFIGURATION
# ---------------------
N_SAMPLES = 500
PGD_STEPS = 50
OUTPUT_FILE = "report/empirical_sweep_results_stl10.json"

print(f"Launching Empirical Epsilon Sweep (n={N_SAMPLES}, pgd_steps={PGD_STEPS})...")
run_interactive_command(
    f"python3 phase2_attacks/eval_empirical_epsilon_sweep.py "
    f"--n-samples {N_SAMPLES} "
    f"--pgd-steps {PGD_STEPS} "
    f"--output-json {OUTPUT_FILE} "
    f"--skip-models static_trades_large,rhan_stl10_large_ep45,rhan_v10_final"
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
