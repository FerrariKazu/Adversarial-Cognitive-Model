#!/usr/bin/env python3
"""
Colab Notebook — RHAN-NXA GENERATION 2 FOUNDATION (S0-S7 Pipeline)
==================================================================

Colab twin of Kaggle_J1_FOUNDATION.py (identical protocol, platform-native secrets/paths).
Runs the complete Gen-2 Foundation ladder:
  1. g2_gist_only (30 eps, clean warmup, D1 deviation, Gate C @ ep 10, Gate R @ ep 17)
  2. recurrence_only (15 eps, T=4 fixed glimpses, TRADES curriculum)
  3. belief_no_f (15 eps, evidential uncertainty readout)
  4. belief_with_f (15 eps, learned precision + UpdateNet belief dynamics)
  5. ais_v2_swap (15 eps, AIS-v2 expected uncertainty reduction gaze policy)
  6. gen1_core (15 eps, full Gen-2 core loop)

DURABILITY & HF SYNC:
  - Partitions Gen-2 into dedicated HuggingFace repos:
      * FerrariKazu/rhan-nxa-g2-checkpoints          (final & best checkpoints)
      * FerrariKazu/rhan-nxa-g2-checkpoints-rolling  (rolling state + JSONL log)
  - Colab's /content is wiped when the runtime disconnects:
    Checkpoints and jsonl logs sync to HuggingFace after each epoch.
  - Resume is automatic: Re-running the dispatch cell restores rolling state from HF
    and continues without starting from scratch.

USAGE ON GOOGLE COLAB:
  1. New Colab Notebook, Runtime > Change runtime type > T4 GPU.
  2. Secrets (key icon in left sidebar) > Add new secret 'HF_TOKEN' with notebook access.
  3. Run all cells in sequence:
     Step 1: Environment & Secrets
     Step 2: Clone & Checkout feature/gen2-rebuild
     Step 3: Download official DINOv2-small warm-start checkpoint
     Step 4: ImageNet-100 dataset bootstrap
     Step 5: Smoke pre-flight verification
     Step 6: DISPATCH — Gen-2 Foundation Training Loop
     Step 7: S7 Comprehensive Evaluation Protocol
"""

# %% [markdown]
# ## Step 1: Environment, Dependencies & HuggingFace Secrets

# %%
import os
import sys
import subprocess
import shutil
import time
import json
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# HuggingFace token from Colab Secrets or Environment
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

if not hf_token:
    raise RuntimeError(
        "HF_TOKEN not found! Add your HuggingFace write token in Colab: "
        "Secrets (key icon in the sidebar) > Add secret 'HF_TOKEN' > Enable notebook access."
    )

print(f"✓ HF_TOKEN authenticated (starts with: {hf_token[:4]}...)")

# GPU Check
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU Accelerator Active: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("⚠ WARNING: No GPU detected! Go to Runtime > Change runtime type > T4 GPU.")

# Install missing lightweight dependencies
for mod, pkg in [("huggingface_hub", "huggingface_hub"), ("pyarrow", "pyarrow"), ("pandas", "pandas")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run(["pip", "install", "--quiet", pkg], check=True)

# %% [markdown]
# ## Step 2: Clone & Checkout branch `feature/gen2-rebuild`

# %%
REPO_NAME = "Adversarial-Cognitive-Model"
WORK_DIR = f"/content/{REPO_NAME}"
BRANCH = "feature/gen2-rebuild"

os.chdir("/content")
if not os.path.exists(WORK_DIR):
    print(f"Cloning FerrariKazu/{REPO_NAME}...")
    subprocess.run(f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git", shell=True, check=True)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

print(f"Fetching and checking out branch '{BRANCH}'...")
subprocess.run("git fetch origin", shell=True, check=True)
subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)

sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()
print(f"✓ Checked out {BRANCH} @ commit {sha}")

# %% [markdown]
# ## Step 3: Official DINOv2-small Warm-Start Checkpoint

# %%
DINO_PATH = "checkpoints/dinov2_vits14_pretrain.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("report", exist_ok=True)

if not os.path.exists(DINO_PATH) or os.path.getsize(DINO_PATH) < 80 * 1024 * 1024:
    print(f"Downloading official DINOv2-small weights (~88MB) to {DINO_PATH}...")
    subprocess.run(f"wget -c {DINO_URL} -O {DINO_PATH}", shell=True, check=True)

print(f"✓ DINOv2 checkpoint ready ({os.path.getsize(DINO_PATH) / 1e6:.1f} MB)")

# %% [markdown]
# ## Step 4: HuggingFace Partitioned Repository Initialization

# %%
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

HF_BEST = "FerrariKazu/rhan-nxa-g2-checkpoints"
HF_ROLLING = "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"

api = HfApi(token=hf_token)

# Auto-create repos if they don't already exist
for r in [HF_BEST, HF_ROLLING]:
    try:
        api.create_repo(repo_id=r, repo_type="dataset", private=True, exist_ok=True)
        print(f"✓ HF repo ready: {r}")
    except Exception as e:
        print(f"  HF repo check for {r}: {e}")

def restore_checkpoints_from_hf():
    """Download rolling checkpoints and logs from HF if resuming a session."""
    try:
        files = list_repo_files(repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        restored = 0
        for f in files:
            if f.endswith(".pth") or f.endswith(".jsonl"):
                local_f = os.path.join("checkpoints", os.path.basename(f)) if f.endswith(".pth") else os.path.join("report", os.path.basename(f))
                if not os.path.exists(local_f):
                    print(f"  Restoring {f} from HF...")
                    p = hf_hub_download(repo_id=HF_ROLLING, filename=f, repo_type="dataset", token=hf_token)
                    shutil.copyfile(p, local_f)
                    restored += 1
        print(f"✓ Restored {restored} checkpoint/log file(s) from HF.")
    except Exception as e:
        print(f"  Note: No prior rolling checkpoints found on HF (starting fresh): {e}")

def sync_checkpoints_to_hf():
    """Upload new checkpoints and JSONL log to HF."""
    # Rolling checkpoints + logs -> HF_ROLLING
    for cp in Path("checkpoints").glob("*_rolling.pth"):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name, repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: failed to sync {cp.name} to rolling HF: {e}")

    log_path = Path("report/gen2_epoch_log.jsonl")
    if log_path.exists():
        try:
            api.upload_file(path_or_fileobj=str(log_path), path_in_repo="gen2_epoch_log.jsonl", repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: failed to sync log to HF: {e}")

    # Best / Final checkpoints -> HF_BEST
    for cp in list(Path("checkpoints").glob("*_final.pth")) + list(Path("checkpoints").glob("*_best_*.pth")):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name, repo_id=HF_BEST, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: failed to sync {cp.name} to best HF: {e}")

restore_checkpoints_from_hf()

# %% [markdown]
# ## Step 5: DATA Bootstrap (Pinned ImageNet-100 source to /content/imagenet100)

# %%
DATA_ROOT = os.environ.get("J1_DATA_ROOT", "/content/imagenet100")
CONVERTER = "scripts/prepare_imagenet100.py"

def is_dataset_ready(root: str) -> bool:
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        return False
    return len(os.listdir(train_dir)) == 100 and len(os.listdir(val_dir)) == 100

if is_dataset_ready(DATA_ROOT):
    print(f"✓ ImageNet-100 validated at {DATA_ROOT} (100 classes in train/val).")
else:
    print(f"== Bootstrapping ImageNet-100 pinned source into {DATA_ROOT} ==")
    os.makedirs(DATA_ROOT, exist_ok=True)
    subprocess.run(f"python3 {CONVERTER} --root {DATA_ROOT}", shell=True, check=True)
    assert is_dataset_ready(DATA_ROOT), f"Dataset validation failed at {DATA_ROOT}!"
    print(f"✓ ImageNet-100 bootstrap complete.")

# %% [markdown]
# ## Step 6: THE DISPATCH — Launch Gen-2 Foundation Ladder

# %%
BATCH_SIZE = int(os.environ.get("GEN2_BATCH", "48"))
NUM_WORKERS = int(os.environ.get("GEN2_WORKERS", "4"))
ARM = os.environ.get("GEN2_ARM", "v1")

print(f"Launching Gen-2 Foundation ladder:")
print(f"  data_root:  {DATA_ROOT}")
print(f"  batch_size: {BATCH_SIZE}")
print(f"  arm:        {ARM}")
print(f"  dino:       {DINO_PATH}")

from training.train_gen2_foundation import PHASE_LADDER, run_gen2_phase

for phase in PHASE_LADDER:
    final_pth = f"checkpoints/foundation_{phase}_final.pth"
    if os.path.exists(final_pth):
        print(f"\n✓ Phase '{phase}' is already complete ({final_pth} exists). Skipping to next phase.")
        continue

    print(f"\n{'='*70}\n[Gen-2 Dispatch] Starting Phase: {phase}\n{'='*70}")
    res = run_gen2_phase(
        phase=phase,
        arm=ARM,
        data_root=DATA_ROOT,
        ckpt_dir="checkpoints",
        report_dir="report",
        dino_path=DINO_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        smoke=False,
    )

    # Sync durable checkpoints to HuggingFace after each completed phase
    print(f"Syncing completed phase '{phase}' to HuggingFace...")
    sync_checkpoints_to_hf()

print("\n" + "=" * 70)
print("✓ ALL SIX GEN-2 FOUNDATION PHASES COMPLETE!")
print("=" * 70)

# %% [markdown]
# ## Step 7: S7 Evaluation Protocol (8 Seeds × 300 Images + PGD-50 + EOT-PGD)

# %%
final_ckpt = "checkpoints/foundation_gen1_core_final.pth"
if not os.path.exists(final_ckpt):
    final_ckpt = "checkpoints/foundation_gen1_core_best_clean.pth"

if os.path.exists(final_ckpt):
    print(f"\nRunning S7 Evaluation Protocol on: {final_ckpt}")
    cmd = f"python3 training/eval_gen2_foundation.py --ckpt-path {final_ckpt} --phase gen1_core --arm {ARM} --data-root {DATA_ROOT} --final-verify"
    subprocess.run(cmd, shell=True, check=True)
    sync_checkpoints_to_hf()
else:
    print(f"Final checkpoint {final_ckpt} not yet generated. Complete Step 6 first.")
