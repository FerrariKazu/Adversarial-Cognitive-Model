#!/usr/bin/env python3
"""
Colab: RHAN-v11 Loss-Ablated Evaluation
=========================================
Downloads the loss-ablated v11 checkpoint from HuggingFace and runs
the full 4-suite evaluation (statistical significance, SOTA comparison,
biological claims, diagnostic plots).

Usage: Copy cells into Colab. Set HF_TOKEN in Colab Secrets.
Runtime: ~1.5 hours on T4.
"""

import os, sys, subprocess

def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
run("pip install --quiet huggingface_hub datasets Pillow matplotlib seaborn")

REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'
if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN not found. Set it in Colab Secrets.")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from huggingface_hub import hf_hub_download
os.makedirs('checkpoints', exist_ok=True)
ckpt_path = hf_hub_download(
    repo_id='FerrariKazu/rhan-checkpoints',
    filename='rhan_stl10_v11_rolling.pth',
    repo_type='dataset', token=hf_token, local_dir='checkpoints'
)
print(f"Downloaded: {ckpt_path}")

print("\n>>> Full 4-Suite Evaluation (200 samples, 10 PGD steps)")
run(
    f"python3 phase1_training/eval_rhan_v11.py "
    f"--checkpoint {ckpt_path} "
    f"--num-samples 200 "
    f"--steps 10 "
)

print("\n✓ Evaluation complete.")
