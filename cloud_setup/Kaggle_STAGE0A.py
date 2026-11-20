#!/usr/bin/env python3
"""
Kaggle Notebook — RHAN Stage 0A: Frozen DINOv2-S/14 and DINOv2-B/14
====================================================================
Protocol: rhan-stage0a-dinov2/1

WHAT THIS RUNS
--------------
Two frozen DINOv2 backbones, six independently trained linear heads
(2 backbones x seeds 17/42/99), ImageNet-1k, official 50k validation.
No backbone training. No integrator. No CEE.

DURABILITY
----------
  * Dedicated HF repos:
      FerrariKazu/rhan-stage0a-dinov2-checkpoints
      FerrariKazu/rhan-stage0a-dinov2-checkpoints-rolling
  * Rolling checkpoint contains model, optimizer, scheduler, AMP scaler,
    epoch/global step, Python/NumPy/torch CPU/all CUDA RNG, sampler state,
    resolved config, seed, backbone/dataset/split/code fingerprints.
  * Best checkpoint is selected on a per-class holdout carved from train.
  * Official validation is evaluated once, after training.
  * Re-running the cells after a Kaggle timeout resumes from HF automatically.
  * The run aborts rather than restarting from epoch zero if remote state is
    inaccessible or inconsistent.

GATES
-----
Full extraction/training does not start until:
  1. ImageNet-1k is found (1000 train classes, 50k official val).
  2. The checkpoint-resume drill passes.
  3. The dual-T4 smoke test shows both GPUs participating.
  4. The audit report is written (required first deliverable).

USAGE
-----
  1. Kaggle Notebook, accelerator = GPU T4 x2, Internet ON.
  2. Add-ons > Secrets > add HF_TOKEN (HuggingFace write token).
  3. Attach an ImageNet-1k dataset at /kaggle/input/imagenet-1k (or set
     STAGE0A_DATA_ROOT to its root).
  4. Run all cells in order.
"""

# %% [markdown]
# ## Step 1: Environment, secrets, and pinned code
# %%
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN missing. Add it in Kaggle Add-ons > Secrets.")
print("HF_TOKEN present:", bool(hf_token))

for pkg, mod in [
    ("huggingface_hub", "huggingface_hub"),
    ("timm>=0.9", "timm"),
    ("psutil", "psutil"),
    ("pyarrow", "pyarrow"),
    ("pandas", "pandas"),
]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# %% [markdown]
# ## Step 2: Clone the repository at the Stage 0A branch
# %%
REPO_NAME = "Adversarial-Cognitive-Model"
BRANCH = os.environ.get("STAGE0A_BRANCH", "rhan-exp")
WORK_DIR = Path("/kaggle/working") / REPO_NAME
if not WORK_DIR.exists():
    subprocess.run(
        f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git {WORK_DIR}",
        shell=True, check=True,
    )
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"
subprocess.run("git fetch origin", shell=True, check=True)
subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)
GIT_SHA = subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip()
print("Stage 0A code:", GIT_SHA)

# %% [markdown]
# ## Step 3: Hardware and storage audit
# %%
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("visible GPUs", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.2f} GB  sm_{p.major}{p.minor}")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Stage 0A requires Kaggle T4 x2; only one GPU is visible.")
subprocess.run("nvidia-smi", shell=True)
print("disk:")
subprocess.run("df -h /kaggle/working /kaggle/input 2>/dev/null || df -h /kaggle/working", shell=True)

# %% [markdown]
# ## Step 4: Locate ImageNet-1k
# %%
from stage0a.data import audit_dataset, discover_imagenet1k

candidates = [
    os.environ.get("STAGE0A_DATA_ROOT"),
    "/kaggle/input/imagenet-1k",
    "/kaggle/input/imagenet1k",
    "/kaggle/input/imagenet-1k/imagenet1k",
    "/kaggle/input/imagenet-object-localization-challenge",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC",
]
DATA_ROOT = None
for c in candidates:
    if not c:
        continue
    try:
        discover_imagenet1k(c)
        DATA_ROOT = c
        break
    except Exception:
        continue
if DATA_ROOT is None:
    raise RuntimeError(
        "ImageNet-1k not found. Attach a dataset with train/ and val/ at "
        "/kaggle/input/imagenet-1k, or set STAGE0A_DATA_ROOT. The pipeline "
        "refuses to substitute ImageNet-100."
    )
os.environ["STAGE0A_DATA_ROOT"] = DATA_ROOT
WORK_ROOT = Path(os.environ.get("STAGE0A_WORK_ROOT", "/kaggle/working"))
print("DATA_ROOT:", DATA_ROOT)
print("WORK_ROOT:", WORK_ROOT)
print(json.dumps(audit_dataset(DATA_ROOT), indent=2)[:4000])

# %% [markdown]
# ## Step 5: Dedicated Hugging Face repositories
# %%
from stage0a.hf_sync import HFStore

store = HFStore(token=os.environ["HF_TOKEN"])
store.ensure_repos()
print("best repo:   ", store.best_repo)
print("rolling repo:", store.rolling_repo)

# %% [markdown]
# ## Step 6: Build deterministic indices and the train/holdout split
# %%
def run(cmd, check=True):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(map(str, cmd))}")
    return p.returncode

COMMON = [
    sys.executable, "-m", "stage0a.cli",
    "--data-root", str(DATA_ROOT),
    "--work-root", str(WORK_ROOT),
]
# argparse subcommands precede common options; use the canonical order.
def stage_cmd(sub, *extra):
    return [
        sys.executable, "-m", "stage0a.cli", sub,
        "--data-root", str(DATA_ROOT),
        "--work-root", str(WORK_ROOT),
        *extra,
    ]

run(stage_cmd("prepare"))

# %% [markdown]
# ## Step 7: Verified checkpoint-resume drill
# The drill interrupts a 20-step synthetic run at step 7 and resumes it from
# the rolling checkpoint. It must pass before any real training.
# %%
run(stage_cmd("resume-drill"))

# %% [markdown]
# ## Step 8: Dual-T4 smoke benchmark
# Both GPUs must process images. The report records per-GPU utilization, VRAM,
# batch sizes, images/s, dataloader wait, feature-cache write throughput, and
# head-training throughput.
# %%
run(stage_cmd("smoke", "--smoke-images", "512", "--smoke-batches", "6"))

# %% [markdown]
# ## Step 9: Required first deliverable — audit report
# This writes `stage0a_audit.md` and `stage0a_audit.json` before extraction.
# %%
run(stage_cmd("audit"))
audit_md = WORK_ROOT / "stage0a" / "stage0a_audit.md"
audit_json = WORK_ROOT / "stage0a" / "stage0a_audit.json"
print(audit_md.read_text())
audit = json.loads(audit_json.read_text())
if audit.get("status") != "ready_for_training":
    raise RuntimeError(f"Stage 0A gates are not ready: {audit.get('status')}")

# %% [markdown]
# ## Step 10: Extract pooled features for both backbones
# Only the pooled CLS feature is stored. Full spatial patch-token maps are not
# persisted. Extraction is sharded across both T4s as independent processes.
# %%
run(stage_cmd("extract", "--backbone", "all"))

# %% [markdown]
# ## Step 11: Train six linear heads with automatic resume
# Two independent single-GPU training processes run at a time; DataParallel and
# DDP are intentionally avoided on T4/Turing. Each run syncs rolling/best state
# to its dedicated HF namespace.
# %%
RUN_FULL = os.environ.get("STAGE0A_RUN_FULL", "1") == "1"
if not RUN_FULL:
    raise SystemExit("Gates passed. Set STAGE0A_RUN_FULL=1 to start head training.")
run(stage_cmd("heads", "--backbone", "all", "--gpus", "2"))

# %% [markdown]
# ## Step 12: Aggregate the three seeds and apply the 80% rule
# %%
rc = run(stage_cmd("aggregate"), check=False)
agg_path = WORK_ROOT / "stage0a" / "stage0a_aggregate.json"
if agg_path.exists():
    print(agg_path.read_text())

# %% [markdown]
# ## Step 13: Final remote verification
# Every final artifact and SHA-256 sidecar must be visible remotely before the
# run is declared complete.
# %%
from stage0a.config import BACKBONES, SEEDS

missing = []
for backbone in BACKBONES:
    for seed in SEEDS:
        run_id = f"stage0a/{backbone}/seed{seed}"
        for repo, filename in [
            (store.best_repo, "final.pt"),
            (store.best_repo, "completion.json"),
            (store.best_repo, "official_val_metrics.json"),
            (store.rolling_repo, "rolling.pt"),
        ]:
            remote = store.remote_name(run_id, filename)
            status = store.status(repo, remote)
            sidecar = store.status(repo, remote + ".sha256")
            print(f"{repo}/{remote}: {status}, sidecar={sidecar}")
            if status != "present" or sidecar != "present":
                missing.append(f"{repo}/{remote}")
if missing:
    raise RuntimeError(f"remote finalization incomplete: {missing}")
print("\nSTAGE 0A COMPLETE: all final artifacts and sidecars verified.")
