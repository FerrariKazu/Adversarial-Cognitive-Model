#!/usr/bin/env python3
"""
Colab Notebook — RHAN Stage 0A: Frozen DINOv2-S/14 and DINOv2-B/14
===================================================================
Protocol: rhan-stage0a-dinov2/1

This is the single-GPU Colab twin of cloud_setup/Kaggle_STAGE0A.py.  It uses
the same stage0a package, the same resolved configuration, the same HF repos,
and the same checkpoint format.  The only runtime differences are single-GPU
sharding and a warning in the smoke gate.

USAGE
-----
  1. Colab runtime with one GPU.
  2. Add HF_TOKEN to Colab Secrets and enable notebook access.
  3. Attach/mount ImageNet-1k at /content/imagenet-1k, or set
     STAGE0A_DATA_ROOT.
  4. Run all cells in order.
"""

# %% [markdown]
# ## Step 1: Environment and secrets
# %%
import os
import sys
import json
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN missing. Add it in Colab Secrets and enable notebook access.")
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
# ## Step 2: Clone the same Stage 0A code
# %%
REPO_NAME = "Adversarial-Cognitive-Model"
BRANCH = os.environ.get("STAGE0A_BRANCH", "rhan-exp")
WORK_DIR = Path("/content") / REPO_NAME
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
print("Stage 0A code:", subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip())

# %% [markdown]
# ## Step 3: GPU and storage check
# %%
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("visible GPUs", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    raise RuntimeError("No GPU visible. Set Runtime > Change runtime type > GPU.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.2f} GB")
subprocess.run("nvidia-smi", shell=True)
subprocess.run("df -h /content", shell=True)

# %% [markdown]
# ## Step 4: Locate ImageNet-1k
# %%
from stage0a.data import audit_dataset, describe_search_root, find_imagenet1k

search_roots = [
    p for p in [
        os.environ.get("STAGE0A_DATA_ROOT"),
        "/content/imagenet-1k",
        "/content/imagenet1k",
        "/content/drive/MyDrive/imagenet-1k",
        "/content",
    ] if p
]
DATA_ROOT = find_imagenet1k(search_roots, max_depth=5)
if DATA_ROOT is None:
    details = "\n\n".join(describe_search_root(p, max_depth=3) for p in search_roots[:10])
    raise RuntimeError(
        "ImageNet-1k not found. Attach it under /content or Drive, or set "
        "STAGE0A_DATA_ROOT. The pipeline refuses to substitute ImageNet-100.\n\n"
        f"--- Inspected Paths ---\n{details}\n------------------------"
    )
os.environ["STAGE0A_DATA_ROOT"] = str(DATA_ROOT)
WORK_ROOT = Path(os.environ.get("STAGE0A_WORK_ROOT", "/content"))
print("DATA_ROOT:", DATA_ROOT)
print("WORK_ROOT:", WORK_ROOT)
dataset_audit = audit_dataset(DATA_ROOT)
print(json.dumps(dataset_audit, indent=2)[:4000])
if dataset_audit.get("status") != "ok":
    raise RuntimeError(f"ImageNet-1k dataset gate failed: {dataset_audit}")

# %% [markdown]
# ## Step 5: HF repositories (same namespaces and format as Kaggle)
# %%
from stage0a.hf_sync import HFStore

store = HFStore(token=os.environ["HF_TOKEN"])
store.ensure_repos()
print("best repo:   ", store.best_repo)
print("rolling repo:", store.rolling_repo)

# %% [markdown]
# ## Step 6: Indices, split, and gates
# %%
def stage_cmd(sub, *extra):
    return [
        sys.executable, "-m", "stage0a.cli", sub,
        "--data-root", str(DATA_ROOT),
        "--work-root", str(WORK_ROOT),
        *extra,
    ]

def run(cmd, check=True):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}")
    return p.returncode

run(stage_cmd("prepare"))
run(stage_cmd("resume-drill"))
# Colab is single-GPU by definition; this is not a dual-T4 claim.
run(stage_cmd("smoke", "--allow-single-gpu", "--smoke-images", "256", "--smoke-batches", "4"))
run(stage_cmd("audit"))
print((WORK_ROOT / "stage0a" / "stage0a_audit.md").read_text())
audit = json.loads((WORK_ROOT / "stage0a" / "stage0a_audit.json").read_text())
# Colab deliberately has no dual-T4 gate; it must not claim dual-GPU results.
if audit.get("dataset", {}).get("status") != "ok":
    raise RuntimeError(f"Stage 0A dataset gate failed: {audit.get('dataset')}")

# %% [markdown]
# ## Step 7: Extract pooled features
# %%
run(stage_cmd("extract", "--backbone", "all"))

# %% [markdown]
# ## Step 8: Train six linear heads
# %%
if os.environ.get("STAGE0A_RUN_FULL", "1") != "1":
    raise SystemExit("Gates passed. Set STAGE0A_RUN_FULL=1 to start training.")
run(stage_cmd("heads", "--backbone", "all", "--gpus", "1"))

# %% [markdown]
# ## Step 9: Aggregate and verify
# %%
run(stage_cmd("aggregate"), check=False)
agg_path = WORK_ROOT / "stage0a" / "stage0a_aggregate.json"
if agg_path.exists():
    print(agg_path.read_text())
