#!/usr/bin/env python3
"""
Colab Notebook -- Consolidated 112px Recurrence Flagship Training & Rigorous Evaluation
========================================================================================
Protocol ID: consolidated-flagship-112/1
Target Model: recurrence_only at 112px (4 shifted views, fovea = full frame)
Baselines:    ff_112 (seeds 17, 42, 99)

THE CONSOLIDATED FLAGSHIP SPECIFICATION
---------------------------------------
Part         Choice                                                 Evidence
Backbone     DINOv2-S/14 warm start, matched fraction >=95% logged  proven
Input        112 px, 4 shifted views at existing offsets, fovea=112  resolution proven
Fusion       gist prior z_0, EMA 0.5/0.5, tied refinement x2        proven pipeline
Recipe       TRADES beta=2.5, eps=5.417/255 (0.0940 norm), PGD-4,   proven recipe
             Rung 3 from epoch 1, 15 epochs, same data as ff_112
Evaluation   T always 4, never different                            T-sweep
Excluded     belief, precision, gaze policy, uncertainty, halting   inert / unwired

TRAINING RULES
--------------
1. 3 seeds: 17, 42, 99.
2. Final-epoch checkpoint only (epoch 15) with NO selection on eval images.
3. Fingerprint every checkpoint (phase, eps, beta, epochs, seed, data hash, git commit).
4. Abort guardrails:
   - Abort if eps != 0.0940 +/- 0.001 or KL <= 0.0 on any adversarial batch.
   - Abort if PGD-10 training monitor robust accuracy < 15.0% by epoch 3.
5. Gate 5k clean accuracy against the epoch-15 monitor (within 1.0 pp).
6. Effective batch size = 64 (loader batch; no gradient accumulation).

PRE-REGISTERED DECISION RULE
----------------------------
Flagship is declared BETTER only if:
  1. min(recurrence_112) > max(ff_112) on worst-case robust accuracy across all 5,000 images.
  2. Pooled McNemar 95% confidence interval on discordant pairs excludes zero (p < 0.01).
Otherwise:
  Conclude that at 4x token compute, recurrent fusion does not earn its cost at 112px.

USAGE ON GOOGLE COLAB
---------------------
  1. New Colab notebook. Runtime > Change runtime type > T4 GPU (or A100/V100).
  2. Secrets (key icon in left sidebar) > Add secret "HF_TOKEN" > Enable notebook access.
  3. Run all cells in sequence (or run via terminal: `python3 cloud_setup/colab_consolidated.py`).
     Set `COLAB_COMMIT` to the full 40-character SHA of the commit to run. The launcher aborts when it is unset, is not a full SHA, does not match HEAD, or the working tree is dirty.
  4. Disconnect resilience:
     - Live sync uploads rolling checkpoints & logs to HuggingFace after every epoch.
     - On restart, Step 4 pulls the latest rolling checkpoint and training automatically resumes.
     - Completed final checkpoints (epoch 15) are cached and never re-trained.
"""

# %% [markdown]
# ## Step 1: Environment, Dependencies & HuggingFace Secrets

# %%
import os
import re
import sys
import json
import time
import math
import shutil
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

SESSION_T0 = time.time()

def elapsed_h() -> float:
    return (time.time() - SESSION_T0) / 3600.0

def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88, flush=True)

# Auto-install dependencies on Colab
for pkg, mod in [
    ("huggingface_hub", "huggingface_hub"),
    ("torchattacks", "torchattacks"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pyarrow", "pyarrow"),
]:
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run(["pip", "install", "--quiet", pkg], check=True)

# HuggingFace Auth (Colab userdata with environment fallback)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

if not hf_token:
    print("WARNING: HF_TOKEN not found in environment or Colab secrets.")
    print("Set secret 'HF_TOKEN' with notebook access enabled to enable checkpoint backup and restore.")
else:
    print(f"HF_TOKEN authenticated (starts with: {hf_token[:4]}...)")

# %% [markdown]
# ## Step 2: Repository Clone & Workspace Paths (Colab)

# %%
REPO_NAME = "Adversarial-Cognitive-Model"
COMMIT_SHA = os.environ.get("COLAB_COMMIT", "").strip()
if not re.fullmatch(r"[0-9a-f]{40}", COMMIT_SHA):
    raise SystemExit(
        "ERROR: COLAB_COMMIT must be a full 40-character lowercase git SHA. "
        "Refusing to run from a branch or partial SHA."
    )

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, cwd=str(REPO_DIR)).strip()

def assert_pinned_clean() -> None:
    head = git("rev-parse", "HEAD")
    if head != COMMIT_SHA:
        raise SystemExit(f"ERROR: HEAD {head} != COLAB_COMMIT {COMMIT_SHA}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit("ERROR: working tree is not clean:\n" + dirty)
    print(f"PIN OK: {head}; tree clean")

if os.path.exists("/content"):
    WORK_DIR = Path(f"/content/{REPO_NAME}")
    if not WORK_DIR.exists():
        print(f"Cloning FerrariKazu/{REPO_NAME}...")
        subprocess.run(
            f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git {WORK_DIR}",
            shell=True,
            check=True,
        )
    os.chdir(str(WORK_DIR))
    subprocess.run("git fetch origin", shell=True, check=True)
    subprocess.run(f"git checkout --detach {COMMIT_SHA}", shell=True, check=True)
    REPO_DIR = WORK_DIR.resolve()
else:
    REPO_DIR = Path(".").resolve()

assert_pinned_clean()

sys.path.insert(0, str(REPO_DIR))
os.environ["PYTHONPATH"] = f"{REPO_DIR}:{os.environ.get('PYTHONPATH', '')}"

# Evict stale cached modules
for mod in list(sys.modules.keys()):
    if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
        del sys.modules[mod]

from cloud_setup.ckpt_select import (
    CandidateError,
    can_overwrite_remote,
    files_for_active_runs,
    load_metadata,
    parse_checkpoint_kind,
    select_best_candidate,
    sha256_file,
    validate_candidate,
    validate_reuse_candidate,
)

sha = git("rev-parse", "HEAD")
print(f"Pinned commit: {sha}")

# Define canonical directory layout
OUT_DIR    = REPO_DIR / "output"
CKPT_DIR   = REPO_DIR / "checkpoints"
DATA_ROOT  = Path(os.environ.get("J2_DATA_ROOT", "/content/imagenet100" if os.path.exists("/content") else str(REPO_DIR / "data" / "imagenet100"))).resolve()
VAL_ROOT   = str(DATA_ROOT / "val")
TRAIN_ROOT = str(DATA_ROOT / "train")
DINO_PATH  = str(CKPT_DIR / "dinov2_vits14_pretrain.pth")

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "eval").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "audit_rev1").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "report").mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 2b: Replication Run Allowlist (COLAB_RUNS)

# %%
PREREG_REL = "prereg/stage0_replication_v1_1_DRAFT.json"

def load_pinned_prereg() -> Dict[str, Any]:
    try:
        return json.loads(git("show", f"{COMMIT_SHA}:{PREREG_REL}"))
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot read {PREREG_REL} at {COMMIT_SHA}: {exc}")

REPLICATION_PREREG = load_pinned_prereg()
RUN_SPECS: Dict[str, Dict[str, Any]] = {}
_STAGE_A = REPLICATION_PREREG["design"]["stage_a"]
RUN_SPECS[_STAGE_A["run_name"]] = {**_STAGE_A, "stage": "A", "train": True}
for _run in REPLICATION_PREREG["design"]["stage_b"]["runs"]:
    RUN_SPECS[_run["run_name"]] = {**_run, "stage": "B", "train": True}
for _run_name in REPLICATION_PREREG.get("run_names", {}).get("reused_runs", []):
    RUN_SPECS[_run_name] = {
        "run_name": _run_name,
        "phase": "consolidated_112",
        "phase_override_name": _run_name,
        "training_seed": 42,
        "batch_size": 64,
        "n_epochs": 15,
        "stage": "REUSE",
        "train": False,
        "checkpoint": "checkpoints/foundation_recurrence_112_seed42_final.pth",
    }
STAGE_A_RUN = _STAGE_A["run_name"]
STAGE_B_RUNS = [r["run_name"] for r in REPLICATION_PREREG["design"]["stage_b"]["runs"]]
ALLOWED_RUNS = tuple(RUN_SPECS)

COLAB_RUNS_RAW = os.environ.get("COLAB_RUNS", "").strip()
if not COLAB_RUNS_RAW:
    raise SystemExit(
        "ERROR: COLAB_RUNS is unset or empty. Set a comma-separated allowlist "
        f"from {PREREG_REL}."
    )
_REQUESTED = [x.strip() for x in COLAB_RUNS_RAW.split(",")]
if any(not x for x in _REQUESTED):
    raise SystemExit(f"ERROR: COLAB_RUNS contains an empty entry: {COLAB_RUNS_RAW!r}")
if len(set(_REQUESTED)) != len(_REQUESTED):
    raise SystemExit(f"ERROR: COLAB_RUNS contains duplicates: {COLAB_RUNS_RAW!r}")
_UNKNOWN = [x for x in _REQUESTED if x not in RUN_SPECS]
if _UNKNOWN:
    raise SystemExit(
        f"ERROR: COLAB_RUNS contains names not in {PREREG_REL}: {_UNKNOWN}; "
        f"allowed={list(ALLOWED_RUNS)}"
    )
if any(x in STAGE_B_RUNS for x in _REQUESTED) and STAGE_A_RUN not in _REQUESTED:
    raise SystemExit(
        f"ERROR: Stage B runs require the Stage A gate run {STAGE_A_RUN} in COLAB_RUNS."
    )
ACTIVE_RUNS = tuple(_REQUESTED)
print(f"COLAB_RUNS OK: {list(ACTIVE_RUNS)}")

# %% [markdown]
# ## Step 3: Hardware Verification & Locked Pre-Registration

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.stats as stats
import torchattacks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: No GPU detected! Go to Runtime > Change runtime type > T4 GPU.")

# Import substrate architecture
from gen2_foundation.model import Gen2FoundationModel
from gen2_foundation.backbone import CompactViTGen2
from noesis_vision.models.foveation import foveal_sample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Locked protocol specification -- identical to Kaggle_consolidated.py
PROTOCOL = {
    "protocol_id": "consolidated-flagship-112/1",
    "model_phase": "recurrence_only",
    "fovea_size": 112,
    "num_glimpses": 4,
    "seeds": [17, 42, 99],
    "effective_batch_size": 64,
    "micro_batch_size": 16,
    "grad_accum_steps": 4,
    "n_epochs": 15,
    "trades_beta": 2.5,
    "eps_pixel_255": 5.417,
    "eps_norm": 0.0940,  # 5.417 / 255 / 0.226
    "trades_steps": 4,
    "alpha": 0.05875,    # 2.5 * 0.0940 / 4
    "eval_t": 4,
    "clean_set": {"n": 5000, "note": "full val split"},
    "robust_set": {"n": 5000, "note": "full val split, worst-of-3 attacks"},
    "attacks": {
        "pgd50": {"steps": 50, "restarts": 1, "loss": "ce"},
        "apgd_ce": {"steps": 100, "restarts": 1, "loss": "ce"},
        "apgd_dlr": {"steps": 100, "restarts": 1, "loss": "dlr"},
    },
    "worst_case": "bitwise AND of pgd50, apgd_ce, apgd_dlr",
    "checkpoint_gate_pp": 1.0,
    "early_health_epoch": 3,
    "early_health_floor_rob": 0.15,
    "full_val_path_sha256": "6a78a2fde02bfc1ff93b063eeddb32fb32759d9da4994421adda7353636a3be7",
}

RULES = {
    "decision_rule_superiority": (
        "Flagship is declared superior iff min_seed(rec_112) > max_seed(ff_112) on 5k worst-case "
        "robust accuracy AND pooled McNemar exact two-sided p < 0.01 with net delta > 0."
    ),
    "decision_rule_null": (
        "If min_seed(rec_112) <= max_seed(ff_112) or McNemar p >= 0.05, conclusion is that "
        "recurrent fusion does not earn its 4x compute cost at 112px."
    ),
    "abort_epsilon": (
        "Training immediately aborts if eps_norm != 0.0940 +/- 0.001 or if loss_kl <= 0.0."
    ),
    "abort_epoch3_floor": (
        "Training immediately aborts if PGD-10 monitor robust accuracy < 15.0% by epoch 3."
    ),
    "selection_ban": (
        "Strictly evaluate epoch 15 final checkpoint. Best-validation checkpoint selection is forbidden."
    ),
}

EPS_PIXEL = PROTOCOL["eps_pixel_255"] / 255.0
EPS_NORM  = PROTOCOL["eps_norm"]

def prereg_hash() -> str:
    payload = json.dumps({"protocol": PROTOCOL, "rules": RULES}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

PROTOCOL_HASH = prereg_hash()

def print_preregistration() -> None:
    banner("A. PRE-REGISTRATION (LOCKED: consolidated-flagship-112/1)")
    print(f"  protocol id   : {PROTOCOL['protocol_id']}")
    print(f"  rules hash    : {PROTOCOL_HASH}")
    print(f"  resolution    : {PROTOCOL['fovea_size']} px (4 shifted views, fovea=full frame)")
    print(f"  recipe        : TRADES beta={PROTOCOL['trades_beta']}, eps={PROTOCOL['eps_pixel_255']}/255 ({PROTOCOL['eps_norm']:.4f} norm)")
    print(f"  inner PGD     : {PROTOCOL['trades_steps']} steps, alpha={PROTOCOL['alpha']:.5f}")
    print(f"  epochs        : {PROTOCOL['n_epochs']} (Rung 3 from epoch 1, no warmup)")
    print(f"  seeds         : {PROTOCOL['seeds']}")
    print(f"  effective bs  : {PROTOCOL['effective_batch_size']} (loader batch; no gradient accumulation)")
    print(f"  clean set     : {PROTOCOL['clean_set']['n']} images (full val)")
    print(f"  robust set    : {PROTOCOL['robust_set']['n']} images (full val, worst-of-3 attacks)")
    print(f"  decision rule : min(rec_112) > max(ff_112) AND pooled McNemar p < 0.01")
    print(f"  abort triggers: eps != 0.094, KL <= 0, or epoch 3 PGD-10 < 15.0%")
    print("=" * 88, flush=True)

# %% [markdown]
# ## Step 4: HuggingFace Repos, Live Sync & Checkpoint Restore

# %%
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

HF_BEST    = "FerrariKazu/rhan-nxa-g2-checkpoints"
HF_ROLLING = "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"

if hf_token:
    api = HfApi(token=hf_token)
    for r in [HF_BEST, HF_ROLLING]:
        try:
            api.create_repo(repo_id=r, repo_type="dataset", private=True, exist_ok=True)
            print(f"HF repo ready: {r}")
        except Exception as e:
            print(f"  HF check {r}: {e}")

FINAL_HASH_LEDGER = OUT_DIR / "final_hashes.json"
_REMOTE_EPOCH_CACHE: Dict[Tuple[str, str], Optional[int]] = {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _load_hash_ledger() -> Dict[str, str]:
    if not FINAL_HASH_LEDGER.exists():
        return {}
    try:
        data = json.loads(FINAL_HASH_LEDGER.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _record_final_hash(run_name: str, path: Path) -> str:
    ledger = _load_hash_ledger()
    digest = sha256_file(path)
    ledger[run_name] = digest
    _write_json_atomic(FINAL_HASH_LEDGER, ledger)
    return digest


def _verify_final_hash(run_name: str, path: Path) -> bool:
    ledger = _load_hash_ledger()
    digest = sha256_file(path)
    if run_name in ledger:
        if ledger[run_name] != digest:
            raise SystemExit(
                f"ERROR: final hash mismatch for {run_name}: "
                f"ledger={ledger[run_name]} local={digest}"
            )
        return True
    _record_final_hash(run_name, path)
    return True


def _final_path(run_name: str) -> Path:
    return CKPT_DIR / f"foundation_{run_name}_final.pth"


def _eval_json_path(run_name: str) -> Path:
    return OUT_DIR / "eval" / f"{run_name}_112.json"


def _eval_npz_path(run_name: str) -> Path:
    return OUT_DIR / "eval" / f"{run_name}_112.npz"


def _run_for_filename(filename: str) -> Optional[str]:
    for run_name in ACTIVE_RUNS:
        if parse_checkpoint_kind(filename, run_name):
            return run_name
    return None


def _download_to_part(repo_id: str, filename: str) -> Path:
    cache_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", token=hf_token
    )
    part = CKPT_DIR / (filename + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache_path, part)
    return part


def _remote_epoch(repo_id: str, filename: str) -> Optional[int]:
    key = (repo_id, filename)
    if key in _REMOTE_EPOCH_CACHE:
        return _REMOTE_EPOCH_CACHE[key]
    epoch: Optional[int] = None
    try:
        cache_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", token=hf_token
        )
        state = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("epoch"), int):
            epoch = int(state["epoch"])
    except Exception:
        epoch = None
    _REMOTE_EPOCH_CACHE[key] = epoch
    return epoch


def restore_checkpoints_from_hf() -> None:
    """Restore only fully valid candidates for runs in COLAB_RUNS.

    Candidates are downloaded to .part, inspected internally, validated, and
    only then atomically renamed. Selection is by highest valid epoch across
    both repositories, never by repository or filename order.
    """
    if not hf_token:
        return
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for repo_id in [HF_BEST, HF_ROLLING]:
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Note: {repo_id}: {e}")
            continue
        for filename in files:
            run_name = _run_for_filename(filename)
            if run_name is None:
                continue
            kind = parse_checkpoint_kind(filename, run_name)
            if kind not in ("rolling", "final"):
                continue
            spec = RUN_SPECS[run_name]
            part = None
            try:
                part = _download_to_part(repo_id, filename)
                state = load_metadata(part)
                validate_candidate(
                    state,
                    expected_run=run_name,
                    expected_phase=spec["phase"],
                    expected_commit=COMMIT_SHA,
                    expected_seed=spec["training_seed"],
                    require_run_name=True,
                    require_provenance=True,
                )
                digest = sha256_file(part)
                grouped.setdefault((run_name, kind), []).append({
                    "metadata": state,
                    "source": repo_id,
                    "path": str(part),
                    "sha256": digest,
                    "filename": filename,
                })
            except CandidateError as exc:
                print(f"  Rejected {repo_id}:{filename}: {exc}")
                if part is not None:
                    part.unlink(missing_ok=True)

    restored = 0
    for (run_name, kind), candidates in sorted(grouped.items()):
        try:
            best = select_best_candidate(candidates)
        except CandidateError as exc:
            print(f"  Cannot select {run_name} {kind}: {exc}")
            for c in candidates:
                Path(c["path"]).unlink(missing_ok=True)
            continue
        target = CKPT_DIR / f"foundation_{run_name}_{kind}.pth"
        local_epoch = 0
        if target.exists():
            try:
                local_epoch = int(load_metadata(target).get("epoch", 0))
            except Exception:
                local_epoch = 0
        remote_epoch = int(best["metadata"]["epoch"])
        if remote_epoch <= local_epoch:
            print(
                f"  Keeping local {target.name} (epoch {local_epoch}) over "
                f"remote epoch {remote_epoch}"
            )
            Path(best["path"]).unlink(missing_ok=True)
        else:
            os.replace(best["path"], target)
            if kind == "final":
                _record_final_hash(run_name, target)
            restored += 1
            print(f"  Restored {best['source']}:{best['filename']} -> {target.name} (epoch {remote_epoch})")
        for c in candidates:
            Path(c["path"]).unlink(missing_ok=True)
    print(f"Restored {restored} valid checkpoint(s) for active runs.")


def sync_to_hf() -> None:
    """Upload only active-run checkpoints and never downgrade a remote epoch."""
    if not hf_token:
        return
    api = HfApi(token=hf_token)
    filenames = [p.name for p in CKPT_DIR.iterdir() if p.is_file()]
    active_files = files_for_active_runs(filenames, ACTIVE_RUNS)
    for filename in active_files:
        run_name = _run_for_filename(filename)
        if run_name is None:
            continue
        kind = parse_checkpoint_kind(filename, run_name)
        spec = RUN_SPECS[run_name]
        path = CKPT_DIR / filename
        try:
            state = load_metadata(path)
            validate_candidate(
                state,
                expected_run=run_name,
                expected_phase=spec["phase"],
                expected_commit=COMMIT_SHA,
                expected_seed=spec["training_seed"],
                require_run_name=True,
                require_provenance=True,
            )
        except CandidateError as exc:
            print(f"  Sync skipped {filename}: {exc}")
            continue
        repo_id = HF_ROLLING if kind == "rolling" else HF_BEST
        local_epoch = int(state["epoch"])
        remote_epoch = _remote_epoch(repo_id, filename)
        if not can_overwrite_remote(local_epoch, remote_epoch):
            print(
                f"  Sync refused downgrade {filename}: local epoch {local_epoch} "
                f"< remote epoch {remote_epoch}"
            )
            continue
        if remote_epoch == local_epoch:
            print(f"  Sync skipped {filename}: remote already at epoch {remote_epoch}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            _REMOTE_EPOCH_CACHE[(repo_id, filename)] = local_epoch
        except Exception as e:
            print(f"  Warning: sync {filename}: {e}")


# %% [markdown]
# ## Step 5: DINOv2 Weights & ImageNet-100 Data Bootstrap

# %%
def ensure_dino_weights(path: str = DINO_PATH) -> str:
    p = Path(path)
    if p.exists() and p.stat().st_size > 80 * 1024 * 1024:
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official DINOv2-small weights to {p} ...")
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    subprocess.run(["curl", "-sSL", "-o", str(p), url], check=True)
    assert p.exists() and p.stat().st_size > 80 * 1024 * 1024, "DINO download failed"
    return str(p)

def bootstrap_imagenet100(data_dir: Path) -> None:
    data_dir = Path(data_dir).resolve()
    td, vd = data_dir / "train", data_dir / "val"
    if td.is_dir() and vd.is_dir():
        n_val = sum(1 for _ in vd.glob("*/*.JPEG")) + sum(1 for _ in vd.glob("*/*.jpg"))
        if n_val == 5000:
            print(f"ImageNet-100 already bootstrapped at {data_dir} (val=5000).")
            return

    data_dir.mkdir(parents=True, exist_ok=True)
    prep_script = REPO_DIR / "scripts" / "prepare_imagenet100.py"
    if prep_script.exists():
        print(f"Running official {prep_script} --root {data_dir} ...")
        res = subprocess.run([sys.executable, str(prep_script), "--root", str(data_dir)])
        if res.returncode == 0:
            print("ImageNet-100 bootstrap via prepare_imagenet100.py complete.")
            return
        print(f"prepare_imagenet100.py exited with {res.returncode}; falling back to direct parquet extraction...")

    banner("BOOTSTRAPPING IMAGENET-100 FROM HUGGINGFACE")
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "clane9/imagenet-100"
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    train_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    val_files = sorted([f for f in files if f.startswith("data/val-") and f.endswith(".parquet")])

    for split_name, shard_list in [("val", val_files), ("train", train_files)]:
        split_dir = data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Unpacking {split_name} ({len(shard_list)} shards) ...")
        for shard in shard_list:
            local_parquet = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset")
            table = pq.read_table(local_parquet)
            cols = table.to_pydict()
            imgs = cols["image"]
            lbls = cols["label"]
            for img_data, lbl in zip(imgs, lbls):
                class_name = str(lbl)
                target_folder = split_dir / class_name
                target_folder.mkdir(parents=True, exist_ok=True)
                img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
                img_path = target_folder / f"{hashlib.md5(img_bytes).hexdigest()[:12]}.JPEG"
                if not img_path.exists():
                    with open(img_path, "wb") as f_out:
                        f_out.write(img_bytes)

    print("ImageNet-100 bootstrap complete.")

# %% [markdown]
# ## Step 6: Attacks & Losses (Standalone Harness & TRADES)

# %%
def run_pgd_kl_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float = EPS_NORM,
    steps: int = 4,
    alpha: float = 0.05875,
) -> torch.Tensor:
    """Canonical 4-step TRADES inner PGD attack."""
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            p_clean = F.softmax(model(x), dim=-1)

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            out_adv = model(x_adv)
            loss_kl = F.kl_div(F.log_softmax(out_adv, dim=-1), p_clean, reduction="batchmean")
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    if was_training:
        model.train()
    return x_adv.detach()

def run_pgd_ce_monitor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = EPS_NORM,
    steps: int = 50,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Canonical PGD-CE attack for evaluation."""
    if alpha is None:
        alpha = 2.5 * eps / steps
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            out = model(x_adv)
            loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    if was_training:
        model.train()
    return x_adv.detach()

def trades_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    beta: float = 2.5,
    eps: float = EPS_NORM,
    steps: int = 4,
    alpha: float = 0.05875,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TRADES loss: L_natural + beta * L_robust_kl."""
    out_clean = model(x_clean)
    loss_nat = F.cross_entropy(out_clean, y)

    x_adv = run_pgd_kl_attack(model, x_clean, eps=eps, steps=steps, alpha=alpha)
    model.train()
    out_adv = model(x_adv)
    loss_kl = F.kl_div(
        F.log_softmax(out_adv, dim=-1),
        F.softmax(out_clean.detach(), dim=-1),
        reduction="batchmean",
    )
    loss_total = loss_nat + beta * loss_kl
    return loss_total, loss_nat.detach(), loss_kl.detach()

# %% [markdown]
# ## Step 7: Data Loaders & Dataset Hash

# %%
def get_loaders(
    data_root: Path,
    img_size: int = 112,
    batch_size: int = 48,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=tf_val)
    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=tf_train)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    g_mon = torch.Generator().manual_seed(12345)
    mon_indices = torch.randperm(len(val_ds), generator=g_mon)[:500].tolist()
    monitor_loader = DataLoader(
        Subset(val_ds, mon_indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, monitor_loader

# %% [markdown]
# ## Step 8: Training Function for the Consolidated Flagship

# %%
from training.train_gen2_foundation import run_gen2_phase

def train_consolidated_seed(seed: int) -> Path:
    banner(f"TRAINING CONSOLIDATED FLAGSHIP -- SEED {seed}")
    final_ckpt = CKPT_DIR / f"foundation_recurrence_112_seed{seed}_final.pth"
    rolling_ckpt = CKPT_DIR / f"foundation_recurrence_112_seed{seed}_rolling.pth"

    if final_ckpt.exists():
        print(f"Final checkpoint already exists at {final_ckpt}. Skipping training.")
        return final_ckpt

    if rolling_ckpt.exists():
        try:
            st = torch.load(rolling_ckpt, map_location="cpu", weights_only=False)
            ep = st.get("epoch", 0) + 1
            print(f"\n>>> RESUME: rolling checkpoint found at {rolling_ckpt}")
            print(f"    run_gen2_phase will resume from epoch {ep} onwards.")
        except Exception as e:
            print(f"\n>>> RESUME: rolling checkpoint found at {rolling_ckpt} (inspect notice: {e})")
    else:
        print(f"\n>>> FRESH START: no rolling checkpoint for seed {seed}. Training from epoch 1.")

    ensure_dino_weights(DINO_PATH)

    # Invokes the exact canonical ff_112 training engine unchanged,
    # swapping the model class to Gen2FoundationModel(phase="recurrence_only", fovea_size=112)
    # with DINOv2 warm-start and the active abort assertions.
    eff_batch = int(os.environ.get("CONSOLIDATED_BATCH", PROTOCOL.get("effective_batch_size", 48)))
    run_gen2_phase(
        phase="consolidated_112",
        arm="v1",
        data_root=str(DATA_ROOT),
        ckpt_dir=str(CKPT_DIR),
        report_dir=str(OUT_DIR),
        dino_path=DINO_PATH,
        batch_size=eff_batch,
        num_workers=2,
        smoke=False,
        n_epochs=PROTOCOL["n_epochs"],
        training_seed=seed,
        phase_override_name=f"recurrence_112_seed{seed}",
        epoch_callback=lambda p, ep: sync_to_hf(),
    )

    assert final_ckpt.exists(), f"Expected final checkpoint at {final_ckpt}"
    print(f"Verified final checkpoint for seed {seed} at {final_ckpt}")
    sync_to_hf()
    return final_ckpt

def _restore_remote_final_if_verified(run_name: str) -> bool:
    ledger = _load_hash_ledger()
    if run_name not in ledger:
        return False
    expected = ledger[run_name]
    spec = RUN_SPECS[run_name]
    for repo_id in [HF_BEST, HF_ROLLING]:
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
        except Exception:
            continue
        for filename in files:
            if parse_checkpoint_kind(filename, run_name) != "final":
                continue
            part = None
            try:
                part = _download_to_part(repo_id, filename)
                state = load_metadata(part)
                validate_candidate(
                    state,
                    expected_run=run_name,
                    expected_phase=spec["phase"],
                    expected_commit=COMMIT_SHA,
                    expected_seed=spec["training_seed"],
                    require_run_name=True,
                    require_provenance=True,
                )
                if sha256_file(part) != expected:
                    print(f"  Remote final {repo_id}:{filename} does not match local record")
                    part.unlink(missing_ok=True)
                    continue
                os.replace(part, _final_path(run_name))
                print(f"  Verified remote final {repo_id}:{filename}; skipping training")
                return True
            except CandidateError as exc:
                print(f"  Remote final rejected {repo_id}:{filename}: {exc}")
                if part is not None:
                    part.unlink(missing_ok=True)
    return False


def train_run(run_name: str) -> Path:
    spec = RUN_SPECS[run_name]
    final = _final_path(run_name)
    banner(f"RUN {run_name} ({spec['stage']})")
    if not spec.get("train", False):
        if not final.exists():
            raise SystemExit(f"ERROR: reuse checkpoint missing for {run_name}: {final}")
        validate_reuse_candidate(
            load_metadata(final),
            expected_phase=spec["phase"],
            expected_epoch=int(spec["n_epochs"]),
            expected_seed=int(spec["training_seed"]),
        )
        _verify_final_hash(run_name, final)
        print(f"Reusing verified final checkpoint: {final}")
        return final
    if final.exists():
        validate_candidate(
            load_metadata(final),
            expected_run=run_name,
            expected_phase=spec["phase"],
            expected_commit=COMMIT_SHA,
            expected_seed=int(spec["training_seed"]),
            require_run_name=True,
            require_provenance=True,
        )
        _verify_final_hash(run_name, final)
        print(f"Final checkpoint already exists and is verified: {final}")
        return final
    if _restore_remote_final_if_verified(run_name):
        return final
    ensure_dino_weights(DINO_PATH)
    run_gen2_phase(
        phase=spec["phase"],
        arm="v1",
        data_root=str(DATA_ROOT),
        ckpt_dir=str(CKPT_DIR),
        report_dir=str(OUT_DIR),
        dino_path=DINO_PATH,
        batch_size=int(spec["batch_size"]),
        num_workers=2,
        smoke=False,
        n_epochs=int(spec["n_epochs"]),
        training_seed=int(spec["training_seed"]),
        phase_override_name=run_name,
        epoch_callback=lambda p, ep: sync_to_hf(),
    )
    if not final.exists():
        raise SystemExit(f"ERROR: expected final checkpoint at {final}")
    _record_final_hash(run_name, final)
    print(f"Verified final checkpoint for {run_name} at {final}")
    sync_to_hf()
    return final


# %% [markdown]
# ## Step 9: Full 5,000-Image Evaluation Battery

# %%
class _NormWrap(nn.Module):
    """Re-normalise inside forward so torchattacks can work in [0,1] pixel space."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_m", m)
        self.register_buffer("_s", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x in [0,1]
        return self.base((x - self._m.to(x)) / self._s.to(x))


def full_5k_battery(model: nn.Module, label: str) -> Dict[str, Any]:
    """Clean + PGD-50 + APGD-CE + APGD-DLR on all 5,000 validation images."""
    banner(f"FULL 5k EVALUATION -- {label}")
    # Two transforms: normalised (for PGD/clean) and raw [0,1] (for torchattacks)
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val_01 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),  # [0,1] -- no normalise; NormWrap handles it
    ])
    val_ds = datasets.ImageFolder(VAL_ROOT, transform=tf_val)
    val_ds_01 = datasets.ImageFolder(VAL_ROOT, transform=tf_val_01)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    loader_01 = DataLoader(val_ds_01, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # NormWrap so torchattacks applies eps in [0,1] pixel domain
    model_01 = _NormWrap(model).to(DEVICE)
    model_01.eval()
    apgd_ce = torchattacks.APGD(model_01, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(model_01, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="dlr", verbose=False)

    preds_dict: Dict[str, List[np.ndarray]] = {"clean": [], "pgd50": [], "apgd_ce": [], "apgd_dlr": []}
    corr_dict: Dict[str, List[np.ndarray]] = {"clean": [], "pgd50": [], "apgd_ce": [], "apgd_dlr": []}

    model.eval()
    loader_01_iter = iter(loader_01)
    for b_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_01, _ = next(loader_01_iter)  # unnormalised [0,1] for torchattacks
        x_01 = x_01.to(DEVICE)
        y_np = y.cpu().numpy()

        # 1. Clean (use normalised x)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_clean = model(x).argmax(dim=-1).cpu().numpy()
        preds_dict["clean"].append(p_clean)
        corr_dict["clean"].append((p_clean == y_np).astype(np.int8))

        # 2. PGD-50 (normalised domain, custom loop handles clamp correctly)
        x_pgd50 = run_pgd_ce_monitor(model, x, y, eps=EPS_NORM, steps=50)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_pgd50 = model(x_pgd50).argmax(dim=-1).cpu().numpy()
        preds_dict["pgd50"].append(p_pgd50)
        corr_dict["pgd50"].append((p_pgd50 == y_np).astype(np.int8))

        # 3. APGD-CE (torchattacks, [0,1] domain via NormWrap, eps=EPS_PIXEL)
        x_apgd_ce = apgd_ce(x_01, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_apgd_ce = model_01(x_apgd_ce).argmax(dim=-1).cpu().numpy()
        preds_dict["apgd_ce"].append(p_apgd_ce)
        corr_dict["apgd_ce"].append((p_apgd_ce == y_np).astype(np.int8))

        # 4. APGD-DLR (torchattacks, [0,1] domain via NormWrap, eps=EPS_PIXEL)
        x_apgd_dlr = apgd_dlr(x_01, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_apgd_dlr = model_01(x_apgd_dlr).argmax(dim=-1).cpu().numpy()
        preds_dict["apgd_dlr"].append(p_apgd_dlr)
        corr_dict["apgd_dlr"].append((p_apgd_dlr == y_np).astype(np.int8))

        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
            print(f"  [5k Eval {b_idx+1}/{len(loader)}] Batch processed.", flush=True)

    res = {}
    for k in ("clean", "pgd50", "apgd_ce", "apgd_dlr"):
        res[f"corr_{k}"] = np.concatenate(corr_dict[k])
        res[f"pred_{k}"] = np.concatenate(preds_dict[k])
        res[f"acc_{k}"] = float(res[f"corr_{k}"].mean())

    res["corr_worst_case"] = (res["corr_pgd50"] & res["corr_apgd_ce"] & res["corr_apgd_dlr"]).astype(np.int8)
    res["acc_worst_case"] = float(res["corr_worst_case"].mean())

    print(f"\n{label} Results (N=5,000):")
    print(f"  Clean:      {res['acc_clean']*100:.2f}%")
    print(f"  PGD-50:     {res['acc_pgd50']*100:.2f}%")
    print(f"  APGD-CE:    {res['acc_apgd_ce']*100:.2f}%")
    print(f"  APGD-DLR:   {res['acc_apgd_dlr']*100:.2f}%")
    print(f"  Worst-Case: {res['acc_worst_case']*100:.2f}%\n")
    return res

# %% [markdown]
# ## Step 10: Statistical Synthesis & Decision Rule

# %%
def mcnemar_test(corr_a: np.ndarray, corr_b: np.ndarray) -> Dict[str, Any]:
    """Exact two-sided McNemar binomial test with 95% confidence interval."""
    assert len(corr_a) == len(corr_b)
    b = int(((corr_a == 1) & (corr_b == 0)).sum())
    c = int(((corr_a == 0) & (corr_b == 1)).sum())
    n_disc = b + c
    delta = (corr_a.mean() - corr_b.mean()) * 100.0

    if n_disc == 0:
        return {"b": b, "c": c, "p_exact": 1.0, "delta_pp": delta, "ci_95": (delta, delta)}

    p_exact = float(stats.binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue)
    chi2 = float((abs(b - c) - 1.0) ** 2 / n_disc) if n_disc > 0 else 0.0

    # Agresti-Min 95% CI on discordant proportion
    se = math.sqrt((b + c) - ((b - c) ** 2 / len(corr_a))) / len(corr_a) * 100.0
    ci_low = delta - 1.96 * se
    ci_high = delta + 1.96 * se

    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "delta_pp": delta,
        "p_exact": p_exact,
        "chi2": chi2,
        "ci_95": (ci_low, ci_high),
    }

# %% [markdown]
# ## Step 11: Main Orchestration & HuggingFace Dual-Upload

# %%
def eval_run_resumable(run_name: str) -> Dict[str, Any]:
    json_path = _eval_json_path(run_name)
    if json_path.exists():
        try:
            previous = json.loads(json_path.read_text())
            if previous.get("status") == "complete" and previous.get("run_name") == run_name:
                print(f"Evaluation already complete: {json_path}")
                return previous
        except Exception:
            pass
    final = _final_path(run_name)
    if not final.exists():
        raise SystemExit(f"ERROR: final checkpoint missing for evaluation: {final}")
    spec = RUN_SPECS[run_name]
    arch_phase = "recurrence_only" if spec["phase"] == "consolidated_112" else spec["phase"]
    model = Gen2FoundationModel(
        phase=arch_phase, arm="v1", fovea_size=112, num_glimpses=4
    ).to(DEVICE)
    state = torch.load(final, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state.get("model", state))
    res = full_5k_battery(model, run_name)
    payload = {
        "status": "complete",
        "run_name": run_name,
        "phase": spec["phase"],
        "seed": int(spec["training_seed"]),
        "final_sha256": sha256_file(final),
        "acc_clean": float(res["acc_clean"]),
        "acc_pgd50": float(res["acc_pgd50"]),
        "acc_apgd_ce": float(res["acc_apgd_ce"]),
        "acc_apgd_dlr": float(res["acc_apgd_dlr"]),
        "acc_worst_case": float(res["acc_worst_case"]),
    }
    _write_json_atomic(json_path, payload)
    npz_path = _eval_npz_path(run_name)
    npz_tmp = npz_path.with_suffix(".npz.tmp")
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(
            f,
            corr_clean=res["corr_clean"],
            corr_pgd50=res["corr_pgd50"],
            corr_apgd_ce=res["corr_apgd_ce"],
            corr_apgd_dlr=res["corr_apgd_dlr"],
            corr_worst_case=res["corr_worst_case"],
        )
    os.replace(npz_tmp, npz_path)
    return payload


def _print_replication_decision() -> None:
    groups = {
        "flagship": [STAGE_A_RUN, "recurrence_112_seed42", "recurrence_112_seed99_b64"],
        "ff_b64": ["ff_112_seed17_b64", "ff_112_seed42_b64", "ff_112_seed99_b64"],
    }
    accs: Dict[str, List[float]] = {}
    for group, run_names in groups.items():
        vals: List[float] = []
        for run_name in run_names:
            path = _eval_json_path(run_name)
            if not path.exists():
                print(f"DECISION INCOMPLETE: missing evaluation for {run_name}")
                return
            data = json.loads(path.read_text())
            vals.append(float(data["acc_worst_case"]) * 100.0)
        accs[group] = vals
    rec = accs["flagship"]
    ff = accs["ff_b64"]
    mean_gap = sum(rec) / 3.0 - sum(ff) / 3.0
    positive = min(rec) > max(ff) and mean_gap >= 1.0
    reversed_ = max(rec) < min(ff)
    verdict = "POSITIVE" if positive else ("REVERSED" if reversed_ else "NULL")
    payload = {
        "flagship_worst_case_pp": rec,
        "ff_b64_worst_case_pp": ff,
        "min_flagship_pp": min(rec),
        "max_ff_b64_pp": max(ff),
        "mean_gap_pp": mean_gap,
        "verdict": verdict,
    }
    _write_json_atomic(OUT_DIR / "replication_decision.json", payload)
    print(
        f"REPLICATION DECISION: {verdict} | flagship={rec} | ff_b64={ff} "
        f"| mean_gap={mean_gap:+.2f} pp"
    )


def main_replication() -> None:
    banner("MATCHED REPLICATION -- STAGE A GATE + STAGE B + EVAL")
    print(f"Active runs: {list(ACTIVE_RUNS)}")
    print(f"Pinned prereg: {PREREG_REL}")
    ensure_dino_weights(DINO_PATH)
    bootstrap_imagenet100(DATA_ROOT)
    restore_checkpoints_from_hf()

    if STAGE_A_RUN in ACTIVE_RUNS:
        train_run(STAGE_A_RUN)
        result = eval_run_resumable(STAGE_A_RUN)
        worst = float(result["acc_worst_case"]) * 100.0
        print(f"STAGE A worst-case: {worst:.2f}%")
        if worst <= 33.0:
            print("STAGE A STOP: <= 33.0%; Stage B is not launched; the NULL verdict stands.")
            return
        if worst >= 35.44:
            print("STAGE A CONTINUE: >= 35.44%.")
        else:
            print("STAGE A BORDERLINE: 33.0% < worst-case < 35.44%; continue with Stage B.")

    for run_name in ACTIVE_RUNS:
        if run_name == STAGE_A_RUN:
            continue
        train_run(run_name)
        eval_run_resumable(run_name)

    _print_replication_decision()


def main() -> None:
    print_preregistration()

    # 1. Ensure Data & Weights
    ensure_dino_weights(DINO_PATH)
    bootstrap_imagenet100(DATA_ROOT)

    # 2. Train Consolidated Flagship (Seeds 17, 42, 99)
    flagship_ckpts = {}
    for s in PROTOCOL["seeds"]:
        flagship_ckpts[s] = train_consolidated_seed(s)

    # 3. Locate / Download the 3 ff_112 Baselines
    ff112_ckpts = {
        17: CKPT_DIR / "foundation_ff_112_final.pth",
        42: CKPT_DIR / "foundation_ff_112_seed42_final.pth",
        99: CKPT_DIR / "foundation_ff_112_seed99_final.pth",
    }

    # If any ff_112 ckpts missing locally, try fetching from HF
    for s, p in ff112_ckpts.items():
        if not p.exists():
            from huggingface_hub import hf_hub_download
            try:
                print(f"Downloading {p.name} from HuggingFace...")
                dl = hf_hub_download(
                    repo_id="FerrariKazu/rhan-nxa-g2-checkpoints",
                    filename=p.name,
                    local_dir=str(CKPT_DIR),
                )
            except Exception as e:
                print(f"Could not download {p.name}: {e}")

    # 4. Evaluate all 6 checkpoints on full 5,000 images
    eval_results = {}
    per_sample_bundle = {}

    # Evaluate Flagship seeds
    for s in PROTOCOL["seeds"]:
        lbl = f"recurrence_112_seed{s}"
        m = Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112, num_glimpses=4).to(DEVICE)
        st = torch.load(flagship_ckpts[s], map_location=DEVICE, weights_only=False)
        m.load_state_dict(st.get("model", st))
        res = full_5k_battery(m, lbl)
        eval_results[lbl] = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        per_sample_bundle[f"{lbl}_clean"] = res["corr_clean"]
        per_sample_bundle[f"{lbl}_worst"] = res["corr_worst_case"]

    # Evaluate ff_112 Baseline seeds
    for s in PROTOCOL["seeds"]:
        lbl = f"ff_112_seed{s}"
        if ff112_ckpts[s].exists():
            m_ff = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112, num_glimpses=4).to(DEVICE)
            st_ff = torch.load(ff112_ckpts[s], map_location=DEVICE, weights_only=False)
            m_ff.load_state_dict(st_ff.get("model", st_ff))
            res_ff = full_5k_battery(m_ff, lbl)
            eval_results[lbl] = {k: v for k, v in res_ff.items() if not isinstance(v, np.ndarray)}
            per_sample_bundle[f"{lbl}_clean"] = res_ff["corr_clean"]
            per_sample_bundle[f"{lbl}_worst"] = res_ff["corr_worst_case"]
        else:
            print(f"WARNING: Baseline {lbl} not found at {ff112_ckpts[s]}! Skipping.")

    # 5. Pre-registered Hypothesis Testing
    banner("PRE-REGISTERED HYPOTHESIS TESTING & DECISION RULE")
    rec_worst_accs = [eval_results[f"recurrence_112_seed{s}"]["acc_worst_case"] for s in PROTOCOL["seeds"]]
    min_rec = min(rec_worst_accs)

    ff_present = [s for s in PROTOCOL["seeds"] if f"ff_112_seed{s}" in eval_results]
    if len(ff_present) == len(PROTOCOL["seeds"]):
        ff_worst_accs = [eval_results[f"ff_112_seed{s}"]["acc_worst_case"] for s in PROTOCOL["seeds"]]
        max_ff = max(ff_worst_accs)

        # Pooled McNemar test across all 3 seed pairs
        pooled_rec = np.concatenate([per_sample_bundle[f"recurrence_112_seed{s}_worst"] for s in PROTOCOL["seeds"]])
        pooled_ff = np.concatenate([per_sample_bundle[f"ff_112_seed{s}_worst"] for s in PROTOCOL["seeds"]])
        pooled_test = mcnemar_test(pooled_rec, pooled_ff)

        cond1_pass = min_rec > max_ff
        cond2_pass = (pooled_test["p_exact"] < 0.01) and (pooled_test["ci_95"][0] > 0.0)

        print(f"Condition 1: min(rec_112) > max(ff_112)")
        print(f"  min(rec_112) = {min_rec*100:.2f}% | max(ff_112) = {max_ff*100:.2f}% -> {'PASS' if cond1_pass else 'FAIL'}")
        print(f"Condition 2: Pooled McNemar CI excludes 0 (p < 0.01)")
        print(f"  Delta: +{pooled_test['delta_pp']:.2f} pp | 95% CI: [{pooled_test['ci_95'][0]:.2f}, {pooled_test['ci_95'][1]:.2f}] | p = {pooled_test['p_exact']:.3e} -> {'PASS' if cond2_pass else 'FAIL'}")

        if cond1_pass and cond2_pass:
            verdict = "FLAGSHIP CONFIRMED: Recurrent loop achieves superior robustness at 112px."
        else:
            verdict = "NULL / COST INEFFECTIVE: Recurrent loop does not earn its 4x compute cost at 112px."

        print(f"\nFINAL VERDICT:\n  {verdict}\n")
    else:
        verdict = "INCOMPLETE: One or more ff_112 baselines were missing from evaluation."
        pooled_test = {}

    # 6. Save Bundle and Report
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_hash": PROTOCOL_HASH,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eval_results": eval_results,
        "pooled_test": pooled_test,
        "verdict": verdict,
    }

    report_path = OUT_DIR / "consolidated_flagship_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    bundle_path = OUT_DIR / "per_sample_bundle_112.npz"
    np.savez_compressed(bundle_path, **per_sample_bundle)
    print(f"Saved report to {report_path}")
    print(f"Saved per-sample bundle to {bundle_path}")

    # 7. Push to HuggingFace (dual upload: archive and canonical top-level)
    hf_token_val = os.environ.get("HF_TOKEN")
    if hf_token_val:
        banner("UPLOADING TO HUGGINGFACE")
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token_val)
        repo_id = "FerrariKazu/rhan-nxa-g2-checkpoints"
        _ts_up = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"closure/consolidated_112/{_ts_up}"

        api.upload_file(path_or_fileobj=str(report_path), path_in_repo=f"{prefix}/report.json", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(bundle_path), path_in_repo=f"{prefix}/per_sample_bundle.npz", repo_id=repo_id)
        for s in PROTOCOL["seeds"]:
            ckpt_path = str(flagship_ckpts[s])
            # Archive copy (timestamped, for provenance)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"{prefix}/foundation_recurrence_112_seed{s}_final.pth",
                repo_id=repo_id,
            )
            # Canonical top-level copy (required for OPAL gate in colab_j2_eval_train.py)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"foundation_recurrence_112_seed{s}_final.pth",
                repo_id=repo_id,
            )
            print(f"  Uploaded flagship seed {s} (archive + top-level canonical)")
        print("Upload complete!")

    banner("RUN COMPLETE")


if __name__ == "__main__":
    main_replication()
