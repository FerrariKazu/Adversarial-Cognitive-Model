"""
HF persistence module for J1 foundation training.

Provides per-epoch immutable HF archive persistence, best-checkpoint
persistence, rolling recovery mirror, atomic local checkpoint writes,
SHA-256 verification, remote artifact verification, persistent manifest,
append-only event log, automatic resume, corruption fallback, and
config-integrity checks.

This module is designed to integrate with the existing trainer without
duplicating existing persistence utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import random as _random
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_HF_CHECKPOINT_REPO = "FerrariKazu/rhan-nxa-checkpoints"
DEFAULT_HF_ROLLING_REPO = "FerrariKazu/rhan-nxa-checkpoints-rolling"
DEFAULT_HF_REPO_TYPE = "dataset"
DATASET_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"


def get_hf_checkpoint_repo() -> str:
    """Get HF checkpoint repository from environment or default."""
    return os.environ.get("RHAN_HF_CHECKPOINT_REPO", DEFAULT_HF_CHECKPOINT_REPO)


def get_hf_rolling_repo() -> str:
    """Get HF rolling repository from environment or default."""
    return os.environ.get("RHAN_HF_ROLLING_REPO", DEFAULT_HF_ROLLING_REPO)


def get_hf_repo_type() -> str:
    """Get HF repo type from environment or default."""
    return os.environ.get("RHAN_HF_REPO_TYPE", DEFAULT_HF_REPO_TYPE)


def get_hf_token() -> Optional[str]:
    """Get HF token from environment. Never logs or exposes the actual token."""
    return os.environ.get("HF_TOKEN")


def is_hf_persistence_enabled(use_hf: bool, hf_token: Optional[str],
                               smoke: bool = False) -> bool:
    """Check if HF persistence should be enabled."""
    if smoke:
        return False
    if not use_hf:
        return False
    if hf_token is None:
        return False
    return True


# ── Runtime environment detection ─────────────────────────────────────────────

def detect_runtime_env() -> str:
    """Detect runtime environment: COLAB, KAGGLE, or unknown."""
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RUN_ENV"):
        return "COLAB"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "KAGGLE"
    return "UNKNOWN"


# ── Run identity ──────────────────────────────────────────────────────────────

def generate_run_id(phase: str, recipe_version: str,
                    config_hash: str) -> str:
    """Generate stable run ID for J1 foundation training."""
    return f"J1_{phase}_{recipe_version}_{config_hash[:12]}"


# ── Checkpoint state capture ──────────────────────────────────────────────────

def capture_checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    training_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    config_hash: str,
    seed: int,
    code_commit: str,
    gpu_name: Optional[str] = None,
    capture_numpy: bool = True,
) -> Dict[str, Any]:
    """Capture ALL state required to resume training accurately."""
    state: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "training_history": list(training_history),
        "config": config,
        "config_hash": config_hash,
        "seed": int(seed),
        "python_rng_state": _random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
        "amp_scaler_state": (
            scaler.state_dict() if scaler is not None else None
        ),
        "code_commit": code_commit,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
    }
    if capture_numpy:
        try:
            import numpy as np
            state["numpy_rng_state"] = np.random.get_state()
        except Exception:
            pass
    return state


# ── SHA-256 computation ───────────────────────────────────────────────────────

def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Atomic checkpoint write ───────────────────────────────────────────────────

def atomic_checkpoint_write(
    checkpoint_dir: str,
    state_dict: Dict[str, Any],
) -> str:
    """Write checkpoint atomically: tmp -> fsync -> rename.

    Returns path to final checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp_path = os.path.join(checkpoint_dir, "checkpoint.pt.tmp")
    final_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    try:
        torch.save(state_dict, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp_path, final_path)
        return final_path
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        raise


# ── Checkpoint metadata ───────────────────────────────────────────────────────

def write_checkpoint_metadata(
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    sha256: str,
    size_bytes: int,
    metric_name: str,
    metric_value: float,
    best_metric: float,
    best_epoch: int,
    config_hash: str,
    code_commit: str,
    run_id: str,
    phase: str,
    environment: str,
    gpu: Optional[str],
    kind: str = "epoch",
) -> Dict[str, Any]:
    """Write metadata.json sidecar for a checkpoint.

    Returns the metadata dict.
    """
    metadata = {
        "stage": "J1.5",
        "experiment": "J1_FOUNDATION",
        "arm": "TRADES",
        "phase": phase,
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "seed": None,  # Will be filled by caller if needed
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "config_hash": config_hash,
        "git_commit": code_commit,
        "dataset_revision": DATASET_REVISION,
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": size_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "gpu": gpu,
        "kind": kind,
    }
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    tmp_meta = metadata_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_meta, metadata_path)
    return metadata


def write_sha256_txt(checkpoint_dir: str, sha256: str) -> None:
    """Write sha256.txt sidecar."""
    sha_path = os.path.join(checkpoint_dir, "sha256.txt")
    tmp_sha = sha_path + ".tmp"
    with open(tmp_sha, "w") as f:
        f.write(sha256 + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_sha, sha_path)


# ── HF upload ─────────────────────────────────────────────────────────────────

def upload_to_hf(
    local_path: str,
    repo_path: str,
    repo_id: str,
    repo_type: str,
    token: str,
    commit_message: str,
) -> bool:
    """Upload file to Hugging Face.

    Returns True on success, False on failure.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
        )
        return True
    except Exception as e:
        print(f"WARNING: HF upload failed ({repo_path}): {e}", flush=True)
        return False


# ── Remote verification ───────────────────────────────────────────────────────

def verify_remote_checkpoint(
    repo_id: str,
    repo_type: str,
    token: str,
    repo_path: str,
    expected_sha256: str,
    expected_size: int,
) -> Tuple[bool, Optional[str]]:
    """Verify remote checkpoint exists and matches expected hash/size.

    Returns (success, error_message).
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # List files to verify existence
        files = list(api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            path=repo_path,
        ))
        if not files:
            return False, f"remote checkpoint {repo_path} not found"

        # Download and verify
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = api.hf_hub_download(
                repo_id=repo_id,
                filename=os.path.basename(repo_path),
                repo_type=repo_type,
                token=token,
                local_dir=tmpdir,
            )
            actual_sha256 = compute_file_sha256(downloaded)
            actual_size = os.path.getsize(downloaded)

            if actual_sha256 != expected_sha256:
                return False, (
                    f"SHA-256 mismatch: expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )
            if actual_size != expected_size:
                return False, (
                    f"size mismatch: expected {expected_size}, "
                    f"got {actual_size}"
                )
            return True, None

    except Exception as e:
        return False, str(e)


# ── Manifest management ───────────────────────────────────────────────────────

def load_or_create_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load existing manifest or create new one."""
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {
        "run_id": None,
        "stage": "J1",
        "arm": "TRADES",
        "phase": None,
        "status": "RUNNING",
        "config_hash": None,
        "git_commit": None,
        "dataset_revision": DATASET_REVISION,
        "epochs": {},
        "latest_committed_epoch": 0,
        "best_epoch": 0,
    }


def update_manifest_atomically(
    manifest_path: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Update manifest atomically: write to tmp -> fsync -> rename."""
    manifest = load_or_create_manifest(manifest_path)
    manifest.update(updates)

    # Ensure epochs dict exists
    if "epochs" not in manifest:
        manifest["epochs"] = {}

    tmp_path = manifest_path + ".tmp"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, manifest_path)
    return manifest


def update_epoch_in_manifest(
    manifest_path: str,
    epoch: int,
    sha256: str,
    status: str = "COMMITTED",
    best: bool = False,
) -> Dict[str, Any]:
    """Update epoch entry in manifest."""
    manifest = load_or_create_manifest(manifest_path)
    manifest.setdefault("epochs", {})[str(epoch)] = {
        "status": status,
        "sha256": sha256,
        "best": best,
    }

    # Update latest committed epoch
    committed = [
        int(e) for e, info in manifest["epochs"].items()
        if info.get("status") == "COMMITTED"
    ]
    if committed:
        manifest["latest_committed_epoch"] = max(committed)

    # Update best epoch
    best_epochs = [
        int(e) for e, info in manifest["epochs"].items()
        if info.get("best") and info.get("status") == "COMMITTED"
    ]
    if best_epochs:
        manifest["best_epoch"] = max(best_epochs)

    return update_manifest_atomically(manifest_path, manifest)


# ── Event log ─────────────────────────────────────────────────────────────────

def log_event(
    events_path: str,
    run_id: str,
    epoch: int,
    event: str,
    sha256: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append event to append-only events.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "epoch": epoch,
        "event": event,
    }
    if sha256:
        record["sha256"] = sha256
    if extra:
        record.update(extra)

    os.makedirs(os.path.dirname(events_path), exist_ok=True)
    # Write directly to file (not atomic for append-only logs)
    with open(events_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── Discover committed epochs from HF ─────────────────────────────────────────

def discover_committed_epochs_from_hf(
    repo_id: str,
    repo_type: str,
    token: str,
    run_id: str,
) -> Dict[int, Dict[str, Any]]:
    """Discover committed epochs from HF immutable archive.

    Returns dict mapping epoch number to metadata.
    """
    discovered: Dict[int, Dict[str, Any]] = {}
    try:
        from huggingface_hub import HfApi
        import re

        api = HfApi(token=token)
        base_path = f"j1_foundation/{run_id}/trades/backbone_only"

        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            path=base_path,
        ):
            if (item.path.endswith("checkpoint.pt")
                    and "/epoch_" in item.path):
                m = re.search(r"epoch_(\d+)/checkpoint\.pt", item.path)
                if m:
                    epoch_num = int(m.group(1))
                    meta_path = item.path.replace("checkpoint.pt", "metadata.json")
                    try:
                        meta_file = api.hf_hub_download(
                            repo_id=repo_id,
                            filename=meta_path,
                            repo_type=repo_type,
                            token=token,
                            local_dir=os.path.join("checkpoints"),
                            local_dir_use_sibling=False,
                        )
                        with open(meta_file) as f:
                            meta = json.load(f)
                        discovered[epoch_num] = {
                            "sha256": meta.get("checkpoint_sha256"),
                            "status": "COMMITTED",
                            "best": meta.get("kind") == "best",
                            "metric_value": meta.get("metric_value"),
                            "best_metric": meta.get("best_metric"),
                            "best_epoch": meta.get("best_epoch"),
                        }
                    except Exception:
                        pass
    except Exception as e:
        print(f"WARNING: could not discover epochs from HF: {e}", flush=True)
    return discovered


# ── Find and restore checkpoint ───────────────────────────────────────────────

def find_best_restore_checkpoint(
    committed_epochs: Dict[int, Dict[str, Any]],
    repo_id: str,
    repo_type: str,
    token: str,
    run_id: str,
    download_dir: str,
) -> Optional[Dict[str, Any]]:
    """Find the best checkpoint to restore, with corruption fallback.

    Searches backward from newest committed epoch.
    Returns checkpoint state dict or None if no valid checkpoint found.
    """
    if not committed_epochs:
        return None

    sorted_epochs = sorted(committed_epochs.keys(), reverse=True)

    for epoch in sorted_epochs:
        info = committed_epochs[epoch]
        if info.get("status") != "COMMITTED":
            continue

        repo_path = (
            f"j1_foundation/{run_id}/trades/backbone_only/"
            f"epoch_{epoch:03d}/checkpoint.pt"
        )

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            downloaded = api.hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                repo_type=repo_type,
                token=token,
                local_dir=download_dir,
                local_dir_use_sibling=False,
            )

            # Verify SHA-256
            actual_sha256 = compute_file_sha256(downloaded)
            expected_sha256 = info.get("sha256")
            if actual_sha256 != expected_sha256:
                print(
                    f"CHECKPOINT_CORRUPT: epoch {epoch} SHA-256 mismatch",
                    flush=True,
                )
                continue

            # Load and return state
            state = torch.load(downloaded, map_location="cpu", weights_only=False)
            print(
                f"RECOVERY_FALLBACK: restored from epoch {epoch}",
                flush=True,
            )
            return state

        except Exception as e:
            print(
                f"CHECKPOINT_CORRUPT: epoch {epoch} download/verify failed: {e}",
                flush=True,
            )
            continue

    return None


# ── HF persistence preflight ──────────────────────────────────────────────────

def run_hf_preflight(
    checkpoint_repo: str,
    rolling_repo: str,
    repo_type: str,
    token: str,
    run_id: str,
) -> bool:
    """Run HF persistence preflight checks.

    Returns True if all checks pass.
    """
    print("J1 HF persistence preflight", flush=True)
    print(f"HF_TOKEN: {'present' if token else 'MISSING'}", flush=True)
    print(f"Archive repo: {checkpoint_repo}", flush=True)
    print(f"Rolling repo: {rolling_repo}", flush=True)
    print(f"Repo type: {repo_type}", flush=True)

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # Check authentication
    try:
        whoami = api.whoami()
        print(f"Authentication: OK (user: {whoami.get('name', 'unknown')})",
              flush=True)
    except Exception as e:
        print(f"Authentication: FAILED ({e})", flush=True)
        return False

    # Check remote access
    try:
        api.list_repo_tree(repo_id=checkpoint_repo, repo_type=repo_type, limit=1)
        print("Remote access: OK", flush=True)
    except Exception as e:
        print(f"Remote access: FAILED ({e})", flush=True)
        return False

    # Write test: create temp artifact and verify
    import tempfile
    probe_content = json.dumps({"probe": True, "timestamp": _time.time()})
    probe_repo_path = f"j1_foundation/.preflight/{run_id}/probe.txt"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as tmp:
            tmp.write(probe_content)
            tmp_path = tmp.name

        # Upload
        if not upload_to_hf(
            tmp_path, probe_repo_path, checkpoint_repo, repo_type, token,
            "J1 HF persistence preflight probe",
        ):
            print("Write test: FAILED", flush=True)
            return False

        # Verify read-back
        downloaded = api.hf_hub_download(
            repo_id=checkpoint_repo,
            filename=probe_repo_path,
            repo_type=repo_type,
            token=token,
            local_dir=os.path.join("checkpoints"),
            local_dir_use_sibling=False,
        )
        with open(downloaded) as f:
            verified = json.load(f)
        if verified.get("probe") is True:
            print("Write test: OK", flush=True)
            print("Read-back verification: OK", flush=True)
        else:
            print("Write test: FAILED (read-back mismatch)", flush=True)
            return False

        # Cleanup probe
        try:
            api.delete_repo_file(
                path_in_repo=probe_repo_path,
                repo_id=checkpoint_repo,
                repo_type=repo_type,
                token=token,
                commit_message="J1 HF persistence preflight cleanup",
            )
            print("Probe cleanup: OK", flush=True)
        except Exception as e:
            print(f"Probe cleanup: WARNING ({e})", flush=True)

        return True

    except Exception as e:
        print(f"Write test: FAILED ({e})", flush=True)
        return False

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Config integrity check ────────────────────────────────────────────────────

def check_config_integrity(
    existing_config_hash: str,
    existing_git_commit: str,
    current_config_hash: str,
    current_git_commit: str,
) -> Tuple[bool, str]:
    """Check if current config matches persisted run config.

    Returns (ok, message).
    """
    if existing_config_hash != current_config_hash:
        return False, (
            f"Existing J1 run has configuration hash {existing_config_hash}. "
            f"Current configuration hash {current_config_hash}. "
            f"Refusing to resume because this would mix experimental conditions. "
            f"Use --new-experiment for a separate run."
        )
    if existing_git_commit != current_git_commit:
        return False, (
            f"Existing J1 run has git commit {existing_git_commit}. "
            f"Current git commit {current_git_commit}. "
            f"Refusing to resume because code has changed."
        )
    return True, "config integrity check passed"


# ── Main persistence coordinator ──────────────────────────────────────────────

class HFPersistenceCoordinator:
    """Coordinates HF persistence for J1 foundation training."""

    def __init__(
        self,
        cfg: Any,
        phase: str,
        run_id: str,
        config_hash: str,
        code_commit: str,
        hf_token: str,
        ckpt_dir: str,
        runs_dir: str,
        use_hf: bool = True,
        smoke: bool = False,
    ):
        self.cfg = cfg
        self.phase = phase
        self.run_id = run_id
        self.config_hash = config_hash
        self.code_commit = code_commit
        self.hf_token = hf_token
        self.ckpt_dir = ckpt_dir
        self.runs_dir = runs_dir
        self.use_hf = use_hf
        self.smoke = smoke

        self.checkpoint_repo = get_hf_checkpoint_repo()
        self.rolling_repo = get_hf_rolling_repo()
        self.repo_type = get_hf_repo_type()
        self.environment = detect_runtime_env()

        self.enabled = is_hf_persistence_enabled(use_hf, hf_token, smoke)
        self.committed_epochs: Dict[int, Dict[str, Any]] = {}
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.training_history: List[Dict[str, Any]] = []
        self.global_step = 0

        # Paths
        self.manifest_path = os.path.join(runs_dir, run_id, "manifest.json")
        self.events_path = os.path.join(runs_dir, run_id, "events.jsonl")

    def preflight(self) -> bool:
        """Run HF persistence preflight."""
        if not self.enabled:
            print(f"[{self.phase}] HF persistence preflight skipped (HF disabled)",
                  flush=True)
            return False
        return run_hf_preflight(
            self.checkpoint_repo,
            self.rolling_repo,
            self.repo_type,
            self.hf_token,
            self.run_id,
        )

    def discover_committed_epochs(self) -> Dict[int, Dict[str, Any]]:
        """Discover committed epochs from HF."""
        if not self.enabled:
            return {}
        self.committed_epochs = discover_committed_epochs_from_hf(
            self.checkpoint_repo,
            self.repo_type,
            self.hf_token,
            self.run_id,
        )
        return self.committed_epochs

    def reconcile_manifest(self) -> None:
        """Reconcile manifest with discovered HF state."""
        if not self.enabled or not self.committed_epochs:
            return

        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        manifest = load_or_create_manifest(self.manifest_path)

        # Ensure run identity
        if manifest.get("run_id") is None:
            manifest["run_id"] = self.run_id
            manifest["stage"] = "J1"
            manifest["arm"] = "TRADES"
            manifest["phase"] = self.phase
            manifest["config_hash"] = self.config_hash
            manifest["git_commit"] = self.code_commit
            manifest["dataset_revision"] = DATASET_REVISION

        # Reconcile epochs
        for epoch, info in self.committed_epochs.items():
            epoch_str = str(epoch)
            if epoch_str not in manifest["epochs"]:
                manifest["epochs"][epoch_str] = info
            elif manifest["epochs"][epoch_str].get("status") != "COMMITTED":
                manifest["epochs"][epoch_str]["status"] = "COMMITTED"
                manifest["epochs"][epoch_str]["sha256"] = info["sha256"]

        # Update latest committed
        committed = [
            int(e) for e, info in manifest["epochs"].items()
            if info.get("status") == "COMMITTED"
        ]
        if committed:
            manifest["latest_committed_epoch"] = max(committed)
            self.start_epoch = max(committed)
        else:
            self.start_epoch = 0

        # Update best epoch
        best_epochs = [
            int(e) for e, info in manifest["epochs"].items()
            if info.get("best") and info.get("status") == "COMMITTED"
        ]
        if best_epochs:
            manifest["best_epoch"] = max(best_epochs)
            self.best_epoch = max(best_epochs)
            # Get best metric from history
            for e in sorted(committed, reverse=True):
                if manifest["epochs"][str(e)].get("best"):
                    self.best_metric = manifest["epochs"][str(e)].get("metric_value", -float('inf'))
                    break

        # Write reconciled manifest
        update_manifest_atomically(self.manifest_path, manifest)

    def get_start_epoch(self) -> int:
        """Get the epoch to start training from."""
        return getattr(self, 'start_epoch', 0)

    def persist_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[Any],
        epoch: int,
        global_step: int,
        val_acc: float,
        training_history: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> bool:
        """Persist an epoch checkpoint to HF.

        Returns True if epoch was successfully committed.
        """
        self.global_step = global_step
        self.training_history = list(training_history)

        # Check if this is a new best
        is_best = val_acc > self.best_metric
        if is_best:
            self.best_metric = val_acc
            self.best_epoch = epoch

        if not self.enabled:
            return False

        # Epoch directory for local storage
        epoch_dir = os.path.join(
            self.ckpt_dir,
            "hf_checkpoints",
            "j1_foundation",
            self.run_id,
            "trades",
            "backbone_only",
            f"epoch_{epoch:03d}",
        )

        # Capture full state
        checkpoint_state = capture_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            training_history=training_history,
            config=config,
            config_hash=self.config_hash,
            seed=int(self.cfg.seed) if hasattr(self.cfg, 'seed') else 41,
            code_commit=self.code_commit,
            gpu_name=(torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None),
        )

        # Write atomic checkpoint
        checkpoint_path = atomic_checkpoint_write(epoch_dir, checkpoint_state)

        # Compute SHA-256 and size
        sha256 = compute_file_sha256(checkpoint_path)
        size_bytes = os.path.getsize(checkpoint_path)

        # Write metadata and sha256 sidecars
        write_checkpoint_metadata(
            checkpoint_dir=epoch_dir,
            epoch=epoch,
            global_step=global_step,
            sha256=sha256,
            size_bytes=size_bytes,
            metric_name="val_acc",
            metric_value=val_acc,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            config_hash=self.config_hash,
            code_commit=self.code_commit,
            run_id=self.run_id,
            phase=self.phase,
            environment=self.environment,
            gpu=(torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else None),
            kind="epoch",
        )
        write_sha256_txt(epoch_dir, sha256)

        # Upload to immutable archive
        archive_repo_path = (
            f"j1_foundation/{self.run_id}/trades/backbone_only/"
            f"epoch_{epoch:03d}/checkpoint.pt"
        )
        upload_ok = upload_to_hf(
            checkpoint_path,
            archive_repo_path,
            self.checkpoint_repo,
            self.repo_type,
            self.hf_token,
            f"epoch {epoch} checkpoint: {self.run_id}",
        )

        if not upload_ok:
            log_event(
                self.events_path, self.run_id, epoch,
                "HF_ARCHIVE_UPLOAD_FAILED", sha256,
            )
            print(
                f"[{self.phase}] WARNING: epoch {epoch} upload failed",
                flush=True,
            )
            return False

        # Verify remote
        verified, error = verify_remote_checkpoint(
            self.checkpoint_repo,
            self.repo_type,
            self.hf_token,
            archive_repo_path,
            sha256,
            size_bytes,
        )

        if not verified:
            log_event(
                self.events_path, self.run_id, epoch,
                "HF_ARCHIVE_VERIFICATION_FAILED", sha256,
                {"error": error},
            )
            print(
                f"[{self.phase}] WARNING: epoch {epoch} verification failed: {error}",
                flush=True,
            )
            return False

        # Update manifest
        update_epoch_in_manifest(
            self.manifest_path,
            epoch,
            sha256,
            status="COMMITTED",
            best=is_best,
        )

        # Log events
        log_event(self.events_path, self.run_id, epoch, "CHECKPOINT_LOCAL_COMMITTED", sha256)
        log_event(self.events_path, self.run_id, epoch, "HF_ARCHIVE_UPLOAD_STARTED", sha256)
        log_event(self.events_path, self.run_id, epoch, "HF_ARCHIVE_UPLOAD_COMPLETED", sha256)
        log_event(self.events_path, self.run_id, epoch, "HF_ARCHIVE_VERIFIED", sha256)
        log_event(self.events_path, self.run_id, epoch, "CHECKPOINT_COMMITTED", sha256)

        # Update rolling mirror
        self._update_rolling_mirror(
            checkpoint_state, epoch, global_step, val_acc, sha256
        )

        if is_best:
            self._persist_best_checkpoint(
                checkpoint_state, epoch, global_step, val_acc, sha256
            )
            log_event(
                self.events_path, self.run_id, epoch,
                "BEST_CHECKPOINT_COMMITTED", sha256,
            )

        print(
            f"[{self.phase}] Epoch {epoch} COMMITTED to HF (sha256={sha256[:16]}...)",
            flush=True,
        )
        return True

    def _update_rolling_mirror(
        self,
        checkpoint_state: Dict[str, Any],
        epoch: int,
        global_step: int,
        val_acc: float,
        sha256: str,
    ) -> None:
        """Update rolling recovery mirror."""
        rolling_dir = os.path.join(
            self.ckpt_dir,
            "hf_checkpoints",
            "j1_foundation",
            self.run_id,
            "trades",
            "backbone_only",
            "latest",
        )

        # Write checkpoint
        rolling_path = atomic_checkpoint_write(rolling_dir, checkpoint_state)
        rolling_sha256 = compute_file_sha256(rolling_path)
        rolling_size = os.path.getsize(rolling_path)

        # Write metadata
        write_checkpoint_metadata(
            checkpoint_dir=rolling_dir,
            epoch=epoch,
            global_step=global_step,
            sha256=rolling_sha256,
            size_bytes=rolling_size,
            metric_name="val_acc",
            metric_value=val_acc,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            config_hash=self.config_hash,
            code_commit=self.code_commit,
            run_id=self.run_id,
            phase=self.phase,
            environment=self.environment,
            gpu=(torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else None),
            kind="rolling",
        )
        write_sha256_txt(rolling_dir, rolling_sha256)

        # Upload to rolling repo
        rolling_repo_base = (
            f"j1_foundation/{self.run_id}/trades/backbone_only/latest"
        )
        for filename in ["checkpoint.pt", "metadata.json", "sha256.txt"]:
            local_file = os.path.join(rolling_dir, filename)
            if os.path.exists(local_file):
                upload_to_hf(
                    local_file,
                    f"{rolling_repo_base}/{filename}",
                    self.rolling_repo,
                    self.repo_type,
                    self.hf_token,
                    f"rolling update: epoch {epoch}",
                )

        # Write state.json
        state_json = {
            "run_id": self.run_id,
            "latest_epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_metric": float(self.best_metric),
            "config_hash": self.config_hash,
            "git_commit": self.code_commit,
            "sha256": rolling_sha256,
        }
        state_json_path = os.path.join(rolling_dir, "state.json")
        tmp_state = state_json_path + ".tmp"
        with open(tmp_state, "w") as f:
            json.dump(state_json, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_state, state_json_path)

        # Upload state.json
        upload_to_hf(
            state_json_path,
            f"{rolling_repo_base}/state.json",
            self.rolling_repo,
            self.repo_type,
            self.hf_token,
            f"state update: epoch {epoch}",
        )

        log_event(
            self.events_path, self.run_id, epoch,
            "ROLLING_UPDATED", rolling_sha256,
        )

    def _persist_best_checkpoint(
        self,
        checkpoint_state: Dict[str, Any],
        epoch: int,
        global_step: int,
        val_acc: float,
        sha256: str,
    ) -> None:
        """Persist best checkpoint as immutable artifact."""
        best_dir = os.path.join(
            self.ckpt_dir,
            "hf_checkpoints",
            "j1_foundation",
            self.run_id,
            "trades",
            "backbone_only",
            "best",
            f"epoch_{epoch:03d}",
        )

        # Write checkpoint
        best_path = atomic_checkpoint_write(best_dir, checkpoint_state)
        best_sha256 = compute_file_sha256(best_path)
        best_size = os.path.getsize(best_path)

        # Write metadata
        write_checkpoint_metadata(
            checkpoint_dir=best_dir,
            epoch=epoch,
            global_step=global_step,
            sha256=best_sha256,
            size_bytes=best_size,
            metric_name="val_acc",
            metric_value=val_acc,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            config_hash=self.config_hash,
            code_commit=self.code_commit,
            run_id=self.run_id,
            phase=self.phase,
            environment=self.environment,
            gpu=(torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else None),
            kind="best",
        )
        write_sha256_txt(best_dir, best_sha256)

        # Upload to immutable archive
        best_repo_base = (
            f"j1_foundation/{self.run_id}/trades/backbone_only/best/"
            f"epoch_{epoch:03d}"
        )
        for filename in ["checkpoint.pt", "metadata.json", "sha256.txt"]:
            local_file = os.path.join(best_dir, filename)
            if os.path.exists(local_file):
                upload_to_hf(
                    local_file,
                    f"{best_repo_base}/{filename}",
                    self.checkpoint_repo,
                    self.repo_type,
                    self.hf_token,
                    f"best checkpoint: epoch {epoch}",
                )

    def restore_from_hf(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Restore state from HF.

        Returns (start_epoch, restored_state).
        """
        if not self.enabled:
            return 0, {}

        # Discover committed epochs
        self.committed_epochs = self.discover_committed_epochs()
        if not self.committed_epochs:
            return 0, {}

        # Reconcile manifest
        self.reconcile_manifest()

        # Find best restore checkpoint
        state = find_best_restore_checkpoint(
            self.committed_epochs,
            self.checkpoint_repo,
            self.repo_type,
            self.hf_token,
            self.run_id,
            self.ckpt_dir,
        )

        if state is None:
            if self.committed_epochs:
                raise RuntimeError(
                    "Could not restore any committed checkpoint from HF. "
                    "Training cannot continue safely."
                )
            return 0, {}

        # Restore state
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

        if state.get("scaler_state_dict") and scaler is not None:
            scaler.load_state_dict(state["scaler_state_dict"])

        start_epoch = int(state.get("epoch", 0))
        self.best_metric = float(state.get("best_metric", -float('inf')))
        self.best_epoch = int(state.get("best_epoch", 0))
        self.training_history = list(state.get("training_history", []))
        self.global_step = int(state.get("global_step", 0))

        # Restore RNG states
        if state.get("python_rng_state"):
            _random.setstate(state["python_rng_state"])
        if state.get("numpy_rng_state"):
            try:
                import numpy as np
                np.random.set_state(state["numpy_rng_state"])
            except Exception:
                pass
        if state.get("torch_cpu_rng_state"):
            torch.set_rng_state(state["torch_cpu_rng_state"])
        if state.get("torch_cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["torch_cuda_rng_state"])

        print(
            f"[{self.phase}] Resumed from epoch {start_epoch} "
            f"(best_metric={self.best_metric:.4f} at epoch {self.best_epoch})",
            flush=True,
        )
        return start_epoch, state

    def check_config_mismatch(
        self,
        existing_manifest: Optional[Dict[str, Any]],
    ) -> None:
        """Check for config mismatch and raise if detected."""
        if existing_manifest is None:
            return

        existing_hash = existing_manifest.get("config_hash")
        existing_commit = existing_manifest.get("git_commit")

        if existing_hash and existing_hash != self.config_hash:
            raise SystemExit(
                f"ERROR: Existing J1 run has configuration hash {existing_hash}.\n"
                f"Current configuration hash {self.config_hash}.\n"
                f"Refusing to resume because this would mix experimental conditions.\n"
                f"Use --new-experiment for a separate run."
            )

        if existing_commit and existing_commit != self.code_commit:
            raise SystemExit(
                f"ERROR: Existing J1 run has git commit {existing_commit}.\n"
                f"Current git commit {self.code_commit}.\n"
                f"Refusing to resume because code has changed."
            )

    def training_complete(self) -> None:
        """Log training completion."""
        if not self.enabled:
            return
        log_event(
            self.events_path, self.run_id, self.cfg.epochs if hasattr(self.cfg, 'epochs') else 60,
            "TRAINING_COMPLETED",
            extra={
                "best_val_acc": float(self.best_metric),
                "best_epoch": int(self.best_epoch),
                "run_id": self.run_id,
            },
        )
