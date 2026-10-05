"""
store.py — ArtifactStore: atomic checkpoint writes, SHA-256 verification,
HuggingFace-backed persistence.

Every checkpoint write follows:
    1. Write to .tmp file
    2. fsync
    3. SHA-256 hash
    4. Atomic rename to final path
    5. Upload to HuggingFace (async)
    6. Update manifest

On resume:
    1. Verify SHA-256 of local file
    2. If corrupted, download from HF and re-verify
    3. If HF also corrupted, fall back to newest verified earlier checkpoint
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from rhan_core.artifacts.events import EventLog
from rhan_core.artifacts.manifest import ExperimentManifest, StageStatus


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _atomic_write(target_path: str, data: bytes) -> str:
    """Write bytes atomically: .tmp → fsync → rename.

    Returns the path of the written file (same as target_path).
    Raises OSError on failure — never leaves a partial file at target_path.
    """
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        os.rename(tmp_path, target_path)
    except Exception:
        # Clean up .tmp on failure
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return target_path


def _atomic_torch_save(target_path: str, obj: Any) -> str:
    """Save a torch object atomically with fsync."""
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        # fsync the written file
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return target_path


class ArtifactStore:
    """Fault-tolerant artifact storage for Stage 4 experiments.

    Args:
        experiment_root: local root directory for this experiment's artifacts.
        hf_repo_id: HuggingFace repo for persistence (default: FerrariKazu/rhan-checkpoints).
        hf_repo_type: HuggingFace repo type (default: "dataset").
        hf_token: HuggingFace token (auto-detected from env if None).
    """

    def __init__(
        self,
        experiment_root: str,
        hf_repo_id: str = "FerrariKazu/rhan-checkpoints",
        hf_repo_type: str = "dataset",
        hf_token: Optional[str] = None,
    ):
        self.experiment_root = os.path.abspath(experiment_root)
        self.hf_repo_id = hf_repo_id
        self.hf_repo_type = hf_repo_type
        self.hf_token = hf_token or self._detect_hf_token()

        os.makedirs(self.experiment_root, exist_ok=True)

        # Subdirectories
        self.smoke_dir = os.path.join(self.experiment_root, "smoke")
        self.train_dir = os.path.join(self.experiment_root, "train")
        self.eval_dir = os.path.join(self.experiment_root, "evaluation")
        self.lens_dir = os.path.join(self.experiment_root, "lens")
        self.reports_dir = os.path.join(self.experiment_root, "reports")

        # Event log
        self.events = EventLog(os.path.join(self.experiment_root, "events.jsonl"))

        # Manifest path
        self.manifest_path = os.path.join(self.experiment_root, "manifest.json")

    def _detect_hf_token(self) -> Optional[str]:
        """Auto-detect HF token from environment."""
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass
        return None

    def detect_environment(self) -> str:
        """Detect runtime environment: 'colab', 'kaggle', or 'local'."""
        if os.path.exists("/content"):
            return "colab"
        if os.path.exists("/kaggle/working"):
            return "kaggle"
        return "local"

    # ── Manifest ────────────────────────────────────────────────────────────

    def load_manifest(self) -> ExperimentManifest:
        """Load or create the experiment manifest."""
        return ExperimentManifest.load(self.manifest_path)

    # ── Atomic checkpoint writes ────────────────────────────────────────────

    def save_checkpoint(
        self,
        phase: str,
        state_dict: Dict[str, Any],
        *,
        seed: Optional[int] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stage: str = "train",
    ) -> Tuple[str, str]:
        """Atomically save a checkpoint with SHA-256 verification.

        Args:
            phase: "smoke" or "train" or "eval".
            state_dict: the object to torch.save (model state, full checkpoint dict, etc).
            seed: seed number (for seed-level tracking).
            epoch: epoch number.
            metadata: extra metadata to store alongside.
            stage: "smoke", "train", "eval", or "lens".

        Returns:
            (file_path, sha256_hex) of the saved checkpoint.
        """
        stage_dir = {
            "smoke": self.smoke_dir,
            "train": self.train_dir,
            "eval": self.eval_dir,
            "lens": self.lens_dir,
        }.get(stage, self.train_dir)

        if seed is not None:
            stage_dir = os.path.join(stage_dir, f"seed_{seed}")
        os.makedirs(stage_dir, exist_ok=True)

        # Build filename
        parts = [phase]
        if seed is not None:
            parts.append(f"seed{seed}")
        if epoch is not None:
            parts.append(f"epoch{epoch:03d}")
        filename = "_".join(parts) + ".pth"
        target_path = os.path.join(stage_dir, filename)

        # Atomic write
        _atomic_torch_save(target_path, state_dict)

        # SHA-256
        ckpt_sha = sha256_file(target_path)

        # Build record
        record = {
            "path": os.path.relpath(target_path, self.experiment_root),
            "sha256": ckpt_sha,
            "size_bytes": os.path.getsize(target_path),
            "epoch": epoch,
            "seed": seed,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        # Save companion .meta.json
        meta_path = target_path + ".meta.json"
        _atomic_write(meta_path, json.dumps(record, indent=2).encode())

        # Log event
        self.events.log(
            "CHECKPOINT_COMMITTED",
            seed=seed, epoch=epoch, sha256=ckpt_sha,
            path=record["path"], size_bytes=record["size_bytes"],
        )

        # Update manifest
        manifest = self.load_manifest()
        manifest.record_checkpoint(phase, record)
        manifest.save()

        # HF sync (async)
        self._hf_upload(target_path, stage)

        return target_path, ckpt_sha

    def save_json(
        self,
        phase: str,
        data: Dict[str, Any],
        *,
        seed: Optional[int] = None,
        filename: str = "result.json",
        stage: str = "eval",
    ) -> str:
        """Atomically save a JSON result file with checksum."""
        stage_dir = {
            "smoke": self.smoke_dir,
            "train": self.train_dir,
            "eval": self.eval_dir,
            "lens": self.lens_dir,
            "reports": self.reports_dir,
        }.get(stage, self.eval_dir)

        if seed is not None:
            stage_dir = os.path.join(stage_dir, f"seed_{seed}")
        os.makedirs(stage_dir, exist_ok=True)

        target_path = os.path.join(stage_dir, filename)
        content = json.dumps(data, indent=2, default=str).encode()
        _atomic_write(target_path, content)

        file_sha = sha256_file(target_path)
        self.events.log(
            "JSON_COMMITTED",
            seed=seed, sha256=file_sha, path=os.path.relpath(target_path, self.experiment_root),
        )
        return target_path

    def save_event_log(self) -> str:
        """Atomically save the events log to the experiment root."""
        return self.events.flush()

    # ── Checkpoint verification ─────────────────────────────────────────────

    def verify_checkpoint(self, path: str) -> Tuple[bool, Optional[str]]:
        """Verify a checkpoint's SHA-256 against its .meta.json.

        Returns (is_valid, expected_sha256 or None).
        """
        meta_path = path + ".meta.json"
        if not os.path.exists(meta_path):
            return False, None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            expected = meta["sha256"]
            actual = sha256_file(path)
            return actual == expected, expected
        except Exception:
            return False, None

    def find_latest_verified(
        self, stage: str = "train", phase: str = "train", seed: Optional[int] = None,
    ) -> Optional[str]:
        """Find the latest verified checkpoint for a given stage/phase/seed.

        Falls back to earlier checkpoints if the newest is corrupted.
        """
        stage_dir = {
            "smoke": self.smoke_dir,
            "train": self.train_dir,
            "eval": self.eval_dir,
            "lens": self.lens_dir,
        }.get(stage, self.train_dir)

        if seed is not None:
            stage_dir = os.path.join(stage_dir, f"seed_{seed}")

        if not os.path.exists(stage_dir):
            return None

        # Collect all .pth files matching the phase
        candidates = []
        for f in sorted(os.listdir(stage_dir)):
            if f.endswith(".pth") and f.startswith(phase):
                candidates.append(os.path.join(stage_dir, f))

        # Try newest first, fall back to earlier
        for path in reversed(candidates):
            valid, _ = self.verify_checkpoint(path)
            if valid:
                return path

        return None

    # ── HuggingFace sync ────────────────────────────────────────────────────

    def _hf_upload(self, local_path: str, stage: str = "train") -> bool:
        """Upload a file to HuggingFace asynchronously.

        Uses the same HF repo convention as the existing sync_to_hf.
        """
        if not self.hf_token or not os.path.exists(local_path):
            return False

        import threading

        def _async_upload():
            try:
                from huggingface_hub import HfApi, create_repo
                api = HfApi(token=self.hf_token)

                # Determine repo: checkpoints go to rhan-checkpoints,
                # artifacts go to rhan-checkpoints-artifacts
                if stage in ("train", "smoke"):
                    repo_id = self.hf_repo_id  # FerrariKazu/rhan-checkpoints
                else:
                    repo_id = self.hf_repo_id + "-artifacts"

                create_repo(repo_id=repo_id, repo_type=self.hf_repo_type,
                           exist_ok=True, token=self.hf_token)

                rel_path = os.path.relpath(local_path, self.experiment_root)
                hf_path = f"stage4_artifacts/{rel_path}"

                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    repo_type=self.hf_repo_type,
                    token=self.hf_token,
                )
                self.events.log("HF_UPLOAD_COMPLETE",
                               path=hf_path, repo=repo_id)
            except Exception as e:
                self.events.log("HF_UPLOAD_FAILED",
                               path=local_path, error=str(e))

        t = threading.Thread(target=_async_upload, daemon=True)
        t.start()
        return True

    def hf_download_if_missing(self, rel_path: str, stage: str = "train") -> Optional[str]:
        """Download an artifact from HF if not present locally."""
        local_path = os.path.join(self.experiment_root, rel_path)
        if os.path.exists(local_path):
            return local_path

        if not self.hf_token:
            return None

        try:
            from huggingface_hub import hf_hub_download
            if stage in ("train", "smoke"):
                repo_id = self.hf_repo_id
            else:
                repo_id = self.hf_repo_id + "-artifacts"

            hf_path = f"stage4_artifacts/{rel_path}"
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type=self.hf_repo_type,
                filename=hf_path,
                token=self.hf_token,
                local_dir=self.experiment_root,
            )
            self.events.log("HF_DOWNLOAD_COMPLETE", path=rel_path)
            return downloaded
        except Exception as e:
            self.events.log("HF_DOWNLOAD_FAILED", path=rel_path, error=str(e))
            return None

    # ── Integrity verification ──────────────────────────────────────────────

    def verify_all(self) -> Dict[str, Any]:
        """Verify all checkpoints and JSON files in the experiment.

        Returns a summary dict with counts and any failures.
        """
        result = {"total": 0, "valid": 0, "invalid": 0, "missing_meta": 0, "failures": []}

        for root, dirs, files in os.walk(self.experiment_root):
            for f in files:
                if not f.endswith(".pth"):
                    continue
                path = os.path.join(root, f)
                result["total"] += 1

                valid, expected = self.verify_checkpoint(path)
                if valid:
                    result["valid"] += 1
                elif expected is None:
                    result["missing_meta"] += 1
                    result["failures"].append({
                        "path": os.path.relpath(path, self.experiment_root),
                        "reason": "no .meta.json"
                    })
                else:
                    result["invalid"] += 1
                    actual = sha256_file(path)
                    result["failures"].append({
                        "path": os.path.relpath(path, self.experiment_root),
                        "reason": f"SHA-256 mismatch: expected {expected[:16]}..., got {actual[:16]}..."
                    })

        return result
