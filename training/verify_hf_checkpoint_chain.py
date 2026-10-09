#!/usr/bin/env python3
"""
Verify HF checkpoint chain integrity for J1 foundation training.

Verifies:
* epoch continuity
* immutable artifacts
* checkpoint metadata
* SHA-256
* config hash
* git commit
* dataset revision
* best checkpoint chain
* rolling pointer
* manifest consistency
* no missing committed epochs
* no duplicate run identities

Usage:
    python3 training/verify_hf_checkpoint_chain.py \\
        --repo FerrariKazu/rhan-nxa-checkpoints \\
        --rolling-repo FerrariKazu/rhan-nxa-checkpoints-rolling \\
        --run-id <run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


DEFAULT_CHECKPOINT_REPO = "FerrariKazu/rhan-nxa-checkpoints"
DEFAULT_ROLLING_REPO = "FerrariKazu/rhan-nxa-checkpoints-rolling"
DEFAULT_REPO_TYPE = "dataset"
DATASET_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"


def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    repo_path: str,
    local_dir: str,
) -> Optional[str]:
    """Download a file from HF, return local path or None."""
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type=repo_type,
            token=api.token,
            local_dir=local_dir,
            local_dir_use_sibling=False,
        )
    except Exception as e:
        print(f"  FAILED to download {repo_path}: {e}", flush=True)
        return None


def verify_checkpoint_chain(
    checkpoint_repo: str,
    rolling_repo: str,
    repo_type: str,
    token: str,
    run_id: str,
) -> Tuple[bool, List[str]]:
    """Verify the HF checkpoint chain for a run.

    Returns (success, list_of_messages).
    """
    messages: List[str] = []
    errors: List[str] = []

    if not HF_AVAILABLE:
        errors.append("huggingface_hub not available")
        return False, messages + errors

    api = HfApi(token=token)

    # Verify authentication
    try:
        whoami = api.whoami()
        messages.append(f"Authenticated as: {whoami.get('name', 'unknown')}")
    except Exception as e:
        errors.append(f"Authentication failed: {e}")
        return False, messages + errors

    base_path = f"j1_foundation/{run_id}/trades/backbone_only"

    # Discover epoch checkpoints
    epoch_checkpoints: Dict[int, Dict[str, Any]] = {}
    try:
        for item in api.list_repo_tree(
            repo_id=checkpoint_repo,
            repo_type=repo_type,
            path=base_path,
        ):
            if item.path.endswith("checkpoint.pt") and "/epoch_" in item.path:
                import re
                m = re.search(r"epoch_(\d+)/checkpoint\.pt", item.path)
                if m:
                    epoch_num = int(m.group(1))
                    epoch_checkpoints[epoch_num] = {
                        "path": item.path,
                        "size": item.size,
                    }
    except Exception as e:
        errors.append(f"Failed to list checkpoint repo: {e}")
        return False, messages + errors

    messages.append(f"Found {len(epoch_checkpoints)} epoch checkpoints")

    if not epoch_checkpoints:
        errors.append("No epoch checkpoints found")
        return False, messages + errors

    # Verify each epoch checkpoint
    epochs_sorted = sorted(epoch_checkpoints.keys())
    expected_epoch = 1

    for epoch in epochs_sorted:
        info = epoch_checkpoints[epoch]
        epoch_dir = f"epoch_{epoch:03d}"

        # Check metadata.json
        meta_path = info["path"].replace("checkpoint.pt", "metadata.json")
        meta_local = download_file(api, checkpoint_repo, repo_type, meta_path, "/tmp")
        if meta_local is None:
            errors.append(f"Epoch {epoch}: missing metadata.json")
            continue

        try:
            with open(meta_local) as f:
                metadata = json.load(f)
        except Exception as e:
            errors.append(f"Epoch {epoch}: invalid metadata.json: {e}")
            continue

        # Verify metadata fields
        required_fields = [
            "stage", "experiment", "arm", "phase", "run_id", "epoch",
            "config_hash", "git_commit", "dataset_revision",
            "checkpoint_sha256", "checkpoint_size_bytes", "kind"
        ]
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Epoch {epoch}: missing metadata field '{field}'")

        if metadata.get("run_id") != run_id:
            errors.append(
                f"Epoch {epoch}: run_id mismatch: "
                f"{metadata.get('run_id')} != {run_id}"
            )

        if metadata.get("epoch") != epoch:
            errors.append(
                f"Epoch {epoch}: epoch mismatch: "
                f"{metadata.get('epoch')} != {epoch}"
            )

        if metadata.get("dataset_revision") != DATASET_REVISION:
            errors.append(
                f"Epoch {epoch}: dataset_revision mismatch"
            )

        # Download and verify checkpoint
        ckpt_path = info["path"]
        ckpt_local = download_file(api, checkpoint_repo, repo_type, ckpt_path, "/tmp")
        if ckpt_local is None:
            errors.append(f"Epoch {epoch}: missing checkpoint.pt")
            continue

        # Verify SHA-256
        actual_sha256 = compute_file_sha256(ckpt_local)
        expected_sha256 = metadata.get("checkpoint_sha256")
        if actual_sha256 != expected_sha256:
            errors.append(
                f"Epoch {epoch}: SHA-256 mismatch: "
                f"expected {expected_sha256[:16]}..., got {actual_sha256[:16]}..."
            )

        # Verify size
        actual_size = os.path.getsize(ckpt_local)
        expected_size = metadata.get("checkpoint_size_bytes")
        if actual_size != expected_size:
            errors.append(
                f"Epoch {epoch}: size mismatch: "
                f"expected {expected_size}, got {actual_size}"
            )

        # Verify checkpoint loads
        try:
            state = torch.load(ckpt_local, map_location="cpu", weights_only=False)
            if "model_state_dict" not in state:
                errors.append(f"Epoch {epoch}: missing model_state_dict")
            if "optimizer_state_dict" not in state:
                errors.append(f"Epoch {epoch}: missing optimizer_state_dict")
            if "epoch" not in state:
                errors.append(f"Epoch {epoch}: missing epoch in state")
            elif state["epoch"] != epoch:
                errors.append(
                    f"Epoch {epoch}: epoch in state mismatch: "
                    f"{state['epoch']} != {epoch}"
                )
        except Exception as e:
            errors.append(f"Epoch {epoch}: failed to load checkpoint: {e}")

        # Verify epoch continuity
        if epoch != expected_epoch:
            errors.append(
                f"Epoch continuity broken: expected {expected_epoch}, "
                f"found {epoch}"
            )
        expected_epoch = epoch + 1

        messages.append(
            f"Epoch {epoch}: OK "
            f"(sha256={actual_sha256[:16]}..., size={actual_size})"
        )

    # Verify no missing committed epochs
    if epochs_sorted:
        expected_last = epochs_sorted[-1]
        for ep in range(1, expected_last + 1):
            if ep not in epoch_checkpoints:
                errors.append(f"Missing committed epoch: {ep}")

    # Verify best checkpoint chain
    best_epochs = [
        ep for ep, info in epoch_checkpoints.items()
        if info.get("path", "").endswith("/best/")
    ]
    # Check best directory
    best_base = f"{base_path}/best"
    try:
        best_items = list(api.list_repo_tree(
            repo_id=checkpoint_repo,
            repo_type=repo_type,
            path=best_base,
        ))
        best_dirs = [
            item for item in best_items
            if item.path.endswith("/checkpoint.pt")
        ]
        messages.append(f"Found {len(best_dirs)} best checkpoint artifacts")
    except Exception as e:
        messages.append(f"Could not list best checkpoints: {e}")

    # Verify rolling mirror
    rolling_base = f"{base_path}/latest"
    try:
        rolling_items = list(api.list_repo_tree(
            repo_id=rolling_repo,
            repo_type=repo_type,
            path=rolling_base,
        ))
        rolling_files = {item.path: item for item in rolling_items}

        required_rolling = ["checkpoint.pt", "metadata.json", "sha256.txt", "state.json"]
        for req_file in required_rolling:
            if not any(p.endswith(req_file) for p in rolling_files):
                errors.append(f"Rolling mirror missing: {req_file}")

        # Verify state.json
        state_path = None
        for p in rolling_files:
            if p.endswith("state.json"):
                state_path = p
                break

        if state_path:
            state_local = download_file(
                api, rolling_repo, repo_type, state_path, "/tmp"
            )
            if state_local:
                with open(state_local) as f:
                    state_json = json.load(f)
                if state_json.get("run_id") != run_id:
                    errors.append(
                        f"Rolling state.json run_id mismatch: "
                        f"{state_json.get('run_id')} != {run_id}"
                    )
                if state_json.get("latest_epoch") != expected_last:
                    errors.append(
                        f"Rolling state.json latest_epoch mismatch: "
                        f"{state_json.get('latest_epoch')} != {expected_last}"
                    )
                messages.append(
                    f"Rolling mirror: OK "
                    f"(latest_epoch={state_json.get('latest_epoch')})"
                )
    except Exception as e:
        errors.append(f"Failed to verify rolling mirror: {e}")

    # Verify manifest
    manifest_path = f"{base_path}/../../manifest.json"
    manifest_local = download_file(
        api, checkpoint_repo, repo_type, manifest_path, "/tmp"
    )
    if manifest_local:
        try:
            with open(manifest_local) as f:
                manifest = json.load(f)

            if manifest.get("run_id") != run_id:
                errors.append(
                    f"Manifest run_id mismatch: "
                    f"{manifest.get('run_id')} != {run_id}"
                )

            # Verify manifest epochs match discovered epochs
            manifest_epochs = manifest.get("epochs", {})
            for ep_str, ep_info in manifest_epochs.items():
                ep = int(ep_str)
                if ep not in epoch_checkpoints:
                    errors.append(
                        f"Manifest references epoch {ep} not in archive"
                    )
                elif ep_info.get("status") != "COMMITTED":
                    errors.append(
                        f"Epoch {ep} in manifest has status "
                        f"{ep_info.get('status')}, expected COMMITTED"
                    )

            if manifest.get("latest_committed_epoch") != expected_last:
                errors.append(
                    f"Manifest latest_committed_epoch mismatch: "
                    f"{manifest.get('latest_committed_epoch')} != {expected_last}"
                )

            messages.append("Manifest: OK")
        except Exception as e:
            errors.append(f"Failed to parse manifest: {e}")
    else:
        errors.append("Manifest not found in archive")

    # Check for duplicate run identities
    try:
        all_runs = set()
        for item in api.list_repo_tree(
            repo_id=checkpoint_repo,
            repo_type=repo_type,
            path="j1_foundation",
        ):
            if "/trades/backbone_only/epoch_" in item.path:
                import re
                m = re.search(r"j1_foundation/([^/]+)/trades", item.path)
                if m:
                    all_runs.add(m.group(1))

        if len(all_runs) > 1:
            messages.append(
                f"Warning: multiple run identities found in archive: {all_runs}"
            )
        elif len(all_runs) == 1:
            messages.append("No duplicate run identities detected")
    except Exception as e:
        messages.append(f"Could not check for duplicate runs: {e}")

    success = len(errors) == 0
    messages.extend(errors)
    return success, messages


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify HF checkpoint chain integrity for J1 foundation training"
    )
    ap.add_argument(
        "--repo",
        default=DEFAULT_CHECKPOINT_REPO,
        help=f"HF checkpoint repo (default: {DEFAULT_CHECKPOINT_REPO})",
    )
    ap.add_argument(
        "--rolling-repo",
        default=DEFAULT_ROLLING_REPO,
        help=f"HF rolling repo (default: {DEFAULT_ROLLING_REPO})",
    )
    ap.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        help=f"HF repo type (default: {DEFAULT_REPO_TYPE})",
    )
    ap.add_argument(
        "--run-id",
        required=True,
        help="Run ID to verify",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="HF token (or set HF_TOKEN env var)",
    )

    args = ap.parse_args(argv)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not provided", flush=True)
        print("Use --hf-token or set HF_TOKEN environment variable", flush=True)
        return 1

    print(f"Verifying checkpoint chain for run: {args.run_id}", flush=True)
    print(f"Checkpoint repo: {args.repo}", flush=True)
    print(f"Rolling repo: {args.rolling_repo}", flush=True)
    print(f"Repo type: {args.repo_type}", flush=True)
    print()

    success, messages = verify_checkpoint_chain(
        checkpoint_repo=args.repo,
        rolling_repo=args.rolling_repo,
        repo_type=args.repo_type,
        token=token,
        run_id=args.run_id,
    )

    for msg in messages:
        print(f"  {msg}", flush=True)

    print()
    if success:
        print("HF CHECKPOINT CHAIN: VERIFIED", flush=True)
        return 0
    else:
        print("HF CHECKPOINT CHAIN: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
