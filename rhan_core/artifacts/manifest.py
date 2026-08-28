"""
manifest.py — ExperimentManifest: experiment state machine, seed-level
tracking, config hash verification.

The manifest is the single source of truth for an experiment's state.
It must survive runtime resets via HF sync and contain enough information
for a fresh runtime to reconstruct the experiment state.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class StageStatus(str, Enum):
    """Status of a stage/seed within the experiment."""
    NEW = "NEW"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def is_terminal(self) -> bool:
        return self in (StageStatus.COMPLETED, StageStatus.FAILED)

    def is_resumable(self) -> bool:
        return self in (StageStatus.RUNNING, StageStatus.INTERRUPTED)


def config_hash(config_dict: Dict[str, Any]) -> str:
    """Deterministic hash of an experiment config for contamination detection."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class SeedTracker:
    """Track per-seed status for a single phase (training, evaluation, lens)."""

    def __init__(self):
        self.seeds: Dict[str, Dict[str, Any]] = {}

    def set_status(self, seed: int, status: StageStatus, **kwargs):
        key = str(seed)
        if key not in self.seeds:
            self.seeds[key] = {}
        self.seeds[key]["status"] = status.value
        self.seeds[key]["updated_at"] = time.time()
        for k, v in kwargs.items():
            self.seeds[key][k] = v

    def get_status(self, seed: int) -> StageStatus:
        key = str(seed)
        if key not in self.seeds:
            return StageStatus.NEW
        return StageStatus(self.seeds[key].get("status", "NEW"))

    def completed_seeds(self) -> Set[int]:
        return {int(k) for k, v in self.seeds.items()
                if v.get("status") == StageStatus.COMPLETED.value}

    def next_pending_seed(self, all_seeds: List[int]) -> Optional[int]:
        """Find the next seed that needs work."""
        for s in all_seeds:
            if self.get_status(s).is_resumable() or self.get_status(s) == StageStatus.NEW:
                return s
        return None

    def summary(self, all_seeds: List[int]) -> Dict[str, str]:
        """Status summary for display."""
        result = {}
        for s in all_seeds:
            status = self.get_status(s)
            result[str(s)] = status.value
        return result

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.seeds)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SeedTracker:
        tracker = cls()
        tracker.seeds = d
        return tracker


class ExperimentManifest:
    """Single authoritative state file for a Stage 4 experiment.

    Structure:
    {
        "stage": "4-E1",
        "status": "RUNNING",
        "config_hash": "...",
        "git_commit": "...",
        "base_checkpoint_sha256": "...",
        "created_at": "...",
        "updated_at": "...",

        "seeds": {
            "41": {
                "training": {"status": "COMPLETED", "epoch": 60, "sha256": "..."},
                "evaluation": {"status": "COMPLETED"},
                "lens": {"status": "PENDING"}
            },
            ...
        },

        "smoke": {"status": "COMPLETED", ...},
        "evaluation": {"status": "PENDING"},
        "lens": {"status": "PENDING"},
        "report": {"status": "PENDING"},

        "checkpoints": [...]
    }
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.stage: str = "4-E1"
        self.status: StageStatus = StageStatus.NEW
        self.config_hash: str = ""
        self.git_commit: str = ""
        self.base_checkpoint_sha256: str = ""
        self.created_at: str = ""
        self.updated_at: str = ""

        # Seed-level tracking
        self.seed_trackers: Dict[str, SeedTracker] = {}

        # Stage-level status
        self.smoke_status: StageStatus = StageStatus.NEW
        self.eval_status: StageStatus = StageStatus.NEW
        self.lens_status: StageStatus = StageStatus.NEW
        self.report_status: StageStatus = StageStatus.NEW

        # Checkpoint registry
        self.checkpoints: List[Dict[str, Any]] = []

    def set_config(self, config_dict: Dict[str, Any], git_commit: str,
                   base_checkpoint_sha256: str = ""):
        """Set experiment configuration (must match on resume)."""
        self.config_hash = config_hash(config_dict)
        self.git_commit = git_commit
        self.base_checkpoint_sha256 = base_checkpoint_sha256
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.updated_at = self.created_at

    def verify_config(self, config_dict: Dict[str, Any], git_commit: str) -> bool:
        """Verify that the current config matches the manifest's config.

        Returns True if compatible (resume OK), False if contaminated.
        """
        current_hash = config_hash(config_dict)
        if self.config_hash and current_hash != self.config_hash:
            return False
        # Git commit mismatch is a warning, not a block
        return True

    def set_seed_status(self, phase: str, seed: int, status: StageStatus, **kwargs):
        """Set the status of a specific seed in a phase."""
        if phase not in self.seed_trackers:
            self.seed_trackers[phase] = SeedTracker()
        self.seed_trackers[phase].set_status(seed, status, **kwargs)
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def get_seed_status(self, phase: str, seed: int) -> StageStatus:
        if phase not in self.seed_trackers:
            return StageStatus.NEW
        return self.seed_trackers[phase].get_status(seed)

    def completed_seeds(self, phase: str) -> Set[int]:
        if phase not in self.seed_trackers:
            return set()
        return self.seed_trackers[phase].completed_seeds()

    def next_pending_seed(self, phase: str, all_seeds: List[int]) -> Optional[int]:
        if phase not in self.seed_trackers:
            self.seed_trackers[phase] = SeedTracker()
        return self.seed_trackers[phase].next_pending_seed(all_seeds)

    def record_checkpoint(self, phase: str, record: Dict[str, Any]):
        """Record a checkpoint in the registry."""
        self.checkpoints.append({
            "phase": phase,
            **record,
        })
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def is_stage_complete(self, phase: str) -> bool:
        """Check if a phase is marked COMPLETED."""
        if phase == "smoke":
            return self.smoke_status == StageStatus.COMPLETED
        elif phase == "evaluation":
            return self.eval_status == StageStatus.COMPLETED
        elif phase == "lens":
            return self.lens_status == StageStatus.COMPLETED
        elif phase == "report":
            return self.report_status == StageStatus.COMPLETED
        return False

    def set_stage_status(self, phase: str, status: StageStatus, **kwargs):
        """Set the status of a stage."""
        if phase == "smoke":
            self.smoke_status = status
        elif phase == "evaluation":
            self.eval_status = status
        elif phase == "lens":
            self.lens_status = status
        elif phase == "report":
            self.report_status = status
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "stage": self.stage,
            "status": self.status.value,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "smoke": {"status": self.smoke_status.value},
            "evaluation": {"status": self.eval_status.value},
            "lens": {"status": self.lens_status.value},
            "report": {"status": self.report_status.value},
            "seeds": {},
            "checkpoints": self.checkpoints,
        }
        for phase, tracker in self.seed_trackers.items():
            d["seeds"][phase] = tracker.to_dict()
        return d

    def save(self):
        """Save manifest atomically."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2, default=str).encode()

        import tempfile
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), suffix=".tmp")
        try:
            os.write(fd, content)
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp, self.path)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> ExperimentManifest:
        """Load or create a manifest."""
        manifest = cls(path)
        if not os.path.exists(path):
            manifest.created_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            manifest.updated_at = manifest.created_at
            return manifest

        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest.created_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            manifest.updated_at = manifest.created_at
            return manifest

        manifest.stage = d.get("stage", "4-E1")
        manifest.status = StageStatus(d.get("status", "NEW"))
        manifest.config_hash = d.get("config_hash", "")
        manifest.git_commit = d.get("git_commit", "")
        manifest.base_checkpoint_sha256 = d.get("base_checkpoint_sha256", "")
        manifest.created_at = d.get("created_at", "")
        manifest.updated_at = d.get("updated_at", "")
        manifest.checkpoints = d.get("checkpoints", [])

        # Stage statuses
        for key, attr in [("smoke", "smoke_status"), ("evaluation", "eval_status"),
                          ("lens", "lens_status"), ("report", "report_status")]:
            stage_d = d.get(key, {})
            setattr(manifest, attr, StageStatus(stage_d.get("status", "NEW")))

        # Seed trackers
        seeds_d = d.get("seeds", {})
        for phase, phase_seeds in seeds_d.items():
            manifest.seed_trackers[phase] = SeedTracker.from_dict(phase_seeds)

        return manifest

    # ── Display ──────────────────────────────────────────────────────────────

    def print_status(self, all_seeds: List[int]):
        """Print a human-readable status summary."""
        print(f"\n{'═' * 60}")
        print(f"  Stage {self.stage} — Experiment Status")
        print(f"{'═' * 60}")
        print(f"  Status:     {self.status.value}")
        print(f"  Config:     {self.config_hash[:12]}...")
        print(f"  Git:        {self.git_commit[:12]}...")
        print(f"  Created:    {self.created_at}")
        print(f"  Updated:    {self.updated_at}")

        print(f"\n  Stages:")
        print(f"    Smoke:        {self.smoke_status.value}")
        print(f"    Evaluation:   {self.eval_status.value}")
        print(f"    Lens:         {self.lens_status.value}")
        print(f"    Report:       {self.report_status.value}")

        if any(self.seed_trackers):
            print(f"\n  Seeds:")
            for phase in ["training", "evaluation", "lens"]:
                if phase in self.seed_trackers:
                    summary = self.seed_trackers[phase].summary(all_seeds)
                    completed = len(self.seed_trackers[phase].completed_seeds())
                    print(f"    {phase}: {completed}/{len(all_seeds)} completed")
                    for s in all_seeds:
                        status = summary.get(str(s), "NEW")
                        marker = "✓" if status == "COMPLETED" else \
                                 "→" if status in ("RUNNING", "INTERRUPTED") else \
                                 "○" if status == "PENDING" else "·"
                        print(f"      seed {s:>2}: {marker} {status}")

        print(f"{'═' * 60}")
