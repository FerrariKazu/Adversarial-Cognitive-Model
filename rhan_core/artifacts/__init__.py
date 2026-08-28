"""
rhan_core.artifacts — Fault-tolerant experiment artifact management for Stage 4.

Provides atomic checkpoint writes, SHA-256 verification, HuggingFace-backed
persistence, experiment state machine, seed-level tracking, and append-only
event logging.

Usage:
    from rhan_core.artifacts.store import ArtifactStore
    from rhan_core.artifacts.manifest import ExperimentManifest, StageStatus
    from rhan_core.artifacts.events import EventLog

    store = ArtifactStore(experiment_root="stage4_artifacts/E1")
    manifest = store.load_manifest()

    # Atomic checkpoint write + HF sync
    store.save_checkpoint("train", seed=41, epoch=60, state_dict=model_state,
                          metadata={"val_acc": 54.2})

    # Seed-level tracking
    manifest.set_seed_status("training", 41, "COMPLETED")
    manifest.save()

    # Event log
    store.events.log("SEED_COMPLETED", seed=41, epoch=60)
"""

from rhan_core.artifacts.store import ArtifactStore
from rhan_core.artifacts.manifest import ExperimentManifest, StageStatus
from rhan_core.artifacts.events import EventLog

__all__ = [
    "ArtifactStore",
    "ExperimentManifest",
    "StageStatus",
    "EventLog",
]
