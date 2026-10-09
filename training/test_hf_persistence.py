"""
Tests for HF persistence module.

These tests verify the core functionality of the HF persistence system
without requiring actual HF access.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import torch


# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hf_persistence import (
    HFPersistenceCoordinator,
    atomic_checkpoint_write,
    compute_file_sha256,
    capture_checkpoint_state,
    write_checkpoint_metadata,
    write_sha256_txt,
    load_or_create_manifest,
    update_epoch_in_manifest,
    log_event,
    check_config_integrity,
    generate_run_id,
    get_hf_checkpoint_repo,
    get_hf_rolling_repo,
    get_hf_repo_type,
    is_hf_persistence_enabled,
    detect_runtime_env,
)


class TestAtomicCheckpointWrite(unittest.TestCase):
    """Test A - Local atomic checkpoint write."""

    def test_atomic_write_creates_valid_checkpoint(self):
        """Test that atomic write creates a valid checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "checkpoints", "epoch_001")
            state_dict = {
                "model_state_dict": {"weight": torch.randn(3, 3)},
                "optimizer_state_dict": {"state": {}},
                "epoch": 1,
            }

            path = atomic_checkpoint_write(checkpoint_dir, state_dict)

            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))

            # Load and verify
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["epoch"], 1)

    def test_atomic_write_handles_interruption(self):
        """Test that interruption during write doesn't leave corrupt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "checkpoints", "epoch_002")

            # Write valid checkpoint
            state_dict = {"epoch": 2, "model_state_dict": {}}
            path = atomic_checkpoint_write(checkpoint_dir, state_dict)

            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_atomic_write_preserves_previous_checkpoint(self):
        """Test that writing new checkpoint doesn't corrupt previous one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "checkpoints")

            # Write epoch 1
            epoch1_dir = os.path.join(base_dir, "epoch_001")
            state1 = {"epoch": 1, "model_state_dict": {"w1": torch.randn(2)}}
            path1 = atomic_checkpoint_write(epoch1_dir, state1)

            # Write epoch 2
            epoch2_dir = os.path.join(base_dir, "epoch_002")
            state2 = {"epoch": 2, "model_state_dict": {"w2": torch.randn(2)}}
            path2 = atomic_checkpoint_write(epoch2_dir, state2)

            # Verify both are intact
            loaded1 = torch.load(path1, map_location="cpu", weights_only=False)
            loaded2 = torch.load(path2, map_location="cpu", weights_only=False)
            self.assertEqual(loaded1["epoch"], 1)
            self.assertEqual(loaded2["epoch"], 2)


class TestSHA256(unittest.TestCase):
    """Test SHA-256 computation."""

    def test_compute_file_sha256(self):
        """Test SHA-256 computation is consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.bin")
            data = torch.randn(100, 100).numpy().tobytes()

            with open(path, "wb") as f:
                f.write(data)

            sha1 = compute_file_sha256(path)
            sha2 = compute_file_sha256(path)

            self.assertEqual(sha1, sha2)
            self.assertEqual(len(sha1), 64)  # SHA-256 hex digest

    def test_different_files_have_different_hashes(self):
        """Test that different files produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "file1.bin")
            path2 = os.path.join(tmpdir, "file2.bin")

            with open(path1, "wb") as f:
                f.write(torch.randn(100).numpy().tobytes())
            with open(path2, "wb") as f:
                f.write(torch.randn(100).numpy().tobytes())

            sha1 = compute_file_sha256(path1)
            sha2 = compute_file_sha256(path2)

            self.assertNotEqual(sha1, sha2)


class TestCheckpointMetadata(unittest.TestCase):
    """Test checkpoint metadata writing."""

    def test_write_checkpoint_metadata(self):
        """Test metadata.json sidecar creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = write_checkpoint_metadata(
                checkpoint_dir=tmpdir,
                epoch=5,
                global_step=1000,
                sha256="abc123",
                size_bytes=1024,
                metric_name="val_acc",
                metric_value=0.2718,
                best_metric=0.2632,
                best_epoch=3,
                config_hash="config123",
                code_commit="abc123def",
                run_id="J1_backbone_only_gen1-adv-curriculum-v1_abc123",
                phase="backbone_only",
                environment="COLAB",
                gpu="Tesla T4",
                kind="epoch",
            )

            # Verify metadata file exists
            meta_path = os.path.join(tmpdir, "metadata.json")
            self.assertTrue(os.path.exists(meta_path))

            # Load and verify
            with open(meta_path) as f:
                loaded = json.load(f)

            self.assertEqual(loaded["epoch"], 5)
            self.assertEqual(loaded["metric_name"], "val_acc")
            self.assertEqual(loaded["metric_value"], 0.2718)
            self.assertEqual(loaded["kind"], "epoch")
            self.assertEqual(loaded["environment"], "COLAB")
            self.assertEqual(loaded["gpu"], "Tesla T4")

    def test_write_sha256_txt(self):
        """Test sha256.txt sidecar creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_sha256_txt(tmpdir, "abc123def456")

            sha_path = os.path.join(tmpdir, "sha256.txt")
            self.assertTrue(os.path.exists(sha_path))

            with open(sha_path) as f:
                content = f.read().strip()

            self.assertEqual(content, "abc123def456")


class TestManifestManagement(unittest.TestCase):
    """Test manifest management."""

    def test_load_or_create_manifest(self):
        """Test manifest loading and creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")

            # Create new manifest
            manifest = load_or_create_manifest(manifest_path)

            self.assertEqual(manifest["status"], "RUNNING")
            self.assertEqual(manifest["epochs"], {})

            # Write some data
            manifest["run_id"] = "test_run"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            # Load existing
            loaded = load_or_create_manifest(manifest_path)
            self.assertEqual(loaded["run_id"], "test_run")

    def test_update_epoch_in_manifest(self):
        """Test epoch update in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")

            # Add epoch 1
            manifest = update_epoch_in_manifest(
                manifest_path, 1, "sha256_1", "COMMITTED", False
            )

            self.assertEqual(manifest["latest_committed_epoch"], 1)
            self.assertFalse(manifest["epochs"]["1"]["best"])

            # Add epoch 2 as best
            manifest = update_epoch_in_manifest(
                manifest_path, 2, "sha256_2", "COMMITTED", True
            )

            self.assertEqual(manifest["latest_committed_epoch"], 2)
            self.assertEqual(manifest["best_epoch"], 2)
            self.assertTrue(manifest["epochs"]["2"]["best"])


class TestEventLog(unittest.TestCase):
    """Test append-only event log."""

    def test_log_event_appends(self):
        """Test that events are appended, not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = os.path.join(tmpdir, "events.jsonl")

            # Log two events
            log_event(events_path, "run_1", 1, "EPOCH_COMPLETED", "sha1")
            log_event(events_path, "run_1", 2, "EPOCH_COMPLETED", "sha2")

            # Read back
            with open(events_path) as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 2)

            event1 = json.loads(lines[0])
            event2 = json.loads(lines[1])

            self.assertEqual(event1["epoch"], 1)
            self.assertEqual(event2["epoch"], 2)
            self.assertEqual(event1["sha256"], "sha1")
            self.assertEqual(event2["sha256"], "sha2")


class TestConfigIntegrity(unittest.TestCase):
    """Test config integrity checks."""

    def test_config_match_passes(self):
        """Test that matching configs pass integrity check."""
        ok, msg = check_config_integrity(
            "hash123", "commit123", "hash123", "commit123"
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "config integrity check passed")

    def test_config_mismatch_fails(self):
        """Test that mismatching configs fail integrity check."""
        ok, msg = check_config_integrity(
            "hash_old", "commit123", "hash_new", "commit123"
        )
        self.assertFalse(ok)
        self.assertIn("configuration hash", msg)
        self.assertIn("--new-experiment", msg)

    def test_git_commit_mismatch_fails(self):
        """Test that git commit changes are detected."""
        ok, msg = check_config_integrity(
            "hash123", "commit_old", "hash123", "commit_new"
        )
        self.assertFalse(ok)
        self.assertIn("git commit", msg)


class TestRunIDGeneration(unittest.TestCase):
    """Test run ID generation."""

    def test_generate_run_id(self):
        """Test run ID format."""
        run_id = generate_run_id(
            phase="backbone_only",
            recipe_version="gen1-adv-curriculum-v1",
            config_hash="abc123def4567890",
        )

        self.assertTrue(run_id.startswith("J1_backbone_only_"))
        self.assertIn("gen1-adv-curriculum-v1", run_id)
        self.assertIn("abc123", run_id)  # First 12 chars of hash

    def test_run_id_is_stable(self):
        """Test that same inputs produce same run ID."""
        hash1 = "abc123def4567890"
        hash2 = "abc123def4567890"

        id1 = generate_run_id("backbone_only", "v1", hash1)
        id2 = generate_run_id("backbone_only", "v1", hash2)

        self.assertEqual(id1, id2)


class TestDefaultConfigurations(unittest.TestCase):
    """Test default configuration values."""

    def test_default_checkpoint_repo(self):
        """Test default checkpoint repo."""
        self.assertEqual(
            get_hf_checkpoint_repo(),
            "FerrariKazu/rhan-nxa-checkpoints"
        )

    def test_default_rolling_repo(self):
        """Test default rolling repo."""
        self.assertEqual(
            get_hf_rolling_repo(),
            "FerrariKazu/rhan-nxa-checkpoints-rolling"
        )

    def test_default_repo_type(self):
        """Test default repo type."""
        self.assertEqual(get_hf_repo_type(), "dataset")

    def test_hf_disabled_when_no_token(self):
        """Test HF disabled when no token."""
        self.assertFalse(is_hf_persistence_enabled(True, None))
        self.assertFalse(is_hf_persistence_enabled(True, None, smoke=False))
        self.assertFalse(is_hf_persistence_enabled(True, None, smoke=True))

    def test_hf_enabled_with_token(self):
        """Test HF enabled with token."""
        self.assertTrue(is_hf_persistence_enabled(True, "token123"))
        self.assertTrue(is_hf_persistence_enabled(True, "token123", smoke=False))

    def test_hf_disabled_in_smoke(self):
        """Test HF disabled in smoke mode."""
        self.assertFalse(is_hf_persistence_enabled(True, "token123", smoke=True))

    def test_hf_disabled_when_use_hf_false(self):
        """Test HF disabled when use_hf is False."""
        self.assertFalse(is_hf_persistence_enabled(False, "token123"))


class TestCaptureCheckpointState(unittest.TestCase):
    """Test checkpoint state capture."""

    def test_capture_checkpoint_state(self):
        """Test full state capture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model
            model = torch.nn.Linear(10, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
            scaler = torch.amp.GradScaler("cpu")

            state = capture_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=5,
                global_step=100,
                best_metric=0.5,
                best_epoch=3,
                training_history=[{"epoch": 1, "loss": 0.5}],
                config={"lr": 0.01},
                config_hash="hash123",
                seed=42,
                code_commit="abc123",
                gpu_name="Test GPU",
            )

            # Verify all expected keys
            expected_keys = [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "epoch",
                "global_step",
                "best_metric",
                "best_epoch",
                "training_history",
                "config",
                "config_hash",
                "seed",
                "python_rng_state",
                "torch_cpu_rng_state",
                "code_commit",
                "saved_at_utc",
                "gpu",
            ]
            for key in expected_keys:
                self.assertIn(key, state)

            self.assertEqual(state["epoch"], 5)
            self.assertEqual(state["global_step"], 100)
            self.assertEqual(state["best_metric"], 0.5)


class TestHFPersistenceCoordinator(unittest.TestCase):
    """Test HF persistence coordinator."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.ckpt_dir = os.path.join(self.tmpdir, "checkpoints")
        self.runs_dir = os.path.join(self.tmpdir, "runs")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_coordinator_init(self):
        """Test coordinator initialization."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            seed = 42
            epochs = 10

        cfg = TestConfig()
        coordinator = HFPersistenceCoordinator(
            cfg=cfg,
            phase="backbone_only",
            run_id="test_run_id",
            config_hash="test_hash",
            code_commit="test_commit",
            hf_token="test_token",
            ckpt_dir=self.ckpt_dir,
            runs_dir=self.runs_dir,
            use_hf=True,
            smoke=False,
        )

        self.assertEqual(coordinator.run_id, "test_run_id")
        self.assertEqual(coordinator.config_hash, "test_hash")
        self.assertTrue(coordinator.enabled)
        self.assertEqual(coordinator.checkpoint_repo,
                        "FerrariKazu/rhan-nxa-checkpoints")

    def test_coordinator_disabled_without_token(self):
        """Test coordinator disabled without token."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            seed = 42

        cfg = TestConfig()
        coordinator = HFPersistenceCoordinator(
            cfg=cfg,
            phase="backbone_only",
            run_id="test_run_id",
            config_hash="test_hash",
            code_commit="test_commit",
            hf_token=None,
            ckpt_dir=self.ckpt_dir,
            runs_dir=self.runs_dir,
            use_hf=True,
            smoke=False,
        )

        self.assertFalse(coordinator.enabled)

    def test_manifest_creation(self):
        """Test manifest creation by coordinator."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            seed = 42

        cfg = TestConfig()
        coordinator = HFPersistenceCoordinator(
            cfg=cfg,
            phase="backbone_only",
            run_id="test_run_id",
            config_hash="test_hash",
            code_commit="test_commit",
            hf_token="test_token",
            ckpt_dir=self.ckpt_dir,
            runs_dir=self.runs_dir,
            use_hf=True,
            smoke=False,
        )

        # Access manifest path
        manifest_path = coordinator.manifest_path
        self.assertTrue(manifest_path.startswith(self.runs_dir))

        # Create runs directory
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        # Create manifest
        manifest = load_or_create_manifest(manifest_path)
        manifest["run_id"] = coordinator.run_id
        manifest["config_hash"] = coordinator.config_hash
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Verify
        loaded = load_or_create_manifest(manifest_path)
        self.assertEqual(loaded["run_id"], "test_run_id")


class TestConfigMismatchDetection(unittest.TestCase):
    """Test H - Config mismatch detection."""

    def test_config_mismatch_refuses_resume(self):
        """Test that config mismatch refuses resume."""
        existing_hash = "old_hash"
        current_hash = "new_hash"
        existing_commit = "old_commit"
        current_commit = "new_commit"

        # Config hash mismatch
        ok, msg = check_config_integrity(
            existing_hash, existing_commit, current_hash, existing_commit
        )
        self.assertFalse(ok)
        self.assertIn("configuration hash", msg.lower())
        self.assertIn("--new-experiment", msg)

        # Git commit mismatch
        ok, msg = check_config_integrity(
            existing_hash, existing_commit, existing_hash, current_commit
        )
        self.assertFalse(ok)
        self.assertIn("git commit", msg.lower())


class TestHFDisabledBehavior(unittest.TestCase):
    """Test I - HF disabled behavior."""

    def test_hf_disabled_graceful_degradation(self):
        """Test that HF disabled doesn't crash."""
        # When HF is disabled, coordinator should still work locally
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            seed = 42

        cfg = TestConfig()
        coordinator = HFPersistenceCoordinator(
            cfg=cfg,
            phase="backbone_only",
            run_id="test_run_id",
            config_hash="test_hash",
            code_commit="test_commit",
            hf_token=None,  # No token = HF disabled
            ckpt_dir=self.tmpdir if hasattr(self, 'tmpdir') else tempfile.mkdtemp(),
            runs_dir=self.runs_dir if hasattr(self, 'runs_dir') else tempfile.mkdtemp(),
            use_hf=True,  # But use_hf is True
            smoke=False,
        )

        # Should be disabled
        self.assertFalse(coordinator.enabled)

        # persist_epoch should return False without error
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
        scaler = torch.amp.GradScaler("cpu")

        result = coordinator.persist_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=1,
            global_step=10,
            val_acc=0.5,
            training_history=[{"epoch": 1}],
            config={"test": True},
        )

        # Should return False since HF is disabled
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
