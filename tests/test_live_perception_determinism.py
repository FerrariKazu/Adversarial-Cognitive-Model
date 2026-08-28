"""
test_live_perception_determinism.py — verify that the live perception callback
produces identical trajectories to normal inference.

The live mode must NOT modify:
  - model weights
  - AIS equations
  - HPC computation
  - gaze policy
  - halting policy
  - random seeds
  - inference configuration

Run: python -m pytest tests/test_live_perception_determinism.py -v
"""
import sys
import os

import pytest
import torch

# Ensure repo root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext


def _make_model(config=None):
    """Create a model in eval mode with fixed seed."""
    torch.manual_seed(42)
    cfg = config or RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True,
        ais_precision_recon_enabled=False,
        enable_hpc=True, hpc_num_levels=1, hpc_error_weight=0.10)
    model = RHANNext(config=cfg)
    model.eval()
    return model


def _get_trajectory(model, x):
    """Run model with return_trajectory=True, no callback."""
    with torch.no_grad():
        logits, traj = model(x, return_trajectory=True)
    return logits, traj


def _get_callback_snapshots(model, x):
    """Run model with _step_callback, collect snapshots."""
    snapshots = []

    def _capture(data):
        snapshots.append(data)

    with torch.no_grad():
        out = model(x, return_trajectory=True, _step_callback=_capture)
    # out is (logits, trajectory) when return_trajectory=True
    logits = out[0] if isinstance(out, tuple) else out
    return logits, snapshots


class TestLivePerceptionDeterminism:
    """Verify live mode == normal inference."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model = _make_model()
        torch.manual_seed(99)
        self.x = torch.randn(1, 3, 96, 96)

    def test_callback_receives_correct_number_of_steps(self):
        """Callback should fire max_steps times."""
        _, snapshots = _get_callback_snapshots(self.model, self.x)
        assert len(snapshots) == self.model.max_steps

    def test_callback_step_indices_are_sequential(self):
        """Callback steps should be 0, 1, 2, ... max_steps-1."""
        _, snapshots = _get_callback_snapshots(self.model, self.x)
        for i, snap in enumerate(snapshots):
            assert snap["step"] == i

    def test_callback_contains_all_expected_keys(self):
        """Each snapshot should have all the expected state keys."""
        _, snapshots = _get_callback_snapshots(self.model, self.x)
        required_keys = {
            "step", "max_steps", "gaze_x_norm", "gaze_y_norm",
            "foveal_crop", "predicted_crop", "pi_d", "error_mag",
            "uncertainty", "gate_alpha", "recon_error", "step_belief",
            "continuation", "halted",
        }
        for snap in snapshots:
            missing = required_keys - set(snap.keys())
            assert not missing, f"Step {snap['step']}: missing keys {missing}"

    def test_callback_hpc_keys_when_hpc_active(self):
        """When HPC is active, snapshots should have HPC keys."""
        _, snapshots = _get_callback_snapshots(self.model, self.x)
        for snap in snapshots:
            assert "hpc_error" in snap
            assert "hpc_error_map" in snap
            assert "hpc_prediction" in snap
            assert snap["hpc_error"] is not None

    def test_gaze_trajectory_matches(self):
        """Gaze positions from callback should match trajectory actions."""
        traj_logits, traj = _get_trajectory(self.model, self.x)
        cb_logits, snapshots = _get_callback_snapshots(self.model, self.x)

        # Same final prediction
        assert torch.allclose(traj_logits, cb_logits, atol=1e-6)

        # Same gaze trajectory
        traj_actions = traj["actions"]  # list of (1, 2) tensors
        for t, snap in enumerate(snapshots):
            traj_gx = float(traj_actions[t][0, 0])
            traj_gy = float(traj_actions[t][0, 1])
            assert abs(snap["gaze_x_norm"] - traj_gx) < 1e-5, \
                f"Step {t}: gaze_x mismatch"
            assert abs(snap["gaze_y_norm"] - traj_gy) < 1e-5, \
                f"Step {t}: gaze_y mismatch"

    def test_pi_d_trajectory_matches(self):
        """Precision trajectory from callback should match trajectory."""
        traj_logits, traj = _get_trajectory(self.model, self.x)
        cb_logits, snapshots = _get_callback_snapshots(self.model, self.x)

        for t, snap in enumerate(snapshots):
            traj_pi = float(traj["precisions"][t][0])
            cb_pi = snap["pi_d"]
            assert abs(cb_pi - traj_pi) < 1e-5, \
                f"Step {t}: pi_d mismatch ({cb_pi} vs {traj_pi})"

    def test_uncertainty_trajectory_matches(self):
        """Uncertainty trajectory from callback should match."""
        traj_logits, traj = _get_trajectory(self.model, self.x)
        cb_logits, snapshots = _get_callback_snapshots(self.model, self.x)

        for t, snap in enumerate(snapshots):
            traj_u = float(traj["uncertainties"][t][0])
            cb_u = snap["uncertainty"]
            assert abs(cb_u - traj_u) < 1e-5, \
                f"Step {t}: uncertainty mismatch"

    def test_hpc_error_trajectory_matches(self):
        """HPC error trajectory from callback should match."""
        traj_logits, traj = _get_trajectory(self.model, self.x)
        cb_logits, snapshots = _get_callback_snapshots(self.model, self.x)

        for t, snap in enumerate(snapshots):
            traj_hpc = float(traj["hpc_errors"][t][0])
            cb_hpc = snap["hpc_error"]
            assert abs(cb_hpc - traj_hpc) < 1e-5, \
                f"Step {t}: hpc_error mismatch"

    def test_belief_tensor_matches(self):
        """Step belief from callback should match trajectory."""
        traj_logits, traj = _get_trajectory(self.model, self.x)
        cb_logits, snapshots = _get_callback_snapshots(self.model, self.x)

        for t, snap in enumerate(snapshots):
            traj_belief = traj["step_beliefs"][t][0]
            cb_belief = snap["step_belief"]
            assert torch.allclose(cb_belief, traj_belief, atol=1e-5), \
                f"Step {t}: belief tensor mismatch"

    def test_final_logits_identical(self):
        """Final classification logits must be identical."""
        traj_logits, _ = _get_trajectory(self.model, self.x)
        cb_logits, _ = _get_callback_snapshots(self.model, self.x)
        assert torch.allclose(traj_logits, cb_logits, atol=1e-6)

    def test_callback_does_not_modify_model(self):
        """Running with callback must not change model weights."""
        weights_before = {
            n: p.clone() for n, p in self.model.named_parameters()
        }
        _get_callback_snapshots(self.model, self.x)
        for name, param in self.model.named_parameters():
            assert torch.equal(param, weights_before[name]), \
                f"Weight {name} was modified by callback"

    def test_multiple_runs_are_deterministic(self):
        """Running the same image twice should produce identical results."""
        torch.manual_seed(99)
        x = torch.randn(1, 3, 96, 96)

        _, snaps1 = _get_callback_snapshots(self.model, x)
        _, snaps2 = _get_callback_snapshots(self.model, x)

        assert len(snaps1) == len(snaps2)
        for t in range(len(snaps1)):
            for key in ["gaze_x_norm", "gaze_y_norm", "pi_d", "uncertainty",
                         "continuation", "halted", "hpc_error"]:
                v1 = snaps1[t][key]
                v2 = snaps2[t][key]
                if isinstance(v1, float):
                    assert abs(v1 - v2) < 1e-6, \
                        f"Step {t}, {key}: non-deterministic ({v1} vs {v2})"
                elif isinstance(v1, bool):
                    assert v1 == v2, \
                        f"Step {t}, {key}: non-deterministic"
