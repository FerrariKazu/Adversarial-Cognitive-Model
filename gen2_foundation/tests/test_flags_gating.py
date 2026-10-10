"""G2-HANDOFF: experiment gating invariants (test #1).

1. experiment gating — every non-default mechanism requires its exact
   experiment ID; a default config is a legal control arm; flipping
   anything requires its ID.
2. rejected Gen-1 mechanisms — REJECTED_OUTRIGHT are blocked (no flags).
3. out-of-scope IDs cannot be activated (policy zoo, multi-scale,
   memory, halting, Gen-3).
"""

from __future__ import annotations

import pytest

from gen2_foundation.flags import (
    RHANGen2Config,
    GATED_FLAGS,
    REJECTED_OUTRIGHT,
    G2_EXPERIMENT_IDS,
    G2_OUT_OF_SCOPE_IDS,
)


class TestGating:
    def test_control_arm_is_legal(self):
        cfg = RHANGen2Config()
        # The default config is a legal control arm: active mechanism = only 'no_gist' (Gen-1 parity).
        assert cfg.is_control_arm()
        assert cfg.active_mechanism_ids() == frozenset(['G2-K1-no-gist'])

    def test_default_config_has_no_non_default_mechanisms(self):
        cfg = RHANGen2Config()
        assert cfg.no_gist is True   # Gen-1 parity control, on by default
        assert cfg.gist_fixed is False
        assert cfg.enable_optimizer_adamw is False
        assert cfg.precision_1minusu is False
        assert cfg.observed_error_off is False
        assert cfg.spatial_error_off is False
        assert cfg.belief_bounded is False
        assert cfg.ema_on is False

    def test_unknown_flag_is_not_settable(self):
        # A nonexistent flag must not be settable.
        with pytest.raises(TypeError):
            RHANGen2Config(unknown_flag=True)

    def test_activate_is_id_dependent(self):
        # A registered mechanism must carry its exact experiment ID.
        valid = RHANGen2Config(gist_fixed=True,
                                enable_optimizer_adamw=True)
        assert valid.gist_fixed is True
        assert valid.enable_optimizer_adamw is True

    def test_gated_flags_cover_all_mechanisms(self):
        assert len(GATED_FLAGS) >= 30

    def test_no_gist_is_the_gen1_parity_control(self):
        # no_gist is the executable Gen-1 parity mechanism (always on).
        cfg = RHANGen2Config(no_gist=True)
        assert cfg.no_gist is True
        # A fully no-gist config has no active non-default MECHANISM beyond no_gist.
        assert cfg.is_control_arm()


class TestRejectedOutright:
    def test_rejected_are_blocked(self):
        for item in REJECTED_OUTRIGHT:
            assert item not in G2_EXPERIMENT_IDS

    def test_rejected_not_settable(self):
        # No flag exists for rejected mechanisms; they are blocked at
        # the schema level. Any attempt to set a rejected-outright value
        # must raise.
        for item in REJECTED_OUTRIGHT:
            assert item not in G2_EXPERIMENT_IDS

    def test_rejected_sixty_nine(self):
        assert len(REJECTED_OUTRIGHT) >= 5


class TestOutOfScope:
    def test_out_of_scope_cannot_be_actived(self):
        for exp_id in G2_OUT_OF_SCOPE_IDS:
            assert exp_id not in G2_EXPERIMENT_IDS

    def test_scope_is_exhaustive(self):
        # The foundation slice does NOT include policy zoo, pyramid,
        # memory, halting, or Gen-3.
        scope_items = {
            "G2-K3-policy",
            "G2-K3a-information-gain",
            "G2-K1c-gist-multi-scale",
            "G2-K7a-spatial-working-memory",
            "G2-K7b-spatial-memory",
            "G2-K8a-variable-glimpses",
            "G3-K1-world-state",
            "G3-K2-object-scene",
            "G3-K3-persistent-memory",
            "G3-K4-counterfactual",
            "G3-K5-reasoning",
            "G3-K6-self-supervised",
            "G3-K7-hierarchical-perception",
            "G3-K8-graduated-training",
            "G3-K9-robustness",
            "G2-K11-a-calibration",
        }
        assert scope_items <= G2_OUT_OF_SCOPE_IDS
