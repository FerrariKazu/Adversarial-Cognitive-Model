"""
Agent 0 contract tests — schema versioning and gated-flag enforcement.

Per the Agent 0 contract (MASTER_PLAN):
  - the config round-trips through serialization;
  - every PENDING DECISION / DEFERRED / built-last flag defaults to its
    documented safe value (False / None);
  - attempting to set a PENDING-DECISION-gated flag True WITHOUT its
    documented prerequisite raises — never silently succeeds;
  - LOCKED ranges are enforced (schema drift is an error, not a
    setting).
"""
import pytest

from noesis_vision.core.schema import (
    AIS_T0_SCORING,
    FIRST_GLIMPSE_CONVENTION,
    REJECTED_OUTRIGHT,
    SCHEMA_VERSION,
    RHANNXAConfig,
)

# PENDING DECISION / DEFERRED / built-last fields: safe defaults.
PENDING_SAFE_DEFAULTS = {
    "representation_level_uncertainty": False,  # PENDING DECISION (1.A)
    "enable_l_stab_objective": False,           # PENDING DECISION, gated (1.G)
    "enable_episodic_memory": False,            # DEFERRED, gated (1.H)
    "enable_v1_frontend": False,                # EXPERIMENTAL CANDIDATE, built LAST (1.I)
    "adaptive_halting": False,                  # DEFERRED, gated (1.C)
    "enable_sbr": False,                        # LOCKED default (1.D), gated if ever True
    "sbr_num_slots": None,                      # None while S_t is None (1.D)
    "step6_validated_result": None,             # prerequisite record (Part 2/4)
    "temporal_experiment_evidence": None,       # prerequisite record (1.H)
}


def test_schema_version_present_and_round_trips():
    cfg = RHANNXAConfig()
    assert cfg.schema_version == SCHEMA_VERSION
    restored = RHANNXAConfig.from_json(cfg.to_json())
    assert restored == cfg
    assert RHANNXAConfig.from_dict(cfg.to_dict()) == cfg


def test_pending_flags_default_to_safe_values():
    cfg = RHANNXAConfig()
    for name, expected in PENDING_SAFE_DEFAULTS.items():
        assert getattr(cfg, name) == expected, (
            f"{name} must default to its documented safe value {expected!r}"
        )


def test_locked_core_defaults():
    cfg = RHANNXAConfig()
    assert cfg.enable_recurrence is True          # full loop default (1.C)
    assert cfg.within_glimpse_shared_weights is True  # tied recurrence (1.C)
    assert cfg.num_glimpses == 4                  # T=4 LOCKED (1.C)
    assert cfg.enable_evidential_uncertainty is True  # REQUIRED port (1.F)
    assert cfg.enable_ais_v2 is True              # REQUIRED mechanism (1.E)
    assert cfg.error_target == "latent_next_glimpse"  # 1.B design default
    assert cfg.l_stab_diagnostic_only is True     # staged phase A (1.G)


def test_gated_flag_raises_without_prerequisite():
    with pytest.raises(ValueError, match="step6_validated_result"):
        RHANNXAConfig(enable_sbr=True)
    with pytest.raises(ValueError, match="step6_validated_result"):
        RHANNXAConfig(enable_l_stab_objective=True)
    with pytest.raises(ValueError, match="step6_validated_result"):
        RHANNXAConfig(adaptive_halting=True)
    with pytest.raises(ValueError, match="temporal_experiment_evidence"):
        RHANNXAConfig(enable_episodic_memory=True)


def test_gated_flag_allowed_with_prerequisite_recorded():
    cfg = RHANNXAConfig(
        enable_sbr=True,
        sbr_num_slots=4,  # 2-4 slots ONLY if revisited (1.D)
        step6_validated_result="J-confirmed, run <reference>",
    )
    assert cfg.enable_sbr is True


def test_locked_ranges_enforced():
    with pytest.raises(ValueError, match="4-8"):
        RHANNXAConfig(num_candidates=16)  # K LOCKED 4-8 (1.E)
    with pytest.raises(ValueError, match="2-3"):
        RHANNXAConfig(within_glimpse_iters=5)  # LOCKED 2-3 (1.C)
    with pytest.raises(ValueError, match="T=4"):
        RHANNXAConfig(num_glimpses=7)  # T=4 LOCKED for the Gen-1 core (1.C)
    with pytest.raises(ValueError, match="2-4"):
        RHANNXAConfig(sbr_num_slots=16)  # 16-slot REJECTED (1.D)
    with pytest.raises(ValueError, match="latent_next_glimpse"):
        RHANNXAConfig(error_target="pixel")  # pixel target REJECTED (1.B)


def test_sbr_requires_explicit_slot_count():
    with pytest.raises(ValueError, match="sbr_num_slots"):
        RHANNXAConfig(
            enable_sbr=True, step6_validated_result="recorded"
        )  # S_t re-entry is never implicit


def test_unknown_keys_rejected_as_drift():
    with pytest.raises(ValueError, match="schema drift"):
        RHANNXAConfig.from_dict(
            {**RHANNXAConfig().to_dict(), "gaze_epsilon": 1e-3}
        )


def test_locked_boundary_conditions_documented():
    # The t=0 conventions are constants so tests assert against drift.
    assert "E_0 := 0" in FIRST_GLIMPSE_CONVENTION
    assert "t=1" in FIRST_GLIMPSE_CONVENTION
    assert "heuristic-saliency" in AIS_T0_SCORING
    assert len(REJECTED_OUTRIGHT) >= 5  # 1.B / 1.D / 1.H / Part 5 rejections
