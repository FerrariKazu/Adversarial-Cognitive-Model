"""
ABLATION_MATRIX — the A/B/C/D registry for the RHANNext validation campaign.
================================================================================

One declarative entry per configuration family. `status` drives what the
tooling is allowed to do with the entry:

    VALIDATED            -> already trained + evaluated (numbers in the roadmap)
    PENDING              -> to be trained (checkpoint becomes present after)
    SCAFFOLDED_NOT_RUN   -> code-complete + tested, but deliberately NOT run
                            (dormant, exactly like Pillars 3/4)

THIS ROUND: only C_hpc_only gets trained. A and B already have validated
checkpoints. D stays SCAFFOLDED_NOT_RUN — the entry resolves and is tested
(tests/test_ablation_matrix.py), but no training job launches for it.

Entry fields:
    label       -> ckpt-label used in eval result tables / provenance
                   (== the trainer's --ckpt-name for the trainable entries)
    config      -> RHANNextConfig, or None for A_baseline (static TRADES —
                   NOT a RHANNext model at all; eval arch "large")
    checkpoint  -> expected checkpoint path (repo-relative); None = not yet
    arch        -> eval arch: "large" (A) or "next" (B/C/D)
    status      -> VALIDATED / PENDING / SCAFFOLDED_NOT_RUN
    note        -> honest one-line record of where each entry stands

Variant mapping note (B_ais_only): the Stage 1 spec calls this config
"ais_variant=halting_only" — there is no `ais_variant` field on
RHANNextConfig; the variant is expressed with the concrete ablation toggles
(ais_halt_enabled=True, ais_precision_recon_enabled=False), which is exactly
what the Stage 1 notebook launched with (--no-ais-precision-recon).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rhan_core.config.pillar_config import RHANNextConfig

#: w_hpc used by the HPC entries — the separate HPC loss-weight slot (default).
W_HPC = 0.10

#: Status values in the order the tooling may act on them (gate semantics).
VALIDATED = "VALIDATED"
PENDING = "PENDING"
SCAFFOLDED_NOT_RUN = "SCAFFOLDED_NOT_RUN"

#: Statuses a training job may launch for (THIS ROUND: only C).
TRAINABLE_STATUSES = (PENDING,)

#: Statuses the eval registry iterates WITHOUT a checkpoint-present check.
ALWAYS_EVAL_STATUSES = (VALIDATED,)


def _hpc_config(*, ais: bool) -> RHANNextConfig:
    """HPC entries (C / D) share the HPC-only knobs; D adds AIS-v1."""
    kwargs = dict(enable_hpc=True, hpc_num_levels=1,
                  hpc_error_weight=W_HPC)
    if ais:
        kwargs.update(enable_ais=True, ais_halt_enabled=True,
                      ais_precision_recon_enabled=False)  # AIS-v1 halting-only
    return RHANNextConfig(**kwargs)


def _e1_config() -> RHANNextConfig:
    """E1 = AIS-v1 + HPC + recon-mod ON. Only difference from D is
    ais_precision_recon_enabled=True (D has it False)."""
    return RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True,
        ais_precision_recon_enabled=True,   # ← THE CHANGE from D
        enable_hpc=True, hpc_num_levels=1,
        hpc_error_weight=W_HPC)


def _e3_config() -> RHANNextConfig:
    """E3 = AIS-v1 + HPC + T=6 foraging steps. Only difference from D is
    max_foraging_steps=6 (D has it 4)."""
    return RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True,
        ais_precision_recon_enabled=False,
        enable_hpc=True, hpc_num_levels=1,
        hpc_error_weight=W_HPC,
        max_foraging_steps=6)


def _e2_config() -> RHANNextConfig:
    """E2 = AIS-v1 + HPC + SBR. Only difference from D is enable_sbr=True."""
    return RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True,
        ais_precision_recon_enabled=False,
        enable_hpc=True, hpc_num_levels=1,
        hpc_error_weight=W_HPC,
        enable_sbr=True)


def base_checkpoint_key(key: str) -> str:
    """Return the matrix key of the base checkpoint to resume from, or None.

    Stage 4-E1 resumes from D's checkpoint with recon-mod re-enabled.
    Stage 4-E2 resumes from D's checkpoint with SBR enabled.
    Stage 4-E3 resumes from D's checkpoint with T=6.
    """
    return {
        "E1_ais_hpc_recon": "D_ais_plus_hpc",
        "E2_ais_hpc_sbr": "D_ais_plus_hpc",
        "E3_ais_hpc_t6": "D_ais_plus_hpc",
    }.get(key)


ABLATION_MATRIX: Dict[str, Dict[str, Any]] = {
    "A_baseline": {
        "label": "trades_large_baseline",
        "config": None,  # static TRADES, not RHANNext at all (arch "large")
        "checkpoint": "checkpoints/rhan_stl10_large_pseudolabel_best.pth",
        "arch": "large",
        "status": VALIDATED,  # already trained + evaluated (Stage 1)
        "note": "Static TRADES-Large baseline; Stage 1 reference (23.71 "
                "± 3.47 @ eps=0.094, 8 seeds).",
    },
    "B_ais_only": {
        "label": "rhan_next_ais_v1_halting_only",
        "config": RHANNextConfig(enable_ais=True, ais_halt_enabled=True,
                                 ais_precision_recon_enabled=False),
        "checkpoint": "checkpoints/rhan_next_ais_v1_halting_only_best.pth",
        "arch": "next",
        "status": VALIDATED,  # Stage 1 result: +8.5pp @ eps=0.094, NOT significant
        "note": "AIS-v1 (halting-only variant); +8.5pp vs A @ eps=0.094, "
                "8 seeds, positive but NOT significant (2-sigma bar 8.84).",
    },
    "C_hpc_only": {
        "label": "rhan_next_hpc_only",
        "config": _hpc_config(ais=False),
        "checkpoint": "checkpoints/rhan_next_hpc_only_best.pth",
        "arch": "next",
        "status": VALIDATED,
        "note": "HPC-only (enable_ais=False, enable_hpc=True, "
                "hpc_num_levels=1, w_hpc=0.10). 8-seed pinned rerun "
                "(+3.92 pp PGD-50, +4.29 pp PGD-100, both NOT significant "
                "vs baseline). Confirmed: no gradient masking.",
    },
    "D_ais_plus_hpc": {
        "label": "rhan_next_ais_hpc",
        "config": _hpc_config(ais=True),
        "checkpoint": "checkpoints/rhan_next_ais_hpc_best.pth",
        "arch": "next",
        "status": VALIDATED,
        "note": "AIS-v1 (halting-only) + HPC. Stage 3 validated: "
                "PGD-100=34.38±1.94%, Δ=+11.54pp vs A, p≈2×10⁻⁵."
                " (8 seeds, 2026-08-26).",
    },
    "E1_ais_hpc_recon": {
        "label": "rhan_next_ais_hpc_recon",
        "config": _e1_config(),
        "checkpoint": None,  # TO BE TRAINED — Stage 4-E1
        "arch": "next",
        "status": PENDING,
        "note": "AIS-v1 (halting-only) + HPC + recon-mod. Only config change "
                "from D: ais_precision_recon_enabled=True. Base: D's checkpoint "
                "(loads cleanly — D's state_dict already contains "
                "precision_modulator.gain). Stage 4-E1, 16 seeds (41-56).",
    },
    "E2_ais_hpc_sbr": {
        "label": "rhan_next_ais_hpc_sbr",
        "config": _e2_config(),
        "checkpoint": None,  # TO BE TRAINED — Stage 4-E2
        "arch": "next",
        "status": PENDING,
        "note": "AIS-v1 (halting-only) + HPC + SBR. Only config change "
                "from D: enable_sbr=True. Base: D's checkpoint (new slot "
                "parameters appear as missing keys, freshly initialized). "
                "Stage 4-E2, 16 seeds (41-56).",
    },
    "E3_ais_hpc_t6": {
        "label": "rhan_next_ais_hpc_t6",
        "config": _e3_config(),
        "checkpoint": None,  # TO BE TRAINED — Stage 4-E3
        "arch": "next",
        "status": PENDING,
        "note": "AIS-v1 (halting-only) + HPC + T=6 foraging steps. Only "
                "config change from D: max_foraging_steps=6 (D=4). Base: D's "
                "checkpoint. Stage 4-E3, 16 seeds (41-56).",
    },
}


def matrix_keys() -> List[str]:
    """All registry keys in a stable order."""
    return list(ABLATION_MATRIX)


def get_entry(key: str) -> Dict[str, Any]:
    """Return the entry dict for a key; raise KeyError on unknown keys."""
    if key not in ABLATION_MATRIX:
        raise KeyError(
            f"Unknown ablation matrix key {key!r}. Known keys: "
            f"{matrix_keys()}. Use runner.resolve(key) or eval_specs(...).")
    return ABLATION_MATRIX[key]


def status(key: str) -> str:
    return get_entry(key)["status"]


def is_eval_eligible(key: str, checkpoint_present: bool) -> bool:
    """Eval eligibility per the task spec: VALIDATED always; PENDING only
    once its checkpoint is present. SCAFFOLDED_NOT_RUN never."""
    st = get_entry(key)["status"]
    if st == VALIDATED:
        return True
    if st == PENDING:
        return checkpoint_present
    return False
