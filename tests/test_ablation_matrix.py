"""
Stage 2 — ablation matrix registry + runner + eval flag.

Confirms all four A/B/C/D entries resolve to valid, distinct, non-crashing
configs (the task's test_ablation_matrix.py requirement) and that the eval
entrypoint's --ablation-matrix flag builds --ckpt-specs from the registry.
"""
import ast
import os
import shlex
import subprocess
import sys

import pytest

from rhan_core.ablation import matrix as m
from rhan_core.ablation import runner as r
from rhan_core.config.pillar_config import RHANNextConfig


# ── Registry: all four entries resolve, valid + distinct ────────────────────

def test_all_four_keys_resolve():
    assert m.matrix_keys() == ["A_baseline", "B_ais_only",
                               "C_hpc_only", "D_ais_plus_hpc"]
    for key in m.matrix_keys():
        entry = m.get_entry(key)          # must not raise
        assert entry["label"] and entry["arch"] and entry["status"]
        # config None only for A (static TRADES); every other entry must
        # validate (enable_sbr/iwm locked, hpc_num_levels <= 1).
        cfg = entry["config"]
        if cfg is not None:
            cfg.validate()
        assert r.resolve(key)["label"] == entry["label"]


def test_four_entries_are_distinct():
    cfgs = [m.get_entry(k)["config"] for k in m.matrix_keys()]
    assert cfgs[0] is None                      # A: not RHANNext at all
    for i in range(1, len(cfgs)):
        assert cfgs[i] is not None and isinstance(cfgs[i], RHANNextConfig)
        for j in range(i + 1, len(cfgs)):
            assert cfgs[i] != cfgs[j], \
                f"matrix entries must be distinct: {m.matrix_keys()[i]} == {m.matrix_keys()[j]}"
    # C: HPC only, no AIS.  D: AIS-v1 + HPC.
    assert m.get_entry("C_hpc_only")["config"].enable_ais is False
    assert m.get_entry("C_hpc_only")["config"].enable_hpc is True
    assert m.get_entry("D_ais_plus_hpc")["config"].enable_ais is True
    assert m.get_entry("D_ais_plus_hpc")["config"].enable_hpc is True
    assert m.get_entry("B_ais_only")["config"].enable_ais is True
    assert m.get_entry("B_ais_only")["config"].enable_hpc is False
    # D is dormant by registry status, not by a config error.
    assert m.get_entry("D_ais_plus_hpc")["status"] == m.SCAFFOLDED_NOT_RUN


def test_w_hpc_is_the_separate_10_percent_slot():
    """w_hpc must default to 0.10 and be a distinct slot from w_recon."""
    cfg = RHANNextConfig()
    assert cfg.hpc_error_weight == pytest.approx(0.10)
    assert cfg.hpc_error_weight != 0.05
    # The matrix HPC entries pin the same weight explicitly.
    for key in ("C_hpc_only", "D_ais_plus_hpc"):
        assert m.get_entry(key)["config"].hpc_error_weight == pytest.approx(0.10)


# ── Runner: training commands ────────────────────────────────────────────────

def test_train_command_c_only_and_correct_flags():
    argv = r.train_command("C_hpc_only")
    assert "python3" in argv and "phase1_training/train_rhan_next.py" in argv
    assert "--enable-hpc" in argv and "--hpc-num-levels" in argv
    assert "--enable-ais" not in argv            # HPC-only per matrix entry C
    assert "--w-hpc" in argv
    assert argv[argv.index("--ckpt-name") + 1] == "rhan_next_hpc_only"


def test_train_command_extra_args_survive_shell_roundtrip():
    """Regression: 2026-08-12 Colab bug. The notebook passed space-joined
    extra_args ("--max-epochs 15"); after shlex.join + shell=True they arrived
    as ONE argv token and train_rhan_next.py's parse_known_args() SILENTLY
    dropped them — the Stage 2 smoke ran 60 epochs from the WRONG base
    checkpoint (and wrote no diag-json). train_command must normalize so the
    delivered argv is lossless (asserts the exact notebook construction)."""
    argv = r.train_command(
        "C_hpc_only", ckpt_name="rhan_next_hpc_only_smoke",
        extra_args=["--max-epochs 15",
                    "--target-ckpt "
                    "checkpoints/rhan_next_ais_v1_halting_only_best.pth",
                    "--batch-size 16 --accum-steps 16",
                    "--diag-json report/rhan_next_hpc_only_smoke_diag.jsonl",
                    "--force-single-gpu"])
    roundtrip = shlex.split(shlex.join(argv))
    assert roundtrip == argv, (f"shlex round-trip is lossy — the shell would "
                               f"deliver merged tokens:\n{argv}\nvs\n{roundtrip}")
    # The two flags that were silently dropped on Colab must now be real tokens.
    i = argv.index("--max-epochs")
    assert argv[i + 1] == "15"
    j = argv.index("--target-ckpt")
    assert argv[j + 1] == "checkpoints/rhan_next_ais_v1_halting_only_best.pth"
    k = argv.index("--diag-json")
    assert argv[k + 1] == "report/rhan_next_hpc_only_smoke_diag.jsonl"
    assert "--batch-size" in argv and argv[argv.index("--batch-size") + 1] == "16"


def test_train_command_delivered_argv_via_shell_is_lossless():
    """Faithful to the failing path (Popen with shell=True on the shlex.join'd
    command): the 2026-08-12 Colab bug delivered '--max-epochs 15' as ONE argv
    token. The runner must now deliver separate tokens through a real shell."""
    argv = r.train_command(
        "C_hpc_only", ckpt_name="rhan_next_hpc_only_smoke",
        extra_args=["--max-epochs 15",
                    "--target-ckpt "
                    "checkpoints/rhan_next_ais_v1_halting_only_best.pth",
                    "--batch-size 16 --accum-steps 16",
                    "--diag-json report/rhan_next_hpc_only_smoke_diag.jsonl",
                    "--force-single-gpu"])
    probe = shlex.join(argv).replace(
        "python3 phase1_training/train_rhan_next.py",
        f"{shlex.quote(sys.executable)} -c "
        f"\"import sys; print(sys.argv[1:])\"")
    out = subprocess.run(probe, shell=True, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    delivered = ast.literal_eval(out.stdout.strip())   # list printed by python -c
    start = next(i for i, t in enumerate(delivered) if t.startswith("--"))
    assert delivered[start:] == argv[2:]           # flags only, in order


def test_train_command_d_refuses_to_run():
    """D is SCAFFOLDED_NOT_RUN — the runner must refuse, not silently run."""
    with pytest.raises(ValueError, match="SCAFFOLDED_NOT_RUN"):
        r.train_command("D_ais_plus_hpc")
    with pytest.raises(ValueError, match="PENDING"):
        r.train_command("A_baseline")            # not a RHANNext training target


# ── Runner: eval eligibility + specs ─────────────────────────────────────────

def test_eval_eligibility_A_and_B_only():
    # VALIDATED (A, B) always eligible; C (PENDING) only with a checkpoint;
    # D never.
    assert m.is_eval_eligible("A_baseline", checkpoint_present=False)
    assert m.is_eval_eligible("B_ais_only", checkpoint_present=False)
    assert not m.is_eval_eligible("C_hpc_only", checkpoint_present=False)
    assert m.is_eval_eligible("C_hpc_only", checkpoint_present=True)
    assert not m.is_eval_eligible("D_ais_plus_hpc", checkpoint_present=True)


def test_eval_specs_labels_and_archs():
    specs, skipped = r.eval_specs(keys=["A_baseline", "B_ais_only",
                                        "D_ais_plus_hpc"], require_present=False)
    labels = {s["label"] for s in specs}
    assert labels == {"trades_large_baseline", "rhan_next_ais_v1_halting_only"}
    archs = {s["label"]: s["arch"] for s in specs}
    assert archs["trades_large_baseline"] == "large"
    assert archs["rhan_next_ais_v1_halting_only"] == "next"
    assert any("D_ais_plus_hpc" in why for why in skipped)
    # C is PENDING with a DECLARED checkpoint path: eligible once the file
    # exists, skipped with a clear reason while it is absent (env-dependent,
    # so branch on what is actually on disk).
    c_specs, c_skipped = r.eval_specs(keys=["C_hpc_only"])
    if r.checkpoint_exists("C_hpc_only"):
        assert [s["label"] for s in c_specs] == ["rhan_next_hpc_only"]
        assert not any("C_hpc_only" in why for why in c_skipped)
    else:
        assert not c_specs
        assert any("C_hpc_only" in why for why in c_skipped)


def test_checkpoint_paths_resolve_to_repo():
    p = r.resolve_checkpoint_path("A_baseline")
    assert p and p.endswith("checkpoints/rhan_stl10_large_pseudolabel_best.pth")
    assert os.path.isabs(p)
    # C declared its trained-checkpoint path (2026-08-16: Step B complete).
    p_c = r.resolve_checkpoint_path("C_hpc_only")
    assert p_c and p_c.endswith("checkpoints/rhan_next_hpc_only_best.pth")
    assert os.path.isabs(p_c)


# ── eval_rhan.py --ablation-matrix flag plumbing ─────────────────────────────

class _ArgvSwap:
    def __init__(self, argv):
        self.argv = argv
        self._saved = None

    def __enter__(self):
        self._saved = sys.argv
        sys.argv = self.argv
        return self

    def __exit__(self, *exc):
        sys.argv = self._saved
        return False


def _argv(*args):
    return _ArgvSwap(['eval_rhan.py'] + list(args))


def test_ablation_matrix_flag_builds_specs_and_is_consumed():
    import eval_rhan
    with _argv('--ablation-matrix', 'A_baseline', 'B_ais_only',
               '--seeds', '41', '42', '43', '44', '45'):
        specs = eval_rhan._ablation_matrix_specs()
        assert '--ablation-matrix' not in sys.argv      # consumed
        assert len(specs) == 2
        assert specs[0]['label'] == 'trades_large_baseline'
        assert specs[1]['label'] == 'rhan_next_ais_v1_halting_only'
        assert specs[0]['arch'] == 'large' and specs[1]['arch'] == 'next'
        # Each path resolves to a repo checkpoint file.
        for s in specs:
            assert os.path.basename(s['path']).endswith('.pth')


def test_ablation_matrix_flag_equals_form_stripped_and_parsed():
    """--ablation-matrix=A_baseline (equals form) must be parsed AND stripped
    (the frozen parser must never see the flag token)."""
    import eval_rhan
    with _argv('--ablation-matrix=A_baseline',
               '--seeds', '41', '42', '43', '44', '45'):
        specs = eval_rhan._ablation_matrix_specs()
        assert '--ablation-matrix' not in sys.argv
        assert not any(a.startswith('--ablation-matrix') for a in sys.argv)
        assert specs[0]['label'] == 'trades_large_baseline'


def test_ablation_matrix_unknown_key_raises():
    import eval_rhan
    with _argv('--ablation-matrix', 'Not_A_Key'):
        with pytest.raises(KeyError):
            eval_rhan._ablation_matrix_specs()


def test_ablation_matrix_mutually_exclusive_with_ckpt_specs():
    import eval_rhan
    with _argv('--ablation-matrix', 'A_baseline', '--ckpt-specs',
               'x:y.pth:next'):
        with pytest.raises(SystemExit):
            eval_rhan._ablation_matrix_specs()


def test_ablation_matrix_dormant_entry_cannot_eval():
    import eval_rhan
    with _argv('--ablation-matrix', 'D_ais_plus_hpc'):
        with pytest.raises(SystemExit):
            eval_rhan._ablation_matrix_specs()
