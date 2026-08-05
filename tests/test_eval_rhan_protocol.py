"""
Stage 3.1 — protocol guards on the frozen eval entrypoint (eval_rhan.py).

Covers the four guarantees added so every future number routes through a
hardened file: forced norm-space, >=5-seed floor with --allow-quick escape,
the checked-in --self-test reference, and the post-run provenance JSON.
"""
import csv
import hashlib
import json
import os
import sys

import pytest

import eval_rhan  # noqa: E402  (patches _sweep.load_model; no side effects)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── 0. argv scanner (space AND equals forms, last occurrence wins) ──────────

def test_argv_scanner_equals_form():
    with _argv('--output-dir=/tmp/x', '--seeds', '41', '42', '43', '44', '45',
               '--eps-list=0.0', '--baseline-label=base'):
        assert eval_rhan._argv_scalar('--output-dir') == '/tmp/x'
        assert eval_rhan._argv_scalar('--baseline-label') == 'base'
        assert eval_rhan._argv_flag_values('--eps-list', float) == [0.0]
        assert eval_rhan._seeds_from_argv() == [41, 42, 43, 44, 45]


def test_argv_scanner_last_occurrence_wins():
    with _argv('--seeds', '1', '2', '--seeds', '41', '42', '43', '44', '45'):
        assert eval_rhan._seeds_from_argv() == [41, 42, 43, 44, 45]


def test_argv_scanner_seed_equals_form():
    with _argv('--seed=77'):
        assert eval_rhan._seeds_from_argv() == [77]


# ── 1. Norm-space is mandatory ───────────────────────────────────────────────

def test_norm_space_injected_when_absent():
    with _argv('--seeds', '41', '42', '43', '44', '45'):
        eval_rhan._force_norm_space()
        assert '--eps-norm-space' in sys.argv


def test_norm_space_not_duplicated_when_present():
    with _argv('--eps-norm-space', '--seeds', '41', '42', '43', '44', '45'):
        eval_rhan._force_norm_space()
        assert sys.argv.count('--eps-norm-space') == 1


# ── 2. Seed floor ─────────────────────────────────────────────────────────────

def test_fewer_than_five_seeds_aborts():
    with _argv('--seeds', '41', '42', '43'):
        with pytest.raises(SystemExit):
            eval_rhan._enforce_seed_protocol()


def test_default_single_seed_aborts():
    with _argv('--n-samples', '50'):
        with pytest.raises(SystemExit):
            eval_rhan._enforce_seed_protocol()


def test_five_seeds_passes():
    with _argv('--seeds', '41', '42', '43', '44', '45'):
        seeds = eval_rhan._enforce_seed_protocol()
        assert seeds == [41, 42, 43, 44, 45]


def test_allow_quick_waives_floor_and_is_consumed():
    with _argv('--allow-quick', '--seeds', '42'):
        seeds = eval_rhan._enforce_seed_protocol()
        assert seeds == [42]
        # The frozen parser must never see the unknown flag.
        assert '--allow-quick' not in sys.argv


# ── 3. Self-test reference ────────────────────────────────────────────────────

def test_selftest_reference_exists_and_matches_live_model():
    """The checked-in reference must match the CURRENT architecture (so a
    future drift breaks --self-test rather than silently passing)."""
    ref = eval_rhan._selftest_ref()
    assert ref is not None, "eval_rhan_selftest_ref.json missing — regenerate"
    assert ref['schema'] == 'eval_rhan_selftest_v1'

    from rhan_core.config.pillar_config import RHANNextConfig
    from rhan_core.model import RHANNext
    torch = pytest.importorskip('torch')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    model = RHANNext(config=RHANNextConfig()).to(dev).eval()
    keys = sorted(model.state_dict().keys())
    key_sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    n_params = sum(p.numel() for p in model.parameters())
    assert ref['state_dict_key_sha256'] == key_sha
    assert ref['n_state_dict_keys'] == len(keys)
    assert ref['n_params'] == n_params
    assert ref['default_config'] == RHANNextConfig().to_dict()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(eval_rhan.__file__),
                                    'eval_rhan_selftest_ref.json')),
    reason='reference file not checked in')
def test_selftest_cli_passes():
    """The full --self-test path must exit 0 (config + key hash + params +
    forward shapes all match the checked-in reference)."""
    assert eval_rhan._self_test() == 0


# ── 4. Provenance JSON ────────────────────────────────────────────────────────

def test_provenance_json_content(tmp_path):
    """Provenance must include git SHA, checkpoint hash, seed list, timestamp,
    merged results, and recomputed crossover verdicts."""
    ckpt = tmp_path / 'fake.pth'
    ckpt.write_bytes(b'FAKE_CHECKPOINT_BYTES')
    out = tmp_path / 'out'
    out.mkdir()

    agg = out / 'epsilon_sweep_results.csv'
    with open(agg, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ckpt_label', 'eps_pixel', 'eps_norm_R', 'eps_norm_G',
                    'eps_norm_B', 'acc_mean', 'acc_std',
                    'macro_dprime_mean', 'macro_dprime_std', 'n_seeds'])
        w.writerow(['rhan_next_ais', '0.0', '0', '0', '0',
                    '50.0', '2.0', '1.5', '0.1', '5'])
        w.writerow(['rhan_next_ais', '0.094', '0.094', '0.094', '0.094',
                    '25.0', '1.5', '0.6', '0.2', '5'])
        w.writerow(['trades_large_baseline', '0.094', '0.094', '0.094', '0.094',
                    '21.0', '0.5', '0.5', '0.1', '5'])

    with _argv('--seeds', '41', '42', '43', '44', '45',
               '--output-dir', str(out),
               '--ckpt-specs', f'rhan_next_ais:{ckpt}:next',
               '--n-samples', '300', '--pgd-steps', '50',
               '--batch-size', '64', '--eps-list', '0.0', '0.094',
               '--baseline-label', 'trades_large_baseline'):
        eval_rhan._write_provenance()

    prov = json.loads((out / eval_rhan.PROVENANCE_NAME).read_text())
    assert prov['git_sha'] and prov['git_sha'] != 'unknown'
    assert prov['seeds'] == [41, 42, 43, 44, 45]
    assert prov['timestamp_utc']
    assert prov['eps_mode'].startswith('norm_space')
    assert prov['checkpoints'][0]['sha256'] == hashlib.sha256(
        b'FAKE_CHECKPOINT_BYTES').hexdigest()
    assert len(prov['results']) == 3
    v = prov['crossover_verdicts']
    assert len(v) == 1 and v[0]['verdict'] == 'CROSSOVER REAL'
    assert v[0]['eps'] == 0.094 and v[0]['diff_pp'] == 4.0
