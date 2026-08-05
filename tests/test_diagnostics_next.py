"""
Stage 1 — RHANNextEpochDiagnostics (train_rhan_next.py).

Verifies the two telemetry signals Step A of the Stage 1 protocol depends on:
  1. gaze shift distance (||a_t - a_{t-1}|| per boundary + total path),
  2. per-sample halting variance (effective evidence steps = sum of soft
     continuations; std > 0 when halting actually varies per sample),
plus the inherited Pi_D-per-class block and the machine-readable summary.
"""
import json

import torch

from train_rhan_next import RHANNextEpochDiagnostics


def _synthetic_traj(B=8, T=4, halt_frac=0.4):
    """Build a trajectory dict shaped like RHANNext._forage's output.

    Gaze: monotone movement along x (shift = 0.1 per boundary) plus a
    per-sample offset so no two samples are identical.
    Continuations: samples above halt_frac fraction keep full weight (~1.0)
    for all steps; the rest drop to ~0.05 after step 1 (halting fires).
    """
    acts, conts = [], []
    for t in range(T):
        a = torch.zeros(B, 2)
        a[:, 0] = 0.1 * (t + 1) + 0.01 * torch.arange(B, dtype=torch.float32)
        acts.append(a)
        if t < 2:
            conts.append(torch.full((B,), 1.0))
        else:
            c = torch.full((B,), 1.0)
            c[:int(B * halt_frac)] = 0.05
            conts.append(c)
    traj = {
        'actions': acts,
        'precisions': [torch.full((B,), 0.5 + 0.01 * i) for i in range(T)],
        'errors': [torch.rand(B) for _ in range(T)],
        'gate_alphas': [torch.full((B,), 0.5) for _ in range(T)],
        'recon_errors': [torch.full((), 1.0) for _ in range(T)],
        'uncertainties': [torch.full((B,), 0.5) for _ in range(T)],
        'continuations': conts,
        'steps': T,
    }
    return traj


def _feed(diag, traj, labels):
    beta = torch.full((labels.shape[0],), 1.0)
    diag.update(beta, traj, labels)


def test_gaze_shift_distance_is_measured():
    B, T = 8, 4
    diag = RHANNextEpochDiagnostics(max_steps=T)
    labels = torch.arange(B) % 10
    for _ in range(3):                      # 3 batches
        _feed(diag, _synthetic_traj(B, T), labels)

    # Total path = 3 boundaries * 0.1 shift, averaged over the batch.
    assert len(diag.gaze_shifts) == 3 * (T - 1)
    total = torch.cat(diag.gaze_shifts).mean().item()
    assert 0.09 < total < 0.11, f"gaze total path {total:.4f} != ~0.1"

    s = diag.summary_dict(1, 0.031)
    assert 'gaze_shift_total_mean' in s
    assert 0.09 < s['gaze_shift_total_mean'] < 0.11


def test_halt_variance_is_measured():
    B, T = 8, 4
    diag = RHANNextEpochDiagnostics(max_steps=T)
    labels = torch.arange(B) % 10
    for _ in range(5):
        _feed(diag, _synthetic_traj(B, T, halt_frac=0.5), labels)

    eff = torch.cat(diag.effective_steps)
    # Halted samples: 2 + 0.05 + 0.05 = 2.10; continuing: 4.0. So std > 0.
    assert eff.min().item() < 2.5 and eff.max().item() == 4.0
    assert eff.std().item() > 0.5, "effective steps must vary per sample"

    s = diag.summary_dict(1, 0.031)
    assert s['steps_effective_std'] > 0.5
    assert 0.4 < s['frac_halted_any'] < 0.6, "half the samples should show halting"


def test_pi_d_per_class_ordering_car_truck_top():
    """The inherited Pi_D-per-class block must surface car/truck at the top
    when the underlying precisions say so (Step A's 'ordering reproduced')."""
    B, T = 10, 4
    diag = RHANNextEpochDiagnostics(max_steps=T)
    # 5 batches, labels spread evenly; car (2) / truck (9) get highest Pi_D.
    for b in range(5):
        labels = torch.arange(B) % 10
        traj = _synthetic_traj(B, T)
        traj['precisions'] = []
        for t in range(T):
            p = torch.full((B,), 0.5)
            p[labels == 2] = 0.80
            p[labels == 9] = 0.78
            p[labels == 3] = 0.55
            traj['precisions'].append(p)
        _feed(diag, traj, labels)

    s = diag.summary_dict(1, 0.031)
    pd = s['pi_d_per_class']
    assert pd['car'] > pd['cat'] and pd['truck'] > pd['cat']
    top2 = sorted(pd.items(), key=lambda kv: -kv[1])[:2]
    assert {k for k, _ in top2} == {'car', 'truck'}, f"top-2 Pi_D: {top2}"


def test_summary_dict_is_json_serializable():
    B, T = 8, 4
    diag = RHANNextEpochDiagnostics(max_steps=T)
    labels = torch.arange(B) % 10
    _feed(diag, _synthetic_traj(B, T), labels)
    line = json.dumps(diag.summary_dict(3, 0.062))   # must not raise
    obj = json.loads(line)
    assert obj['epoch'] == 3 and obj['eps'] == 0.062
    assert obj['steps_hard_fixed'] == T
