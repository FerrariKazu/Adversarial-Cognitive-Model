"""
Sweep --resume regression tests.

Covers the per-cell resume added to eval_full_epsilon_sweep.py: with --resume,
a (ckpt_label, seed, eps) cell already present in the output dir per-seed CSV
is carried forward — PGD is NOT re-run for it, the model is not reloaded for a
fully-done checkpoint, and the carried rows still land in the fresh per-seed
CSV and the aggregated results CSV.

This is the fix behind the 2026-08-17 re-run lesson: the notebook syncs each
leg's CSVs to HF immediately after the leg, so a session timeout in the next
leg can never lose already-completed cells again — and a re-run with --resume
never re-pays A/B PGD-50 cells that are already done.
"""
import csv as _csv
import sys

import pytest

import eval_full_epsilon_sweep as _sweep  # noqa: E402


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


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, x):
        import torch
        return torch.zeros(x.size(0), 10, device=x.device)


def _fake_load_test_samples(n_samples, seed=42):
    import torch
    return torch.zeros(n_samples, 3, 8, 8), torch.zeros(n_samples, dtype=torch.long)


def _fake_run_pgd(model, x_norm_cpu, y_cpu, eps, steps, device, batch_size=50,
                  norm_space=False):
    import torch
    return x_norm_cpu, torch.zeros(3)


def _fake_dprime(model, x_adv_cpu, y_true_cpu, device, batch_size=50):
    return 0.75


@pytest.fixture
def patched_sweep(monkeypatch, tmp_path):
    """Monkeypatch all heavy sweep machinery; count PGD + model-load calls."""
    calls = {"pgd": 0, "load_model": 0}

    def _counting_load_model(arch, ckpt_path, device, freeze_gaze=False):
        calls["load_model"] += 1
        return _DummyModel()

    def _counting_pgd(model, x_norm_cpu, y_cpu, eps, steps, device,
                      batch_size=50, norm_space=False):
        calls["pgd"] += 1
        return _fake_run_pgd(model, x_norm_cpu, y_cpu, eps, steps, device,
                             batch_size, norm_space)

    monkeypatch.setattr(_sweep, "load_test_samples", _fake_load_test_samples)
    monkeypatch.setattr(_sweep, "load_model", _counting_load_model)
    monkeypatch.setattr(_sweep, "run_pgd_batched", _counting_pgd)
    monkeypatch.setattr(_sweep, "compute_dprime_batched", _fake_dprime)
    return calls, tmp_path


def _run_sweep(out_dir, seeds, eps_list, resume=False):
    argv = ["eval_full_epsilon_sweep.py",
            "--n-samples", "4",
            "--seeds"] + list(map(str, seeds)) + \
           ["--eps-list"] + list(map(str, eps_list)) + \
           ["--pgd-steps", "2",
            "--batch-size", "2",
            "--ckpt-specs", "ckpt_a:/tmp/fake_a.pth:large",
            "ckpt_b:/tmp/fake_b.pth:next",
            "--output-dir", str(out_dir)]
    if resume:
        argv.append("--resume")
    with _ArgvSwap(argv):
        _sweep.main()


def _csv_rows(path):
    with open(path, newline='') as f:
        return list(_csv.DictReader(f))


def test_resume_skips_all_done_cells_and_carries_rows(patched_sweep):
    calls, tmp = patched_sweep
    out = tmp / "sweep"

    # First run: 2 labels x 2 seeds x 2 eps. eps=0.0 is the clean path (no
    # PGD call), so PGD runs only for the eps=0.094 cells: 2 labels x 2 seeds.
    _run_sweep(out, seeds=[41, 42], eps_list=[0.0, 0.094])
    assert calls["pgd"] == 4
    assert calls["load_model"] == 2
    csv_p = out / "epsilon_sweep_per_seed.csv"
    assert csv_p.exists()

    # Second run with --resume: every cell already in the CSV -> zero PGD,
    # zero model loads, rows carried forward into the fresh CSV.
    calls["pgd"] = 0
    calls["load_model"] = 0
    _run_sweep(out, seeds=[41, 42], eps_list=[0.0, 0.094], resume=True)
    assert calls["pgd"] == 0
    assert calls["load_model"] == 0

    rows = _csv_rows(csv_p)
    assert len(rows) == 8             # 2 labels x 2 seeds x 2 eps, all carried
    assert {(r["ckpt_label"], int(r["seed"]), round(float(r["eps_pixel"]), 4))
            for r in rows} == {
        (lab, s, e) for lab in ("ckpt_a", "ckpt_b")
        for s in (41, 42) for e in (0.0, 0.094)}

    # Aggregated CSV must still cover the carried cells (2 labels x 2 eps =
    # 4 rows, n_seeds=2 each).
    agg_rows = _csv_rows(out / "epsilon_sweep_results.csv")
    assert len(agg_rows) == 4
    assert all(int(r["n_seeds"]) == 2 for r in agg_rows)


def test_resume_only_skips_cells_already_present(patched_sweep):
    calls, tmp = patched_sweep
    out = tmp / "sweep"

    # First run: full 2-seed x 2-eps grid for both labels.
    _run_sweep(out, seeds=[41, 42], eps_list=[0.0, 0.094])

    # Second run with --resume but an EXTRA seed (43): only seed-43 cells are
    # new. Both labels have a pending seed-43 cell, so both models reload, but
    # only the new eps=0.094 cells run PGD (one per label).
    calls["pgd"] = 0
    calls["load_model"] = 0
    _run_sweep(out, seeds=[41, 42, 43], eps_list=[0.0, 0.094], resume=True)
    assert calls["pgd"] == 2          # ckpt_a + ckpt_b, seed=43, eps=0.094
    assert calls["load_model"] == 2   # both labels have a new (43, *) cell

    rows = _csv_rows(out / "epsilon_sweep_per_seed.csv")
    assert len(rows) == 12            # 2 labels x 3 seeds x 2 eps
    assert {int(r["seed"]) for r in rows} == {41, 42, 43}


def test_resume_noop_without_flag(patched_sweep):
    """Without --resume, an existing CSV does NOT suppress any work."""
    calls, tmp = patched_sweep
    out = tmp / "sweep"

    _run_sweep(out, seeds=[41], eps_list=[0.094])
    calls["pgd"] = 0
    calls["load_model"] = 0

    _run_sweep(out, seeds=[41], eps_list=[0.094])   # no --resume
    assert calls["pgd"] == 2          # both labels re-ran despite existing CSV
    assert calls["load_model"] == 2
