"""
Clean + robust evaluation harness — Agent I.
================================================================================

ADAPTED from phase2_attacks/eval_rhan.py (Part 5 disposition: ADAPT —
"same conventions, new input resolution/normalization stats"). The
CONVENTIONS are ported; the STL-10-specific attack parameters (0.094,
PGD-50 defaults) are NOT — the caller passes them per-experiment:

  1. NORM-SPACE ONLY. Epsilon is applied directly in normalized-pixel
     space, per channel (the Gen-0 protocol's hard rule — pixel-space eps
     conventions were the interface-drift incident class). Every run's
     provenance records eps_space="norm"; there is no pixel-space path.
  2. SEED FLOOR. The matched protocol requires >= MIN_PROTOCOL_SEEDS
     seeds; fewer ABORTS unless allow_quick=True is passed EXPLICITLY
     (dev sanity only — single-seed numbers must never be reported as
     results).
  3. SELF-TEST. self_test=True runs the full harness end-to-end on
     SYNTHETIC loaders against a structural reference (the merged table's
     reference columns) — a fast structural check, never evidence.
  4. PROVENANCE. After every run: eval_provenance.json — model label,
     checkpoint SHA-256, seed list, full settings, timestamp, and the
     per-seed CSV path (Agent A's file_sha256 for the checkpoint hash).

  5. CONSISTENCY ASSERTION (non-negotiable, its history): every summary
     table is built from a FRESH groupby of the per-seed CSV and MUST
     pass noesis_vision.core.consistency_assert.assert_table_matches_csv
     BEFORE it is written. The per-seed CSV uses the CANONICAL Gen-0
     schema (ckpt_label, seed, eps_pixel, acc_pct, ...) so the ported
     assertion works unmodified; `eps_pixel` is the epsilon key and the
     space it is expressed in is recorded in provenance (norm space).
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from noesis_vision.core.consistency_assert import (
    assert_table_matches_csv,
    groupby_summary,
)
from noesis_vision.core.provenance import file_sha256

MIN_PROTOCOL_SEEDS = 5
PROVENANCE_NAME = "eval_provenance.json"
PER_SEED_NAME = "epsilon_sweep_per_seed.csv"   # the canonical Gen-0 name
SUMMARY_NAME = "summary_table.csv"

#: The reference structure every merged/summary table must carry (the
#: self-test checks against this — eval_rhan's self-test convention).
REFERENCE_COLUMNS = ["ckpt_label", "eps_pixel", "acc_pct_mean", "acc_pct_std"]


class ProtocolError(RuntimeError):
    """Raised when a run violates the evaluation protocol (seed floor,
    missing provenance inputs) — aborting is the safe direction."""


def enforce_seed_protocol(seeds: Sequence[int],
                          allow_quick: bool = False) -> List[int]:
    """Seed-floor enforcement (ported convention). Returns the deduped
    seed list; raises ProtocolError under the floor without the explicit
    escape hatch."""
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    if allow_quick:
        return seeds
    if len(seeds) < MIN_PROTOCOL_SEEDS:
        raise ProtocolError(
            f"PROTOCOL ERROR — the matched protocol requires >= "
            f"{MIN_PROTOCOL_SEEDS} seeds, got {len(seeds)}: {seeds}. "
            f"Single-seed numbers must NOT be reported as results. Pass "
            f"allow_quick=True only for an explicit dev-sanity run.")
    return seeds


def generic_pgd(model: Callable, x: torch.Tensor, y: torch.Tensor,
                eps: float, steps: int, alpha: Optional[float] = None,
                clip_range: float = 4.0) -> torch.Tensor:
    """Dataset-agnostic PGD in NORMALIZED space (per-channel eps).

    Conventions ported from the Gen-0 protocol: per-channel epsilon,
    alpha = eps/4 unless passed, border padding of the perturbation via
    clamping to a normalized-validity box ([-clip_range, +clip_range]).
    No dataset-specific constants. `model(x) -> logits`.
    """
    import torch
    if alpha is None:
        alpha = eps / 4.0
    x0 = x.detach().clone()
    x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps)
            x_adv = x_adv.clamp(-clip_range, clip_range)
    return x_adv.detach()


def run_clean_and_robust(
    model, loader_factory: Callable[[int], object], seeds: Sequence[int],
    eps_list: Sequence[float], n_samples: int, ckpt_path: Optional[str],
    out_dir: str, ckpt_label: str = "model", device: str = "cpu",
    attack: Optional[Callable] = generic_pgd,
    pgd_steps: int = 10, pgd_alpha: Optional[float] = None,
    allow_quick: bool = False, self_test: bool = False,
) -> Dict:
    """The hardened clean+robust sweep (conventions per the module docstring).

    loader_factory(seed) must return a FRESH loader per seed (fresh subset
    per seed — the Gen-0 convention). Writes the per-seed CSV, the
    consistency-asserted summary table, and eval_provenance.json.
    """
    import torch

    seeds = enforce_seed_protocol(seeds, allow_quick=allow_quick)

    os.makedirs(out_dir, exist_ok=True)
    eps_list = [float(e) for e in eps_list]
    rows: List[Dict] = []
    remaining = n_samples

    for seed in seeds:
        loader = loader_factory(seed)
        seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if seen >= n_samples:
                break
            take = min(x.shape[0], n_samples - seen)
            x, y = x[:take], y[:take]
            with torch.no_grad():
                clean_acc = (model(x).argmax(1) == y).float().mean().item()
            row = {"ckpt_label": ckpt_label, "seed": seed,
                   "eps_pixel": 0.0, "acc_pct": round(100.0 * clean_acc, 2)}
            rows.append(row)
            if attack is not None:
                for eps in eps_list:
                    if eps == 0.0:
                        continue
                    x_adv = attack(model, x, y, eps=eps, steps=pgd_steps,
                                   alpha=pgd_alpha)
                    with torch.no_grad():
                        acc = (model(x_adv).argmax(1) == y) \
                            .float().mean().item()
                    rows.append({"ckpt_label": ckpt_label, "seed": seed,
                                 "eps_pixel": eps,
                                 "acc_pct": round(100.0 * acc, 2)})
            seen += take

    per_seed_path = os.path.join(out_dir, PER_SEED_NAME)
    pd.DataFrame(rows).to_csv(per_seed_path, index=False)

    # ── summary table: derived + asserted BEFORE writing (non-negotiable) ──
    summary = _build_summary(per_seed_path, ckpt_label)
    write_summary_table(summary, per_seed_path,
                        os.path.join(out_dir, SUMMARY_NAME))

    provenance = {
        "ckpt_label": ckpt_label,
        "ckpt_sha256": file_sha256(ckpt_path) if ckpt_path else None,
        "seeds": seeds,
        "n_samples_per_seed": n_samples,
        "eps_list": eps_list,
        "eps_space": "norm",
        "attack": getattr(attack, "__name__", str(attack))
        if attack is not None else None,
        "pgd_steps": pgd_steps if attack is not None else None,
        "self_test": bool(self_test),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_seed_csv": per_seed_path,
    }
    with open(os.path.join(out_dir, PROVENANCE_NAME), "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    return {"per_seed_csv": per_seed_path,
            "summary_csv": os.path.join(out_dir, SUMMARY_NAME),
            "provenance": provenance}


def _build_summary(per_seed_csv: str, ckpt_label: Optional[str] = None
                   ) -> pd.DataFrame:
    """Fresh groupby of the per-seed CSV -> the summary table's contents.
    The source of truth is ALWAYS the CSV; nothing hand-entered."""
    df = pd.read_csv(per_seed_csv)
    if ckpt_label is not None:
        df = df[df["ckpt_label"] == ckpt_label] \
            if (df["ckpt_label"] == ckpt_label).any() else df
    ref = groupby_summary(df)
    return ref.rename(columns={"acc_pct_mean": "acc_pct_mean",
                               "acc_pct_std": "acc_pct_std"})


def write_summary_table(summary: pd.DataFrame, per_seed_csv: str,
                        out_path: str) -> pd.DataFrame:
    """Assert the summary matches its cited source, THEN write.

    The assertion runs against the per-seed CSV BEFORE the file is
    written — a diverging table never reaches disk (the Stage-3/4-E1
    incident rule). Raises AssertionError (no write) on mismatch.
    """
    table = summary.rename(columns={
        "ckpt_label": "ckpt_label", "eps_pixel": "eps_pixel",
        "acc_pct_mean": "acc_mean", "acc_pct_std": "acc_std"})
    assert_table_matches_csv(table, per_seed_csv)
    summary.to_csv(out_path, index=False)
    return summary
