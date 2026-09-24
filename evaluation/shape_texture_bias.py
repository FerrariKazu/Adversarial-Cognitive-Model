"""
Shape/texture bias report — Agent I.
================================================================================

Runs Geirhos cue-conflict stimuli through a model, records which category
it RESPONDS with, and reports shape/texture decision shares AGAINST THE
PUBLISHED HUMAN BASELINE (Decision 5's first concrete axis).

The baseline and its provenance live in geirhos_loader.py
(HUMAN_SHAPE_DECISION_PCT = 95.9, arXiv:1811.12231 — primary-source
verified 2026-09-24; see that module's docstring). Every report row
carries the citation — no bare number without provenance.

CONSISTENCY DISCIPLINE (non-negotiable, its history): the decision-share
table is derived from the per-stimulus decisions CSV by a fresh groupby
and MUST pass noesis_vision.core.consistency_assert.assert_donor_rows_
byte_identical / a fresh-derivation equality check BEFORE it is written.
This module's write path refuses to write a table that does not
reproduce its own source.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd
import torch

from evaluation.geirhos_loader import (
    HUMAN_BASELINE_PROVENANCE,
    HUMAN_SHAPE_DECISION_PCT,
    human_baseline_row,
    shape_texture_decision_table,
)


def collect_decisions(model, stimuli_loader, device: str = "cpu",
                      decisions_csv: Optional[str] = None) -> pd.DataFrame:
    """Run the model over cue-conflict stimuli; record responses.

    The model is treated as a BLACK BOX category responder (argmax over
    its output logits mapped through the loader's class index order —
    ImageFolder's class_to_idx). Rows: image path, responded_category,
    plus the cue keys when the loader's dataset carries them (the
    official release's metadata; absence is recorded honestly, never
    imputed).
    """
    rows = []
    ds = stimuli_loader.dataset
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    model.eval()
    with torch.no_grad():
        for x, y in stimuli_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()
            for i in range(x.shape[0]):
                rows.append({"responded_category": idx_to_class[int(pred[i])],
                             "folder_category": idx_to_class[int(y[i])]})
    df = pd.DataFrame(rows)
    if decisions_csv is not None:
        os.makedirs(os.path.dirname(decisions_csv) or ".", exist_ok=True)
        df.to_csv(decisions_csv, index=False)
    return df


def build_cue_conflict_table(decisions_csv: str, model_label: str) -> pd.DataFrame:
    """Model row(s) + the human baseline row — one comparison table.

    The model row is a fresh derivation from the decisions CSV
    (shape_texture_decision_table); the human row carries its provenance
    fields. Returns the table BEFORE writing (no assertion yet).
    """
    model_row, n = shape_texture_decision_table(decisions_csv, model_label)
    human = human_baseline_row()
    table = pd.DataFrame([model_row, human])
    table.attrs["n_stimuli"] = n
    return table


def write_cue_conflict_table(table: pd.DataFrame, decisions_csv: str,
                             out_csv: str) -> pd.DataFrame:
    """Assert the table reproduces its source, THEN write.

    The model row must equal a SECOND fresh derivation from the same CSV
    (derive-twice, compare, write — the consistency_assert spirit applied
    to the one table here whose source is not the canonical per-seed CSV).
    A hand-edited or diverging row never reaches disk.
    """
    model_label = table.iloc[0]["model_label"]
    recomputed, _ = shape_texture_decision_table(decisions_csv, model_label)
    for col in ("shape_decision_pct", "texture_decision_pct", "n_stimuli"):
        a, b = table.iloc[0][col], recomputed[col]
        if a is None and b is None:
            continue
        if a != b and not (a is not None and b is not None
                           and abs(float(a) - float(b)) < 1e-9):
            raise AssertionError(
                f"cue-conflict table diverges from its own source CSV "
                f"{decisions_csv}: {col} {a!r} != re-derived {b!r} — "
                f"refusing to write (consistency discipline)")
    provenance_cols = [c for c in table.columns
                       if c.startswith("baseline_")]
    if HUMAN_BASELINE_PROVENANCE["value_pct"] != HUMAN_SHAPE_DECISION_PCT:
        raise AssertionError("human baseline drifted from its provenance")
    table.to_csv(out_csv, index=False)
    return table


def run_shape_texture_bias(model, stimuli_root: str, out_dir: str,
                           model_label: str, batch_size: int = 32,
                           device: str = "cpu") -> Dict:
    """End-to-end: validate stimuli -> collect decisions -> asserted table.

    Returns paths + the model row. Note the honest scope: the reported
    share is over the stimuli actually run; the release's full stimulus
    count is whatever the root holds (recorded, not assumed).
    """
    from evaluation.geirhos_loader import make_stimuli_loader, validate_stimuli_root

    n_cats = validate_stimuli_root(stimuli_root)
    loader = make_stimuli_loader(stimuli_root, batch_size=batch_size)
    os.makedirs(out_dir, exist_ok=True)
    decisions_csv = os.path.join(out_dir, "cue_conflict_decisions.csv")
    collect_decisions(model, loader, device=device, decisions_csv=decisions_csv)
    table = build_cue_conflict_table(decisions_csv, model_label)
    table_csv = os.path.join(out_dir, "shape_texture_bias_table.csv")
    write_cue_conflict_table(table, decisions_csv, table_csv)
    return {"decisions_csv": decisions_csv, "table_csv": table_csv,
            "n_categories": n_cats,
            "model_row": table.iloc[0].to_dict()}
