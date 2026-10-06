"""
Geirhos cue-conflict (shape-vs-texture) infrastructure — Agent I.
================================================================================

Part 2 step 11 / Decision 5's first concrete axis: report model shape/texture
decisions on Geirhos et al.'s cue-conflict stimuli AGAINST THE PUBLISHED
HUMAN BASELINE.

HUMAN BASELINE — VERIFIED, NOT FABRICATED (Agent I STOP condition):
  * Paper: R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann,
    W. Brendel, "ImageNet-trained CNNs are biased towards texture;
    increasing shape bias improves accuracy and robustness",
    arXiv:1811.12231, ICLR 2019 (oral). Primary source verified
    2026-09-24 (arXiv abstract page + the ICLR paper text hosted at
    openreview.net/pdf?id=Bygh9j09KX).
  * Baseline: human observers made shape-based decisions in 95.9% of
    cue-conflict cases (paper text: "responding with the shape category
    (95.9% ...)"). The stimuli and the human experiments are the paper's
    own; the number below is the paper's reported figure, carried with its
    citation in EVERY report row (no bare number without provenance).

Stimuli: the official release is the texture-vs-shape repository
(github.com/rgeirhos/texture-vs-shape). The loader consumes the cue-conflict
stimuli directory in an ImageFolder-style layout; category-structure
constants from the release are NOT hardcoded here — the directory is the
source of truth, and the report records the observed category count.
"""
from __future__ import annotations

import os

#: The VERIFIED human baseline + its provenance (travels with every report).
HUMAN_SHAPE_DECISION_PCT = 95.9
HUMAN_BASELINE_PROVENANCE = {
    "value_pct": HUMAN_SHAPE_DECISION_PCT,
    "quantity": "human shape-based decision share, texture-shape cue conflict",
    "citation": ("Geirhos, Rubisch, Michaelis, Bethge, Wichmann, Brendel "
                 "(2019), 'ImageNet-trained CNNs are biased towards "
                 "texture; increasing shape bias improves accuracy and "
                 "robustness', arXiv:1811.12231, ICLR 2019 (oral)"),
    "urls": ["https://arxiv.org/abs/1811.12231",
             "https://openreview.net/pdf?id=Bygh9j09KX"],
    "verified_date": "2026-09-24",
    "verified_by": "primary-source check (arXiv page + ICLR paper text)",
}


def human_baseline_row() -> dict:
    """The human baseline as a report row — provenance attached, always."""
    row = {"model_label": "HUMAN (Geirhos et al. 2019)",
           "shape_decision_pct": HUMAN_SHAPE_DECISION_PCT,
           "texture_decision_pct": 100.0 - HUMAN_SHAPE_DECISION_PCT}
    row.update({f"baseline_{k}": v for k, v in
                HUMAN_BASELINE_PROVENANCE.items()})
    return row


def validate_stimuli_root(root: str) -> int:
    """Structural check on a cue-conflict stimuli root (ImageFolder-style
    category directories). Returns the observed category count — recorded
    in reports rather than assumed."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"cue-conflict stimuli root not found: {root}")
    cats = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    if not cats:
        raise ValueError(
            f"cue-conflict root {root} holds no category directories — "
            f"the stimuli release's layout is per-category")
    return len(cats)


def make_stimuli_loader(root: str, batch_size: int, num_workers: int = 2):
    """Loader over the cue-conflict stimuli (no ground-truth labels — the
    task is to record which category the model RESPONDS with)."""
    validate_stimuli_root(root)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = datasets.ImageFolder(root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def shape_texture_decision_table(decisions_csv: str, model_label: str):
    """Model decision shares from a decisions CSV (per-stimulus rows:
    'responded_category' at minimum) -> one summary row.

    The CSV is written by the run harness; the shares here are a fresh
    groupby of that CSV (the consistency_assert discipline — the summary
    is derived, never hand-entered). Returns (model_row, n_stimuli).
    """
    import pandas as pd
    df = pd.read_csv(decisions_csv)
    for col in ("responded_category",):
        if col not in df.columns:
            raise ValueError(
                f"decisions CSV {decisions_csv} lacks required column "
                f"{col!r}")
    n = len(df)
    # Shape vs texture shares require the cue-conflict KEY (which cue the
    # stimulus's shape vs texture category was) — present in the official
    # release's metadata; without it, only per-category response shares
    # can be reported (recorded honestly, no invented mapping).
    if "shape_category" in df.columns and "texture_category" in df.columns:
        shape_hits = ((df["responded_category"] == df["shape_category"])
                      | (df["responded_category"] == df["shape_category"]
                         .astype(str).str.lower())).sum()
        texture_hits = ((df["responded_category"] == df["texture_category"])
                        | (df["responded_category"] == df["texture_category"]
                           .astype(str).str.lower())).sum()
        row = {"model_label": model_label,
               "shape_decision_pct": round(100.0 * shape_hits / n, 2),
               "texture_decision_pct": round(100.0 * texture_hits / n, 2),
               "n_stimuli": n}
    else:
        row = {"model_label": model_label,
               "shape_decision_pct": None,
               "texture_decision_pct": None,
               "n_stimuli": n,
               "note": ("decisions CSV lacks shape/texture cue keys — "
                        "per-category response shares only; no invented "
                        "mapping")}
    return row, n
