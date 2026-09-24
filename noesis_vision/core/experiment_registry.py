"""
Experiment / comparator registry — Agent A. ADAPTED from the Gen-0
pattern (scripts/comparator_registry.py) to the noesis_vision namespace.

Gen-0 rule 1b, carried forward: NEVER re-run eval on an already-validated
checkpoint. Every registry entry records the checkpoint's sha256 so a
silently swapped/corrupted checkpoint file can never have its old numbers
cited against it; `load_comparator` returns the exact validated rows and
refuses (loudly) any request for a seed the comparator was never
evaluated on.

ADAPTED, not verbatim: the Gen-0 file hardcodes the four Gen-0 comparator
entries (their CSV paths, HF paths, and checkpoint hashes are Gen-0
artifacts). This module provides the registry MECHANISM — the same
verification order (file exists / sha256 matches / seeds are a subset) —
while entries are registered per-experiment by the runs that produce them.
The step-6 reference checkpoint (Part 2/4) is the first entry the Gen-1
ladder will register; Gen-0's frozen D record stays in the Gen-0 file.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

from noesis_vision.core.provenance import file_sha256


class ExperimentRegistry:
    """Registry of validated experiment artifacts (comparators).

    Each entry: checkpoint_path, checkpoint_sha256, source_csv,
    validated_seeds, validated_date, plus free-form notes/caveats (e.g.
    seed-count mismatch warnings — the Gen-0 B/C entries' caveat pattern).
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Dict] = {}

    def register(self, label: str, *, checkpoint_path: str,
                 source_csv: str, validated_seeds: Sequence[int],
                 validated_date: str, note: str = "",
                 expected_sha256: Optional[str] = None) -> Dict:
        """Register a validated comparator.

        The checkpoint file must exist at registration time and its actual
        sha256 is recorded (or verified against `expected_sha256` when
        provided) — an entry whose hash does not match a prior registration
        of the same label raises immediately.
        """
        if not label or not isinstance(label, str):
            raise ValueError("label must be a non-empty string")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"cannot register comparator '{label}': checkpoint missing "
                f"at {checkpoint_path} — never register a comparator whose "
                f"file you cannot hash")
        actual = file_sha256(checkpoint_path)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(
                f"comparator '{label}': checkpoint sha256 mismatch — "
                f"expected {expected_sha256}, actual {actual}. A silently "
                f"swapped/corrupted checkpoint can never be registered.")
        if label in self._entries and \
                self._entries[label]["checkpoint_sha256"] != actual:
            raise ValueError(
                f"comparator '{label}' already registered with checkpoint "
                f"sha256 {self._entries[label]['checkpoint_sha256']}; "
                f"refusing to re-register under a different file — register "
                f"a new label instead")

        entry = {
            "label": label,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": actual,
            "source_csv": source_csv,
            "validated_seeds": sorted(set(int(s) for s in validated_seeds)),
            "validated_date": validated_date,
            "note": note,
        }
        self._entries[label] = entry
        return dict(entry)

    def load_comparator(self, label: str,
                        requested_seeds: Optional[Sequence[int]] = None) -> pd.DataFrame:
        """Return the exact validated rows for `label` (never re-runs eval).

        Verification, in order (the Gen-0 pattern):
          1. source_csv exists — fail loudly if missing;
          2. the checkpoint file's CURRENT sha256 must still match the
             registered hash — catches a silently swapped/corrupted
             checkpoint being cited with old numbers;
          3. requested_seeds must be a SUBSET of validated_seeds — raise
             loudly rather than silently returning fewer rows than expected.
        """
        if label not in self._entries:
            raise KeyError(
                f"no comparator registered under '{label}' — registered: "
                f"{sorted(self._entries)}")
        entry = self._entries[label]

        if not os.path.exists(entry["source_csv"]):
            raise FileNotFoundError(
                f"comparator '{label}': source CSV missing: "
                f"{entry['source_csv']}")
        if os.path.exists(entry["checkpoint_path"]):
            current = file_sha256(entry["checkpoint_path"])
            if current != entry["checkpoint_sha256"]:
                raise ValueError(
                    f"comparator '{label}': checkpoint file changed since "
                    f"registration (registered sha256 "
                    f"{entry['checkpoint_sha256']}, current {current}) — "
                    f"refusing to cite old numbers against a new file")
        else:
            raise FileNotFoundError(
                f"comparator '{label}': checkpoint file has disappeared "
                f"since registration: {entry['checkpoint_path']}")

        df = pd.read_csv(entry["source_csv"])
        if "ckpt_label" in df.columns:
            df = df[df["ckpt_label"] == label] if (df["ckpt_label"] == label).any() else df
        if requested_seeds is not None:
            req = set(int(s) for s in requested_seeds)
            missing = req - set(entry["validated_seeds"])
            if missing:
                raise ValueError(
                    f"comparator '{label}' was validated on seeds "
                    f"{entry['validated_seeds']}; refusing to serve "
                    f"unvalidated seeds {sorted(missing)} — request a subset "
                    f"of the validated seeds only")
            if "seed" in df.columns:
                df = df[df["seed"].astype(int).isin(req)]
        return df

    def seed_count_caveat(self, label: str) -> Optional[str]:
        """The explicit warning callers must surface when mixing comparators
        with different validated seed counts (never silently average across
        mismatched n — the Gen-0 B/C caveat rule)."""
        if label not in self._entries:
            return None
        n = len(self._entries[label]["validated_seeds"])
        others = {k: len(v["validated_seeds"])
                  for k, v in self._entries.items() if k != label}
        mismatched = [k for k, m in others.items() if m != n]
        if mismatched:
            return (
                f"comparator '{label}' is {n}-seed validated; "
                f"{mismatched} have different seed counts — any comparison "
                f"mixing them MUST explicitly flag the seed-count mismatch, "
                f"never silently average across mismatched n")
        return None

    def labels(self) -> List[str]:
        return sorted(self._entries)

    def entry(self, label: str) -> Dict:
        if label not in self._entries:
            raise KeyError(f"no comparator registered under '{label}'")
        return dict(self._entries[label])
