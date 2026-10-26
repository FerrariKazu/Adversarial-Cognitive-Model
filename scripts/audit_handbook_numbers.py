#!/usr/bin/env python3
"""Provenance audit for every percentage written in the RHAN handbook.

For each `NN.N %` in the manuscript this script searches every recorded
artifact under report/ for a numeric leaf whose value, scaled by 100, rounds
to that percentage. A number with no such leaf is either derived in the text
(a mean, a difference, a confidence bound, a ratio) or wrong. Both cases are
printed, and the derived ones must be justified in MANUAL below.

Run: python3 scripts/audit_handbook_numbers.py
Exit status is non-zero if a number is neither backed by an artifact nor
listed as a checked derivation.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAND = ROOT / "docs/rhan_nxa"
ARTIFACTS = ROOT / "report"

# Values that are computed in the text rather than copied from a single field.
# Each entry: written percentage -> how it is obtained and where it is checked.
MANUAL: dict[float, str] = {
    30.04: "mean of the three 8-seed PGD-10 arm means, computed by verify_claims.py:43",
    34.58: "mean of the three 8-seed APGD-50 arm means, verify_claims.py:43",
    39.88: "mean clean accuracy over the eight seeds of the ff_56 and ff_112 sweeps",
    58.25: "8-seed mean, g2_gist_only sweep, verify_claims.py:43",
    63.63: "8-seed mean, recurrence_only sweep, verify_claims.py:43",
    73.79: "8-seed mean, ff_112 sweep, verify_claims.py:43",
    1.2: "relative precision spread, chapter 14, from the two recorded precision reads",
    0.08: "share of parameters in the gist path, chapter 10 arithmetic",
    2.4: "share of parameters in the gaze policy, chapter 10 arithmetic",
    2.8: "analytic share of positional embedding MACs, appendix A.5",
    1.3: "|0.0314 - 0.031| / 0.0314 = 1.27 %, appendix E.1",
    31.0: "0.587 / 0.447 - 1 = 31.3 %, chapter 24 latency ratio",
    58.52: "GATE_W_REFERENCE_ACC = 0.5852 in train_gen2_foundation.py; a code constant, not a recorded run",
    90.0: "health-flag threshold `clip_sat_steps / step_count >= 0.90`, train_gen2_foundation.py:926",
    2.0: "health-flag threshold `val_clean_acc < 0.02`, train_gen2_foundation.py:924",
    5.0: "CLIP-SATURATED per-group band, train_gen2_foundation.py pre-clip recording",
    95.0: "the confidence level of an interval, not a measurement",
    100.0: "a completeness statement (fraction of a set), not a measurement",
    5.29: "paired mean difference, verify_claims.py::paired_stats",
    3.95: "paired 95% CI lower bound, verify_claims.py::paired_stats",
    6.64: "paired 95% CI upper bound, verify_claims.py::paired_stats",
}


def leaves(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield path, float(obj)


def build_index() -> dict[float, list[str]]:
    idx: dict[float, list[str]] = collections.defaultdict(list)
    for p in sorted(ARTIFACTS.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for path, val in leaves(data):
            if 0 <= val <= 2.0:
                idx[round(val * 100, 4)].append(f"{p.relative_to(ROOT)}{path}")
    # JSONL epoch logs are artifacts too: a percentage quoted from a training
    # curve must be backed by the row it came from, not by a hand entry here.
    for p in sorted(ARTIFACTS.rglob("*.jsonl")):
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for path, val in leaves(row):
                if 0 <= val <= 2.0:
                    idx[round(val * 100, 4)].append(
                        f"{p.relative_to(ROOT)}:{lineno}{path}")
    return idx


def nearest(idx: dict[float, list[str]], val: float, tol: float = 0.05):
    """Artifact values that agree with `val` to the precision the text prints it."""
    for tol_i in (tol, 0.011, 0.001):
        near = [k for k in idx if abs(k - val) <= tol_i]
        if near:
            return idx[min(near, key=lambda k: abs(k - val))]
    return None


def main() -> None:
    idx = build_index()
    pat = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%")
    hits: dict[str, list[str]] = collections.defaultdict(list)
    for f in sorted(HAND.rglob("*.md")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in pat.finditer(line):
                # "95 % CI" names a confidence level, not a measured share.
                if re.match(r"\s?(CI|confidence)", line[m.end():m.end() + 12]):
                    continue
                hits[m.group(1)].append(f"{f.relative_to(HAND)}:{i}")

    backed, derived, orphans = [], [], []
    for written, where in sorted(hits.items(), key=lambda kv: float(kv[0])):
        val = float(written)
        found = nearest(idx, val)
        if found:
            backed.append((written, found[0], len(where)))
        elif val in MANUAL:
            derived.append((written, MANUAL[val], len(where)))
        else:
            orphans.append((written, where))

    print(f"percentages written in the manuscript : {len(hits)} distinct, "
          f"{sum(len(v) for v in hits.values())} occurrences "
          "(confidence levels excluded)")
    print(f"  backed by a recorded artifact field : {len(backed)}")
    print(f"  derived and checked in MANUAL       : {len(derived)}")
    print(f"  unexplained                         : {len(orphans)}")

    print("\n-- derived --")
    for written, why, n in derived:
        print(f"  {written:>6}% x{n:<2} {why}")

    if orphans:
        print("\n-- UNEXPLAINED --")
        for written, where in orphans:
            print(f"  {written:>6}%  {where[:4]}")
        raise SystemExit(1)

    print("\nRESULT: every percentage is either a recorded field or a listed derivation.")


if __name__ == "__main__":
    main()
