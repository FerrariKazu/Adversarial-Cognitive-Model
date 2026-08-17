#!/usr/bin/env python3
"""Cross-mechanism Π_D comparison (Stage 2 diagnosis, 2026-08-15) — Step 2:
is the HPC smoke's truck-drop the SAME instance of the Stage-1 recon-mod
pattern, or a coincidence?

Stage-1 attribution (report/rhan_next_ais_v1_isolation_verdict.json):
    v12 reference            : car/truck top-2 (no pillars)
    AIS-v1 smoke (both on)   : car/airplane top-2 (truck 3rd)   -> BROKEN
    isoA (halting OFF, recon-mod ON) : car/airplane (truck 3rd) -> BROKEN
    isoB (recon-mod OFF)     : car/truck restored               -> RESTORED
=> recon-mod (a precision-modulated AUXILIARY loss weight) is the driver.

Hypothesis under test: truck is consistently the class that drops out of
Π_D top-2 whenever ANY auxiliary loss (recon-mod OR HPC) is active — a
shared root cause (per-class gradient/β_dynamic competition), not two
separate bugs. This script compiles the per-class Π_D (final epoch + last-2
epoch mean, per the isoA averaging note) from every local telemetry source
plus the recorded v4 smoke, and prints one comparison table.

Sources: local report/*.jsonl + the v4 final-epoch row recorded in the
2026-08-15 Colab log (hardcoded below, source-labeled).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
           'dog', 'horse', 'monkey', 'ship', 'truck']

# v4 final epoch (15) Π_D — transcribed from the 2026-08-15 Colab log
# (report/rhan_next_hpc_only_smoke_v4_diag.jsonl is HF-only; this row is the
# printed telemetry, source: user-supplied run log).
V4_FINAL = {'airplane': 0.3360, 'bird': 0.3039, 'car': 0.3425, 'cat': 0.2789,
            'deer': 0.2428, 'dog': 0.3121, 'horse': 0.2815, 'monkey': 0.2688,
            'ship': 0.2902, 'truck': 0.3237}

# isoB survives only as a reconstructed final-epoch top-2 (the original
# per-epoch JSONL was lost pre-HF-durability; see the file's own note).
ISOB_FINAL = {'car': 0.5149, 'truck': 0.4842}

# Documented v12 reference pattern (no numbers in the repo — top-2 only).
V12_REF_TOP2 = ['car', 'truck']

# AIS-v1 smoke: top-2 recorded in the isolation verdict (full per-class
# telemetry was not carried locally).
SMOKE_TOP2 = ['car', 'airplane']


def load_diag(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(name, mechanisms, pd, source, top2_only=False):
    """pd: dict class->value (full) or list of (class, value) pairs (top-2)."""
    if top2_only:
        top2 = sorted(pd.items(), key=lambda kv: -kv[1])[:2]
        truck_v = pd.get('truck')
        truck_rank = 1 if truck_v == top2[0][1] else (2 if truck_v == top2[1][1] else None)
        return {
            'name': name, 'mechanisms': mechanisms, 'source': source,
            'top2': [k for k, _ in top2], 'truck_rank': truck_rank,
            'truck': truck_v,
            'airplane': pd.get('airplane'), 'car': pd.get('car'),
            'full': None,
        }
    top2 = sorted(pd.items(), key=lambda kv: -kv[1])[:2]
    ranked = sorted(pd.items(), key=lambda kv: -kv[1])
    truck_rank = next((i + 1 for i, (k, _) in enumerate(ranked)
                       if k == 'truck'), None)
    return {
        'name': name, 'mechanisms': mechanisms, 'source': source,
        'top2': [k for k, _ in top2], 'truck_rank': truck_rank,
        'truck': pd.get('truck'), 'airplane': pd.get('airplane'),
        'car': pd.get('car'), 'full': pd,
    }


def main():
    runs = []

    # ── Stage 1: isoA (full telemetry), isoB (top-2 only) ───────────────────
    isoA_rows = load_diag(ROOT / 'report' / 'rhan_next_ais_v1_isoA_diag.jsonl')
    if isoA_rows:
        last = isoA_rows[-1]
        runs.append(summarize('isoA (halting OFF, recon-mod ON)',
                              'AIS: recon-mod only',
                              last.get('pi_d_per_class', {}),
                              f"{ROOT.name}/report/isoA_diag epoch {last['epoch']}"))
        # last-2 epoch mean (isoA averaging note: #2-slot is boundary-level).
        e13, e14 = isoA_rows[-2]['pi_d_per_class'], isoA_rows[-1]['pi_d_per_class']
        avg = {k: (e13[k] + e14[k]) / 2 for k in e13}
        runs.append(summarize('isoA last-2 avg (epochs 13-14)', 'AIS: recon-mod only',
                              avg, 'isoA_diag avg'))
    runs.append(summarize('isoB (recon-mod OFF, halting ON)',
                          'AIS: halting only (validated config)',
                          ISOB_FINAL, 'isolation verdict (reconstructed top-2)',
                          top2_only=True))
    runs.append(summarize('AIS-v1 smoke (both ON)', 'AIS: halting + recon-mod',
                          dict(SMOKE_TOP2 and {k: 1.0 for k in SMOKE_TOP2}),
                          'isolation verdict (top-2 only)', top2_only=True))

    # ── Stage 2: HPC smokes (full telemetry where local) ────────────────────
    for fname, label in [('rhan_next_hpc_only_smoke_diag.jsonl', 'HPC smoke v1'),
                         ('rhan_next_hpc_only_smoke_v3_diag.jsonl', 'HPC smoke v3')]:
        rows = load_diag(ROOT / 'report' / fname)
        if rows:
            last = rows[-1]
            runs.append(summarize(label, 'HPC: L_hpc (w=0.1), no AIS',
                                  last.get('pi_d_per_class', {}),
                                  f"{fname} epoch {last['epoch']}"))
            if len(rows) >= 2:
                e_a, e_b = rows[-2]['pi_d_per_class'], rows[-1]['pi_d_per_class']
                avg = {k: (e_a[k] + e_b[k]) / 2 for k in e_a}
                runs.append(summarize(f'{label} last-2 avg',
                                      'HPC: L_hpc (w=0.1), no AIS',
                                      avg, f'{fname} avg'))
    runs.append(summarize('HPC smoke v4 (final epoch 15)', 'HPC: L_hpc (w=0.1), no AIS',
                          V4_FINAL, 'Colab log 2026-08-15 (transcribed)'))

    # ── Reference ────────────────────────────────────────────────────────────
    runs.append({'name': 'v12 reference (no pillars)', 'mechanisms': 'none',
                 'source': 'documented Stage-1 reference',
                 'top2': V12_REF_TOP2, 'truck_rank': 2,
                 'truck': None, 'airplane': None, 'car': None, 'full': None})

    print("=" * 100)
    print("  Cross-mechanism Π_D comparison — truck's rank vs auxiliary losses")
    print("=" * 100)
    print(f"  {'run':<42}{'mechanisms':<30}{'Π_D top-2':<18}{'truck rank':<11}"
          f"{'truck':>7}{'airplane':>9}{'car':>8}")
    for r in runs:
        top2 = '/'.join(r['top2'])
        tr = r['truck_rank'] if r['truck_rank'] else '-'
        tv = f"{r['truck']:.4f}" if r['truck'] is not None else '  --  '
        av = f"{r['airplane']:.4f}" if r['airplane'] is not None else '  --  '
        cv = f"{r['car']:.4f}" if r['car'] is not None else '  --  '
        print(f"  {r['name']:<42}{r['mechanisms']:<30}{top2:<18}{tr:<11}"
              f"{tv:>7}{av:>9}{cv:>8}")
    print("=" * 100)

    # ── The shared-root-cause read ───────────────────────────────────────────
    print("\n  READ:")
    print("    • Auxiliary loss ACTIVE (recon-mod ON, or HPC ON):")
    for r in runs:
        if r['mechanisms'] not in ('none',) and r['truck_rank'] is not None:
            broken = 'BROKEN' if r['truck_rank'] not in (1, 2) else \
                     ('boundary' if (r['truck_rank'] == 3 and r['full']) else 'ok')
            print(f"      {r['name']:<42} truck rank={r['truck_rank']} "
                  f"top-2={r['top2']} -> {broken}")
    print("    • Auxiliary loss OFF (v12 reference, isoB): truck rank 1-2 "
          "(car/truck restored).")

    # Where full telemetry exists, print the per-class deltas that matter.
    print("\n  Per-class Π_D deltas (truck vs the #2 slot) on full telemetry:")
    for r in runs:
        if r['full'] is None or r['truck'] is None:
            continue
        top2v = sorted(r['full'].items(), key=lambda kv: -kv[1])[:2]
        gap = r['truck'] - top2v[1][1]
        print(f"    {r['name']:<42} truck={r['truck']:.4f} "
              f"#2={top2v[1][0]}={top2v[1][1]:.4f}  gap={gap:+.4f}")


if __name__ == '__main__':
    main()
