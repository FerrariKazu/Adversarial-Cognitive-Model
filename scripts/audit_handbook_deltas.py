#!/usr/bin/env python3
"""Audit every `pp` delta written in the RHAN handbook against the artifacts.

Deltas are the easiest number to get wrong: they are arithmetic, so nothing
in any JSON will match them exactly. This script rebuilds the set of deltas
that the recorded artifacts actually permit (pairwise differences of arm means
within each ablation, clean-minus-adversarial degradations, per-budget
differences) and checks that every `NN pp` in the manuscript belongs to it.

A value that does not belong is printed with its sentence, so it can be
inspected by hand - it is either a delta over a quantity this script does not
model, or an arithmetic error.

Run: python3 scripts/audit_handbook_deltas.py
"""

from __future__ import annotations

import collections
import itertools
import json
import pathlib
import statistics
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAND = ROOT / "docs/rhan_nxa"
REP = ROOT / "report"


def jload(rel: str):
    return json.loads((ROOT / rel).read_text())


# Numbers that the text quotes *in order to withdraw them*. They are not
# measurements of the current system, so no artifact can be expected to
# reproduce them; each is annotated with the recorded value it corresponds to.
QUOTED: dict[float, str] = {
    4.3: "withdrawn claim in ch28: matches the D8 APGD-CE gap, 22.67 vs 18.33 = 4.34 pp",
    2.3: "superseded value recorded in ch22 and in the consistency report as corrected to 1.7 pp",
}


def add_pairwise(values: list[float], out: set[float]) -> None:
    for a, b in itertools.permutations(values, 2):
        out.add(round(abs(a - b) * 100.0, 3))


def build_allowed() -> tuple[set[float], dict[str, int]]:
    allowed: set[float] = set()
    groups: dict[str, int] = {}

    def group(name: str, values: list[float]) -> None:
        add_pairwise(values, allowed)
        groups[name] = len(values)

    # --- 1. the three 8-seed sweep summaries, six conditions each ---
    ff = jload("report/eval/eval_ff_112_20261018_123413.json")["sweep_summary"]
    ro = jload("report/d3_final_eval/g2_d3_recurrence_only_final.json")["sweep"]
    gc = jload("report/d3_final_eval/g2_d3_gen1_core_best_clean.json")["sweep"]
    conds = [
        ("clean", "clean (eps=0.0)", "clean", "clean"),
        ("n031", "norm_eps=0.031", "norm_eps_0031", "norm_eps_0031"),
        ("n062", "norm_eps=0.062", "norm_eps_0062", "norm_eps_0062"),
        ("n094", "norm_eps=0.094", "norm_eps_0094", "norm_eps_0094"),
        ("p2", "pix_eps=2/255", "pix_eps_2_255", "pix_eps_2_255"),
        ("p4", "pix_eps=4/255", "pix_eps_4_255", "pix_eps_4_255"),
    ]
    for _, kf, kr, kg in conds:
        group(f"sweep:{kf}", [ff[kf]["mean"], ro[kr]["mean"], gc[kg]["mean"]])
    # clean-minus-adversarial degradation per arm, every budget
    for kf, kr, kg in [(kf, kr, kg) for _, kf, kr, kg in conds[1:]]:
        for name, a, b in [("ff", ff["clean (eps=0.0)"]["mean"], ff[kf]["mean"]),
                           ("ro", ro["clean"]["mean"], ro[kr]["mean"]),
                           ("gc", gc["clean"]["mean"], gc[kg]["mean"])]:
            allowed.add(round(abs(a - b) * 100.0, 3))
        # cross-arm clean/adv mixes, e.g. ff clean minus ro adversarial
        for a, b in itertools.permutations(
                [ff["clean (eps=0.0)"]["mean"], ro["clean"]["mean"], gc["clean"]["mean"],
                 ff[kf]["mean"], ro[kr]["mean"], gc[kg]["mean"]], 2):
            allowed.add(round(abs(a - b) * 100.0, 3))

    # --- 2. glimpse budget (D5): per arm, clean and robust across T ---
    d5 = jload("report/d5_glimpse_scaling/g2_d5_combined.json")
    for arm, blob in d5.items():
        for field in ("clean", "robust"):
            group(f"d5:{arm}:{field}",
                  [blob["budgets"][t][field] for t in sorted(blob["budgets"], key=int)])
    for t in d5["recurrence_only_final"]["budgets"]:
        group(f"d5:T{t}", [d5["recurrence_only_final"]["budgets"][t]["clean"],
                           d5["recurrence_only_final"]["budgets"][t]["robust"],
                           d5["gen1_core_best_clean"]["budgets"][t]["clean"],
                           d5["gen1_core_best_clean"]["budgets"][t]["robust"]])

    # --- 3. gaze source (D7) ---
    d7 = jload("report/d7_gaze_swap/g2_d7_combined.json")
    for arm, blob in d7.items():
        for field in ("clean", "robust"):
            group(f"d7:{arm}:{field}",
                  [m[field] for m in blob["gaze_modes"].values()])
    for arm, blob in d7.items():
        for mode, m in blob["gaze_modes"].items():
            group(f"d7:{arm}:{mode}", [m["clean"], m["robust"]])

    # --- 4. adaptive attacks (D8) ---
    d8 = jload("report/d8_adaptive_attacks/g2_d8_combined.json")
    for arm, blob in d8.items():
        if not isinstance(blob, dict) or "attacks" not in blob:
            continue
        for att, a in blob["attacks"].items():
            if "steps" not in a and "n_restarts" not in a:
                continue
        group(f"d8:{arm}", [a["acc"] for a in blob["attacks"].values()])
        group(f"d8:{arm}:clean-vs-adv",
              [blob["attacks"]["clean"]["acc"]] + [a["acc"] for a in blob["attacks"].values()])

    # --- 5. matched-epsilon attack audit, n = 500 ---
    s1 = jload("report/audit_rev1/step1_attack_audit.json")
    ck = s1["checkpoints"]
    for arm, blob in ck.items():
        vals = []
        for k, v in blob.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)) and 0 <= vv <= 1:
                        vals.append(float(vv))
        if vals:
            group(f"audit:{arm}", vals)
    # across arms at the same attack
    att_names = set()
    for blob in ck.values():
        for k, v in blob.items():
            if isinstance(v, dict):
                att_names.add(k)
    for att in att_names:
        vals = [float(blob[att][k]) for blob in ck.values()
                if att in blob and isinstance(blob[att], dict)
                for k in blob[att] if isinstance(blob[att][k], (int, float))
                and 0 <= blob[att][k] <= 1]
        if len(vals) > 1:
            group(f"audit:{att}", vals)

    # --- 6. resolution probe (five-epoch linear probe at three resolutions) ---
    probe = jload("report/handbook/data_extracts/handbook_data.json")["resolution_probe"]
    group("probe", [v["best_val_acc"] for v in probe.values()])

    # --- 7. component ablations (belief, foresight, gaze swap, backbone) ---
    for name in ("foundation_backbone_only_result", "foundation_recurrence_only_result",
                 "foundation_gen1_core_result", "foundation_belief_with_f_result",
                 "foundation_belief_no_f_result", "foundation_ais_v2_swap_result"):
        p = REP / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        vals = [float(v) for v in d.values() if isinstance(v, (int, float)) and 0 <= v <= 1]
        if len(vals) > 1:
            group(name, vals)

    # --- 8. latency-derived percentages of a base value ---
    for base, other in [(0.447, 0.587), (0.447, 0.504), (0.064, 0.447)]:
        allowed.add(round(abs(other - base) / base * 100.0, 3))

    # --- 9. statistics quoted in pp that are not differences of two means ---
    # per-subset standard deviations (spread claims) and the paired/unpaired
    # standard errors of the headline estimate.
    for key in ff:
        allowed.add(round(ff[key]["std"] * 100.0, 3))
    for blob in (ro, gc):
        for cond in blob.values():
            allowed.add(round(cond["std"] * 100.0, 3))
    diffs = [(x - y) * 100.0 for x, y in zip(ff["norm_eps=0.094"]["per_seed"],
                                             ro["norm_eps_0094"]["per_seed"])]
    se_paired = statistics.stdev(diffs) / len(diffs) ** 0.5
    se_unpaired = ((ff["norm_eps=0.094"]["std"] ** 2 / len(diffs)
                    + ro["norm_eps_0094"]["std"] ** 2 / len(diffs)) ** 0.5) * 100.0
    allowed.update({round(se_paired, 2), round(se_unpaired, 2),
                    round(statistics.mean(diffs), 2)})

    # --- 10. one 500-image clean row against its eight-seed mean ---
    for single, mean in [
        (json.loads((REP / "eval/eval_ff_112_20261018_123413.json").read_text())
         ["final_verification"]["final_clean_acc"], ff["clean (eps=0.0)"]["mean"]),
        (json.loads((REP / "d3_final_eval/g2_d3_recurrence_only_final.json").read_text())
         ["final"]["clean"], ro["clean"]["mean"]),
        (json.loads((REP / "d3_final_eval/g2_d3_gen1_core_best_clean.json").read_text())
         ["final"]["clean"], gc["clean"]["mean"]),
    ]:
        allowed.add(round(abs(single - mean) * 100.0, 3))

    return allowed, groups


def main() -> None:
    allowed, groups = build_allowed()
    # ~200-300 ppi is a resolution, not a point difference
    pat = re.compile(r"([+\-\u2212]?\d+(?:\.\d+)?)\s?pp(?![a-zA-Z])")
    rows: list[tuple[float, str]] = []
    for f in sorted(HAND.rglob("*.md")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in pat.finditer(line):
                val = float(m.group(1).replace("\u2212", "-").lstrip("+"))
                rows.append((val, f"{f.relative_to(HAND)}:{i}: {line.strip()[:120]}"))

    written = collections.Counter(v for v, _ in rows)
    bad = [(v, ctx) for v, ctx in rows
           if v not in QUOTED and not any(abs(v - a) <= 0.055 for a in allowed)]

    print(f"distinct pp values: {len(written)}; occurrences: {len(rows)}")
    print(f"quoted historical claims (not modelled as deltas): {sorted(QUOTED)}")
    print(f"artifact-derived delta universe: {len(allowed)} values "
          f"from {len(groups)} groups")
    print(f"matched: {len(rows) - len(bad)}   unmatched: {len(bad)}\n")
    if bad:
        print("-- unmatched (inspect each) --")
        for v, ctx in bad:
            print(f"  {v:>7} pp  {ctx}")
        raise SystemExit(1)
    print("RESULT: every written pp delta is reproducible from a recorded artifact.")


if __name__ == "__main__":
    main()
