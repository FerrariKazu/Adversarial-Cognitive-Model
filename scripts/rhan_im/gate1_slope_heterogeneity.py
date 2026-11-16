#!/usr/bin/env python3
"""Gate 1, REPORT-ONLY (change C4): how much does the per-position z-ROC slope vary across the token map?

Reads the run JSON(s) written by gate1_measure_dprime.py (it needs nothing else) and reports, per cell of `native`:
  * the observed map: number of measured positions, lower median (the verdict's convention, torch.median), ordinary median, mean, sd, min, max,
    the fraction of positions outside the slope band [0.8, 1.25], and the medians of the centre block, the outer ring and the four quadrants;
  * an INDEPENDENT-SAMPLE GAUSSIAN REFERENCE: the same statistics for maps simulated from an equal-variance Gaussian sensor (slope 1 by construction),
    at the observed per-position d' and positive count, with `reference_n_null` null samples per position.

HOW TO READ THE REFERENCE. It is an OPTIMISTIC FINITE-SAMPLE REFERENCE: it reproduces only the sampling noise of estimating a slope from n_pos
positives, with all samples independent. It does not reproduce the spatial, class or scene dependence of the real sensor, so real scatter above it is
expected and does not by itself mean the sensor is unequal-variance; it is NOT the calibrated noise floor of the sensor. Nothing here is a threshold,
a test or a confidence interval; no gate decision reads this file. `reference_n_null` is an assumption (the run JSON does not store the null counts).

    python scripts/rhan_im/gate1_slope_heterogeneity.py --run run17.json --run run42.json --run run99.json [--cells op|all] [--n-ref 100] \\
        [--reference-n-null 20000] [--seed 0] [--out slope_heterogeneity.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im import gate1 as g1  # noqa: E402
from rhan_im.sdt import zroc_slope  # noqa: E402

BAND = g1.DEFAULT_THRESHOLDS["zroc_slope"]
LABEL = ("optimistic finite-sample reference: independent Gaussian samples, slope 1 by construction; it does not reproduce the spatial / class / scene "
         "dependence of the real sensor and is not the sensor's calibrated noise floor")


def lower_median(t: torch.Tensor) -> float:
    """torch.median: for an even count the LOWER of the two middle values (the convention of the Gate-1 verdict)."""
    return float(t.median()) if t.numel() else float("nan")


def middle_median(t: torch.Tensor) -> float:
    return float(t.quantile(0.5)) if t.numel() else float("nan")           # linear interpolation: mean of the two middle values for an even count


def map_stats(m: torch.Tensor) -> Dict[str, float]:
    """Statistics of a (tok, tok) slope map; NaN = unmeasured position."""
    tok = m.shape[0]
    ok = torch.isfinite(m)
    v = m[ok]
    a, b = tok // 4, tok - tok // 4
    centre = torch.zeros_like(ok); centre[a:b, a:b] = True
    ring = torch.zeros_like(ok); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    h = tok // 2
    quad = {"top_left": (slice(0, h), slice(0, h)), "top_right": (slice(0, h), slice(h, tok)),
            "bottom_left": (slice(h, tok), slice(0, h)), "bottom_right": (slice(h, tok), slice(h, tok))}

    def sub(mask):
        return lower_median(m[mask & ok])
    out = {"n_measured": int(ok.sum()), "n_positions": int(m.numel()),
           "lower_median": lower_median(v), "middle_median": middle_median(v), "mean": float(v.mean()) if v.numel() else float("nan"),
           "sd": float(v.std()) if v.numel() > 1 else float("nan"), "min": float(v.min()) if v.numel() else float("nan"),
           "max": float(v.max()) if v.numel() else float("nan"),
           "fraction_outside_band": float(((v < BAND[0]) | (v > BAND[1])).float().mean()) if v.numel() else float("nan"),
           "centre_block_lower_median": sub(centre), "outer_ring_lower_median": sub(ring)}
    for name, (r, c) in quad.items():
        mask = torch.zeros_like(ok); mask[r, c] = True
        out[f"{name}_lower_median"] = sub(mask)
    return out


def reference(d_map: torch.Tensor, n_pos: torch.Tensor, n_null: int, n_ref: int, gen: torch.Generator) -> Dict[str, object]:
    """n_ref simulated maps; summarises the across-position sd, the fraction outside the band and the lower median of each map."""
    sds, outs, meds = [], [], []
    tok = d_map.shape[0]
    for _ in range(n_ref):
        m = torch.full((tok, tok), float("nan"))
        for i in range(tok):
            for j in range(tok):
                n, d = int(n_pos[i, j]), float(d_map[i, j])
                if n < 2 or not math.isfinite(d):
                    continue
                pos = d + torch.randn(n, generator=gen, dtype=torch.double)
                neg = torch.randn(n_null, generator=gen, dtype=torch.double)
                m[i, j] = zroc_slope(pos, neg)[0]
        s = map_stats(m)
        sds.append(s["sd"]); outs.append(s["fraction_outside_band"]); meds.append(s["lower_median"])

    def q(x):
        t = torch.tensor([v for v in x if math.isfinite(v)], dtype=torch.double)
        return {"p05": float(t.quantile(0.05)), "p50": float(t.quantile(0.5)), "p95": float(t.quantile(0.95))} if t.numel() else None
    return {"n_ref": n_ref, "n_null_per_position_assumed": n_null, "across_position_sd": q(sds), "fraction_outside_band": q(outs), "lower_median": q(meds), "label": LABEL}


def analyse_run(run: Dict, cells: str, n_ref: int, n_null: int, seed: int) -> Dict[str, object]:
    cfg = run["config"]
    op = f"size{cfg['op_size']}_alpha{cfg['alphas'][0]}"
    out: Dict[str, object] = {"seed": cfg.get("seed"), "checkpoint": run.get("trunk", {}).get("checkpoint"), "operating_cell": op, "cells": {}}
    gen = torch.Generator().manual_seed(seed)
    for key, cell in run["native"].items():
        if cells == "op" and key != op:
            continue
        nan = float("nan")
        slope = torch.tensor([[nan if x is None else x for x in row] for row in cell["zroc_slope"]], dtype=torch.double)
        n_pos = torch.tensor(cell["n_pos"], dtype=torch.double)
        d = torch.tensor([[nan if x is None else x for x in row] for row in cell["d_shift"]], dtype=torch.double)
        out["cells"][key] = {"observed": map_stats(slope), "reference": reference(d, n_pos, n_null, n_ref, gen)}
    out["verdict_uses_this"] = False
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, type=pathlib.Path)
    ap.add_argument("--cells", choices=("op", "all"), default="op")
    ap.add_argument("--n-ref", type=int, default=100)
    ap.add_argument("--reference-n-null", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    res = {"status": "REPORT-ONLY: no gate decision reads this", "reference_label": LABEL, "band": list(BAND),
           "median_convention": "lower_median = torch.median (lower of the two middle values for an even count), as in the verdict; middle_median is the ordinary median",
           "runs": [analyse_run(json.loads(p.read_text()), a.cells, a.n_ref, a.reference_n_null, a.seed) for p in a.run]}
    text = json.dumps(res, indent=2)
    if a.out:
        if a.out.exists():
            raise SystemExit(f"{a.out} exists: refusing to overwrite")
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
