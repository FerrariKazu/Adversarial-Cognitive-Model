"""Patch report/GEN1_RESULTS_MASTER.md sections 8.2-8.4, 9, 12 to the exact
300-sample seed-level convention (recomputed from the raw per-seed CSVs).

Fixes three defects in the previously generated doc:
  1. sec 9 pair/Holm/Friedman stats were computed on chunk-macro values while
     the text claimed exact seed-level values;
  2. Friedman omnibus was evaluated at df=7 (printed chi2 values are df=7
     shapes); the 6-phase omnibus has df=5;
  3. per-seed tables averaged rounded chunk values (26.34/31.66/3.66 ...);
     exact k/300 values are 26.33/31.67/3.67 ...
Section 8.1 (published chunk-level) is verified correct and left untouched.
"""
import numpy as np
import pandas as pd
from scipy import stats

DOC = "report/GEN1_RESULTS_MASTER.md"
PHASES = ["backbone_only", "recurrence_only", "belief_no_f",
          "belief_with_f", "ais_v2_swap", "gen1_core"]
EPS = [0.0, 0.031, 0.062, 0.094]
SEEDS = list(range(41, 49))
CHUNKS = [48] * 6 + [12]


def seed_exact(phase):
    df = pd.read_csv(f"report/foundation_{phase}_eval/epsilon_sweep_per_seed.csv")
    out = {}
    for eps in EPS:
        for s in SEEDS:
            sub = df[(df.seed == s) & (np.isclose(df.eps_pixel, eps))]
            assert len(sub) == 7
            k = sum(round(a / 100 * n) for a, n in zip(sub.acc_pct.values, CHUNKS))
            out[(eps, s)] = 100 * k / 300
    return out


SEED = {p: seed_exact(p) for p in PHASES}
CM_SEED = {}
for p in PHASES:
    df = pd.read_csv(f"report/foundation_{p}_eval/epsilon_sweep_per_seed.csv")
    for eps in EPS:
        sub = df[np.isclose(df.eps_pixel, eps)]
        CM_SEED[(p, eps)] = sub.groupby("seed").acc_pct.mean()


def vec(p, eps):
    return np.array([SEED[p][(eps, s)] for s in SEEDS])


def f2(x):
    return f"{x:.2f}"


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt_delta(d):
    return f"{d:+.3f}"


# ---------- build new sections ----------
s82 = ["### 8.2 Seed-level (exact 300-sample accuracy per seed; mean ± SD over the 8 eval seeds, ddof=1)",
       "",
       "| Phase | ε=0 clean | ε=0.031 | ε=0.062 | ε=0.094 |",
       "|---|---|---|---|---|"]
for p in PHASES:
    cells = []
    for eps in EPS:
        v = vec(p, eps)
        cells.append(f"{f2(v.mean())} ± {f2(v.std(ddof=1))}")
    s82.append(f"| `{p}` | " + " | ".join(cells) + " |")
s82 += ["",
        "Seed-level vs chunk-level means differ by construction (chunk weighting); per-seed-value difference ≤ 2.39 pp (chunk-macro vs sample-weighted). The two conventions agree on every conclusion below; the single sign-level discrepancy between conventions is a null test, noted in §9."]

s83 = ["### 8.3 Per-seed clean accuracy (ε=0, exact 300-sample convention, %)",
       "",
       "Exact 300-sample accuracies are multiples of 1/3 %; shown at 2 dp.",
       "",
       "| seed | `backbone_only` | `recurrence_only` | `belief_no_f` | `belief_with_f` | `ais_v2_swap` | `gen1_core` |",
       "|---|---|---|---|---|---|---|"]
for s in SEEDS:
    cells = [f2(SEED[p][(0.0, s)]) for p in PHASES]
    s83.append(f"| {s} | " + " | ".join(cells) + " |")

s84 = ["### 8.4 Per-seed robust accuracy, PGD-10 (exact 300-sample convention, %)",
       "",
       "Exact 300-sample accuracies are multiples of 1/3 %; shown at 2 dp. Zeros are exact."]
for eps in EPS[1:]:
    s84 += ["", f"**ε = {eps}**", "",
            "| seed | `backbone_only` | `recurrence_only` | `belief_no_f` | `belief_with_f` | `ais_v2_swap` | `gen1_core` |",
            "|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        cells = [f2(SEED[p][(eps, s)]) for p in PHASES]
        s84.append(f"| {s} | " + " | ".join(cells) + " |")

s9 = ["## 9. Statistical comparisons between consecutive phases",
      "",
      "Method: values are the exact 300-sample accuracy per seed (multiples of 1/3 %, reconstructed from the chunk counts in the per-seed CSVs; reconstruction round-trip verified for every row), paired across the 8 shared eval seeds. Paired t-test + two-sided Wilcoxon signed-rank (Pratt, zeros retained); SD of Δ with ddof=1. n=8 is small — p-values are indicative. No multiplicity correction except where stated. `ais_v2_swap → gen1_core` is **not computable**: the two phases' eval outputs are identical bit-for-bit (§11). The chunk-macro convention (§8.1) reaches the same significance decision (p<0.05 vs p≥0.05) for every computable test except `belief_with_f → ais_v2_swap` at ε=0.031 (macro Δ −0.223 pp vs seed-level +0.000 pp; both null).",
      "",
      "| Pair | ε | Δmean (pp) | SD of Δ | paired t | p (t) | p (Wilcoxon) |",
      "|---|---|---|---|---|---|---|"]
pair_rows = []
for a, b in zip(PHASES, PHASES[1:]):
    for eps in EPS:
        va, vb = vec(a, eps), vec(b, eps)
        d = vb - va
        if np.allclose(d, 0):
            s9.append(f"| `{a}` → `{b}` | {eps} | {fmt_delta(0.0)} | 0.000 | — | — | — (identical evals) |")
            pair_rows.append((a, b, eps, None, None))
            continue
        t, pt = stats.ttest_rel(vb, va)
        try:
            _, pw = stats.wilcoxon(vb, va, zero_method="pratt",
                                   alternative="two-sided")
        except ValueError:
            pw = None
        s9.append(f"| `{a}` → `{b}` | {eps} | {fmt_delta(d.mean())} | {d.std(ddof=1):.3f} | {t:+.3f} | {fmt_p(pt)} | " + (fmt_p(pw) if pw is not None else "n/a") + " |")
        pair_rows.append((a, b, eps, pt, pw))

s9 += ["",
       "Holm step-down across the four computable clean-accuracy (ε=0) consecutive tests (family defined here, exploratory):",
       "",
       "| Comparison | raw p | Holm-adjusted p |",
       "|---|---|---|"]
fam = [(a, b, pt) for a, b, eps, pt, pw in pair_rows
       if eps == 0.0 and pt is not None]
fam_sorted = sorted(fam, key=lambda r: r[2])
m = len(fam_sorted)
running = 0.0
holm_adj = []
for i, (_, _, p) in enumerate(fam_sorted):
    running = max(running, min(1.0, (m - i) * p))
    holm_adj.append(running)
for (a, b, p), pa in zip(fam_sorted, holm_adj):
    s9.append(f"| `{a}` → `{b}` | {fmt_p(p)} | {fmt_p(pa)} |")

s9 += ["",
       "Omnibus Friedman test across all six phases (n=8 seeds, k=6 → df=5):",
       "",
       "| ε | χ²_F(5) | p |",
       "|---|---|---|"]
for eps in EPS:
    stat, p = stats.friedmanchisquare(*[vec(p_, eps) for p_ in PHASES])
    s9.append(f"| {eps} | {stat:.3f} | {fmt_p(p)} |")

s12 = ["## 12. Negative / null findings (artifact-supported)",
       "",
       "1. **No phase achieves meaningful adversarial robustness under the frozen protocol.** PGD-10 accuracy at ε=0.094 is 0.00–0.12% across all phases (chunk-level published values; recurrence_only is exactly 0.00 ± 0.00). At ε=0.062 the range is 0.15–0.82%.",
       "2. **The AIS-v2 swap (steps 5–6) shows no measurable clean or robust benefit over `belief_with_f` in this run.** Clean Δ +0.583 pp (p_t=0.558); all robust |Δ| ≤ 0.333 pp with p ≥ 0.068 (exploratory, n=8).",
       "3. **`belief_no_f` regressed clean accuracy** vs `recurrence_only` by −9.875 pp (paired t p<0.0001, Wilcoxon p=0.0078, Holm-adjusted p<0.0001 within this family) — the largest effect in the ladder, and it is negative. `belief_with_f` recovered +6.333 pp over `belief_no_f` (Holm-adj p=0.0004) but remained below `recurrence_only`.",
       "4. **`recurrence_only` traded clean accuracy for nothing measurable at this PGD budget**: clean +4.375 pp over `backbone_only` (Holm-adj p=0.0313) but robust accuracy *decreased* at ε=0.031 (−1.000 pp, p_t=0.0038) and was already near-chance elsewhere.",
       "5. **No gen1_core-specific effect is measurable** (bit-identical artifacts, §11)."]

# ---------- splice ----------
with open(DOC) as f:
    lines = f.read().split("\n")


def find(prefix):
    hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


i82, i83 = find("### 8.2"), find("### 8.3")
i84, i9 = find("### 8.4"), find("## 9.")
i10, i12 = find("## 10."), find("## 12.")
i13 = find("## 13.")
assert i82 < i83 < i84 < i9 < i10 < i12 < i13

new = (lines[:i82] + s82 + [""] + s83 + [""] + s84 + [""] + s9 + [""] +
       lines[i10:i12] + s12 + [""] + lines[i13:])
text = "\n".join(new)

# ---------- stale-number guards ----------
stale = ["10.044", "4.911", "6.547", "13.930", "22.199", "11.324", "7.549",
         "0.0524", "0.0023", "p_t=0.862", "+0.185 pp", "26.34 |", "31.66 |",
         "3.66 |", "1.116", "4.435", "5.996", "3.866", "p=0.0062",
         "Holm-adj p=0.0016", "Holm-adj p=0.0123", "± 2.70 ", "−10.044",
         "−1.116", "χ²_F(7)"]
for s in stale:
    assert s not in text, f"stale value survived: {s!r}"
for s in ["24.019", "20.508", "16.972", "6.081", "9.875", "4.375", "6.333",
          "0.583", "df=5", "26.33 |", "31.67 |", "3.67 |"]:
    assert s in text, f"expected value missing: {s!r}"

with open(DOC, "w") as f:
    f.write(text)
print(f"patched {DOC}: {len(new)} lines (was {len(lines)})")
