"""Verify/correct every derived statistic in report/GEN1_RESULTS_MASTER.md.

Recomputes, from report/foundation_<phase>_eval/epsilon_sweep_per_seed.csv:
  - published chunk-macro summary (must equal summary_table.csv)
  - exact 300-sample per-seed accuracies (count reconstruction, round-trip checked)
  - seed-level mean +/- SD (ddof=1) per phase/eps
  - paired t + Wilcoxon (Pratt) between consecutive phases, seed-level
  - the same on chunk-macro values (to show convention does not flip signs)
  - Holm step-down over the 4 clean-accuracy consecutive tests
  - Friedman omnibus across 6 phases (df = k-1 = 5)
"""
import numpy as np
import pandas as pd
from scipy import stats

PHASES = ["backbone_only", "recurrence_only", "belief_no_f",
          "belief_with_f", "ais_v2_swap", "gen1_core"]
EPS = [0.0, 0.031, 0.062, 0.094]
SEEDS = list(range(41, 49))
CHUNKS = [48] * 6 + [12]  # 6x48 + 1x12 = 300; loader batch 48, n_samples 300

VALID_48 = {round(100 * k / 48, 2) for k in range(49)}
VALID_12 = {round(100 * k / 12, 2) for k in range(13)}


def load(phase):
    return pd.read_csv(f"report/foundation_{phase}_eval/epsilon_sweep_per_seed.csv")


def chunk_macro(phase):
    """Published convention: mean/SD over the 56 chunk rows."""
    df = load(phase)
    out = {}
    for eps in EPS:
        sub = df[np.isclose(df.eps_pixel, eps)]
        assert len(sub) == 56, (phase, eps, len(sub))
        assert set(sub.acc_pct) <= (VALID_48 | VALID_12), "chunk value not k/48 or k/12"
        out[eps] = (sub.acc_pct.mean(), sub.acc_pct.std(ddof=1))
    return out


def seed_exact(phase):
    """Exact 300-sample accuracy per seed, reconstructed from chunk counts."""
    df = load(phase)
    out = {}  # (eps, seed) -> acc pct, exact multiple of 1/3
    for eps in EPS:
        for s in SEEDS:
            sub = df[(df.seed == s) & (np.isclose(df.eps_pixel, eps))]
            assert len(sub) == 7, (phase, eps, s, len(sub))
            k_tot = 0
            for acc, n in zip(sub.acc_pct.values, CHUNKS):
                k = round(acc / 100 * n)
                assert abs(acc - round(100 * k / n, 2)) < 1e-9, (phase, acc, n)
                k_tot += k
            out[(eps, s)] = 100 * k_tot / 300
    return out


print("== chunk-macro vs summary_table.csv ==")
for p in PHASES:
    cm = chunk_macro(p)
    st = pd.read_csv(f"report/foundation_{p}_eval/summary_table.csv")
    for eps in EPS:
        row = st[np.isclose(st.eps_pixel, eps)].iloc[0]
        assert abs(cm[eps][0] - row.acc_pct_mean) < 1e-9, (p, eps)
        assert abs(cm[eps][1] - row.acc_pct_std) < 1e-9, (p, eps)
print("OK: all 6 phases x 4 eps reproduce summary_table.csv exactly")

SEED = {p: seed_exact(p) for p in PHASES}

print("\n== seed-level mean +/- SD (ddof=1) ==")
for p in PHASES:
    cells = []
    for eps in EPS:
        v = np.array([SEED[p][(eps, s)] for s in SEEDS])
        cells.append(f"eps={eps}: {v.mean():.2f} +/- {v.std(ddof=1):.2f}")
    print(f"{p:16s} " + " | ".join(cells))

print("\n== per-seed clean (eps=0) ==")
for p in PHASES:
    print(f"{p:16s}", [f"{SEED[p][(0.0, s)]:.2f}" for s in SEEDS])

print("\n== consecutive paired tests (seed-level, exact) ==")
rows = []
for a, b in zip(PHASES, PHASES[1:]):
    for eps in EPS:
        va = np.array([SEED[a][(eps, s)] for s in SEEDS])
        vb = np.array([SEED[b][(eps, s)] for s in SEEDS])
        d = vb - va
        if np.allclose(d, 0):
            rows.append((a, b, eps, 0.0, 0.0, None, None, None, None))
            print(f"{a:16s}->{b:16s} eps={eps:<5} delta=0.000 (identical evals)")
            continue
        t, pt = stats.ttest_rel(vb, va)
        try:
            w, pw = stats.wilcoxon(vb, va, zero_method="pratt",
                                   alternative="two-sided")
        except ValueError:
            w, pw = None, None
        rows.append((a, b, eps, d.mean(), d.std(ddof=1), t, pt, w, pw))
        print(f"{a:16s}->{b:16s} eps={eps:<5} d={d.mean():+.3f} "
              f"sd={d.std(ddof=1):.3f} t={t:+.3f} p_t={pt:.4f} p_w="
              + (f"{pw:.4f}" if pw is not None else "n/a"))

print("\n== same tests on chunk-macro values (sign check) ==")
CM = {p: chunk_macro(p) for p in PHASES}
# per-seed chunk-macro values = mean of that seed's 7 chunk rows
CM_SEED = {}
for p in PHASES:
    df = load(p)
    for eps in EPS:
        sub = df[np.isclose(df.eps_pixel, eps)]
        CM_SEED[(p, eps)] = sub.groupby("seed").acc_pct.mean()
sign_flips = 0
for a, b in zip(PHASES, PHASES[1:]):
    for eps in EPS:
        va, vb = CM_SEED[(a, eps)], CM_SEED[(b, eps)]
        d = (vb - va).values
        t, pt = stats.ttest_rel(vb, va)
        m = [r for r in rows if r[0] == a and r[1] == b and r[2] == eps][0]
        if m[6] is not None and pt is not None:
            if np.sign(d.mean()) != np.sign(m[3]) and abs(d.mean()) > 1e-9:
                sign_flips += 1
                print(f"SIGN FLIP {a}->{b} eps={eps}: macro {d.mean():+.3f} "
                      f"vs seed {m[3]:+.3f}")
            if (pt < 0.05) != (m[6] < 0.05):
                print(f"p<0.05 flip {a}->{b} eps={eps}: macro p={pt:.4f} "
                      f"seed p={m[6]:.4f}")
print(f"sign flips macro vs seed-level: {sign_flips}")

print("\n== Holm step-down, clean (eps=0) consecutive family, seed-level ==")
fam = [(r[0], r[1], r[6]) for r in rows if r[2] == 0.0 and r[6] is not None]
fam_sorted = sorted(fam, key=lambda r: r[2])
m = len(fam_sorted)
adj = [None] * m
running = 0.0
for i, (a, b, p) in enumerate(fam_sorted):
    running = max(running, min(1.0, (m - i) * p))
    adj[i] = running
for (a, b, p), pa in zip(fam_sorted, adj):
    print(f"{a:16s}->{b:16s} raw p={p:.6f} Holm p={pa:.4f}")

print("\n== Friedman omnibus across 6 phases (df=5), seed-level ==")
for eps in EPS:
    cols = [np.array([SEED[p][(eps, s)] for s in SEEDS]) for p in PHASES]
    stat, p = stats.friedmanchisquare(*cols)
    print(f"eps={eps:<5} chi2_F(5)={stat:.3f} p={p:.4f}")

print("\n== max per-seed-value diff chunk-macro vs seed-level ==")
mx = 0.0
for p in PHASES:
    for eps in EPS:
        va = np.array([SEED[p][(eps, s)] for s in SEEDS])
        vb = CM_SEED[(p, eps)].values
        mx = max(mx, np.abs(va - vb).max())
print(f"max diff = {mx:.2f} pp")
