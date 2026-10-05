# Stage 4-E1: Pre-Registration

**Date frozen:** 2026-08-27 (before any training)
**Model:** E1 (AIS-v1 + HPC-L1 + Recon-Mod)
**Reference:** D (AIS-v1 + HPC-L1, recon-mod OFF) — validated in Stage 3

---

## 1. Ground Truth (DO NOT REINTERPRET)

D (AIS-v1 + HPC-L1, recon-mod OFF) is the validated reference:
- PGD-100 @ ε=0.094, 8 seeds (41–48), datasets==4.7.0 pinned
- D = 34.38 ± 1.94%, A(baseline) = 22.83 ± 3.47%, Δ = +11.54 pp
- Paired t(7) = 10.11, p ≈ 2 × 10⁻⁵

Recon-mod's known behavior (Stage 2 isolation, confirmed causal):
- When active, recon-mod reliably displaces truck from Π_D top-2 (car/truck → car/airplane)
- Independent of which other mechanism it is layered onto
- Never tested alongside BOTH AIS-v1's halting AND HPC simultaneously

---

## 2. Primary Comparison

| Parameter | Value |
|-----------|-------|
| Comparison | E1 vs D (co-evaluated, same session) |
| Primary metric | PGD-100 accuracy |
| Primary epsilon | ε = 0.094 |
| Baseline dataset dep | datasets == 4.7.0 |
| Significance criterion | Δ > 2 · σ_combined (project standard) |
| Seed count | **16** (41–56) — pre-decided, not reactive |
| n_samples per seed | 300 |
| pgd_steps | 100 |

### Seed count rationale

D-vs-A's effect (+11.54 pp) was large and easily detected at 8 seeds.
E1-vs-D is very likely a **smaller** marginal effect (adding one component
on top of an already-strong combination). A smaller effect against similar
baseline variance needs MORE seeds to reliably detect. 16 seeds from the
start; do NOT extend further after seeing results.

---

## 3. Failure Criteria (Pre-Registered)

Any of the following is a legitimate, reportable outcome — not a reason
to retune and rerun:

| Criterion | Threshold | Type |
|-----------|-----------|------|
| E1 does not beat D | Δ ≤ 2 · σ_combined | Primary |
| Clean accuracy drops | > 3 pp below D's | Safety |
| Masking gap exceeds | 1.0 pp (PGD-100 vs PGD-50) | Safety |

---

## 4. Mechanistic Hypotheses

### H1a (Accuracy)
Recon-mod improves PGD-100 robustness beyond D.

### H1b (Π_D reordering — specific, falsifiable)
**Prediction:** E1 will reproduce the car/airplane Π_D top-2 pattern
(truck displaced) that recon-mod produced every time it was active in
Stage 2.

- If E1 DOES show car/airplane top-2: confirms recon-mod's effect is
  robust to the presence of both AIS and HPC simultaneously — expected,
  not a new finding, but worth confirming.
- If E1 does NOT show this reordering: the interesting outcome — it would
  mean AIS+HPC's combined dynamics interact with recon-mod differently
  than either did alone, and warrants bounded isolation.

### H1c (Belief drift)
Measure D AND E1 together, same image set, same session.
No directional prediction pre-registered — this is genuinely new information.

---

## 5. Config Change (E1 vs D)

| Parameter | D | E1 |
|-----------|---|----|
| enable_ais | True | True |
| ais_halt_enabled | True | True |
| **ais_precision_recon_enabled** | **False** | **True** ← THE CHANGE |
| enable_hpc | True | True |
| hpc_num_levels | 1 | 1 |
| hpc_error_weight | 0.10 | 0.10 |

**E1 = D with recon-mod re-enabled.** This is the ONLY config change.

---

## 6. Training Protocol

- **Base checkpoint:** D's best (`rhan_next_ais_hpc_best.pth`)
- **Load mechanism:** D's checkpoint already contains `precision_modulator.gain`
  (the recon-mod flag is trainer-side, not architecture-side; 0 missing keys,
  0 unexpected keys when loaded into E1). E1 starts from D's fully-trained
  weights.
- **Curriculum:** Identical to D (1–20 @ ε=0.031, 21–40 @ ε=0.062,
  41–60 @ ε=0.094)
- **Max epochs:** 60
- **Resume-safe:** `--force-restart` mandatory (code has changed since D's
  training; the commit guard would otherwise block resume)
- **HF sync:** Yes
- **Never --force-restart without justification:** the only justification
  is the code-identity guard (Stage 3 lesson)

### Smoke phase (15 epochs, ε=0.031)
Health gates:
- HPC error trend: declining
- Truck-rank WATCH (non-blocking, logged)
- Gradient-flow check on recon-mod's reactivated path
- disable-recon-mod-reproduces-D backward-compat check

If smoke fails: STOP, diagnose, do not patch blind.

---

## 7. Evaluation Protocol

- **Co-evaluated:** D and E1 in the same eval session, same invocation
- **Seeds:** 16 (41–56)
- **PGD-100 primary, PGD-50 secondary**
- **Lens:** belief drift (D vs E1, same images), gaze trajectory, Π_D
  per-class trajectory including H1b reordering check

---

## 8. Secondary Metrics

| Metric | Purpose |
|--------|---------|
| PGD-50 (masking check) | Verify no gradient masking |
| Clean accuracy | Safety: should not drop >3 pp below D |
| Belief drift | D vs E1, same images — genuinely new |
| Per-class Π_D trajectory | H1b reordering check |
| Gaze trajectory | Verify gaze policy still active |
| HPC error trend | Confirm HPC head still learning |

---

## 9. Deliverable Structure (14 sections)

1. Objective
2. Frozen preregistration (this document)
3. Config
4. Training health
5. Clean performance
6. PGD-100 results (primary)
7. PGD-50 results (masking)
8. D vs E1 comparison (co-evaluated, paired test)
9. Π_D / H1b reordering verdict
10. Belief-drift analysis (D vs E1)
11. Failure modes
12. Limitations
13. Final verdict (two questions: did E1 beat D statistically, AND does
    the internal mechanism behave as H1b predicted)
14. E2 gate status (locked/unlocked and why)

**Internal consistency rule:** Before writing any "Δ vs D" number, re-derive
it by subtraction from the table's own D and E1 rows. If it doesn't match
a simple subtraction, that is the Stage-3 bug recurring — stop and fix.

---

## 10. E2 Gate Status

**E2 (+ SBR) remains fully locked** — scaffolded, not trained, not evaluated —
until E1 passes its own gate. If E1 fails or is CONDITIONAL, E2 does not start;
return to diagnosing E1 first.

---

## Discipline (carried forward unchanged)

- No cherry-picked seeds
- No post-hoc metric changes
- No silent dependency changes
- Preserve negative results
- Treat FAIL as a legitimate outcome
