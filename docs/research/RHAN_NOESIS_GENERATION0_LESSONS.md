# RHAN / NOESIS — GENERATION 0 LESSONS

Generated 2026-09-15 · companion to `RHAN_NOESIS_PROJECT_EVIDENCE_DOSSIER.md` §6 (evidence table) and `RHAN_NOESIS_EXPERIMENT_REGISTRY.csv` (per-experiment rows). This file is the decision-oriented distillation: what Gen 0 actually established, mechanism by mechanism, with the evidence that licenses each verdict. No category is assigned "because it sounds reasonable" — every row carries its evidence pointer.

---

## 1. Lessons that changed the infrastructure (keep forever)

### L1. One mechanism at a time, with machine-checked isolation
Every validated result came from a single-mechanism intervention on a frozen, tested base (B from baseline, C from B, D from B+C, sbr-k from sbr-(k−1)). The one time multiple mechanisms shipped together in the v6 era (dynamic gating + predictive coding + ACT), the result was a regression recorded as such `[ART: NOESIS_FOUNDATION.md "The v6 regression"]`.
**Rule adopted:** on/off tests per mechanism, byte-identity backward-compat tests (`test_hpc_disable_backward_compat.py`, `test_config_backward_compat.py`), gradient-reachability tests (`test_gradient_flow.py`, `test_hpc_gradient_flow.py`). Evidence: tests pass at recorded dates + scaffold acceptance.

### L2. Per-group optimizer + per-group clip is mandatory for auxiliary losses
The Stage-2 starvation incident: w_hpc=0.1 through a shared-LR SGD with a global grad-clip dominated by the backbone TRADES gradient ≈ **1000× attenuation** of the HPC head's updates (measured: raw last-conv grad 0.004 vs 0.54 isolated; per-step |dW| 1.4e-5). Fix (two-group SGD, hpc lr ×6.67, per-group clip) produced a measured **201.8×** movement improvement and a passed pre-flight `[ART: optimizer_configuration_fix_2026_08_13]`.
**Rule adopted:** every new loss-bearing module gets its own optimizer group, a |dW| pre-flight, and a resume guard (group-count/LR-ratio mismatch refuses restore). This is Gen-0's `gen0` stage and it gates everything after.

### L3. Gates must be falsifiable in both directions
The sbr1 symmetric-band collapse gate (|clean − D| ≤ 3pp) **failed the real run for over-performing** (62.49 clean vs D's 54.96) — a strictly better model rejected by a gate whose purpose was collapse detection. Amendment to a one-sided rule was recorded with the original preserved in git history `[ART: sbr1 gate_amendment]`.
**Rule adopted:** every gate states what passes AND what a false rejection looks like; amendments are dated, justified, and never silent.

### L4. Pre-register the criterion, then honor null results
The 2σ conservative criterion (δ > 2σ_combined at ε=0.094) was fixed before runs. Under it: D cleared it (CROSSOVER REAL), C cleared it, B did not (+8.5 vs 8.84 threshold), sbr3 did not (+3.46 vs 7.32). All four verdicts stand as recorded; nulls are outcomes, not embarrassments — the roadmap says this verbatim in at least three stage entries.
**Consequence:** several "not significant" verdicts are power artifacts (n=5–8, seed σ≈2.4–3.2pp ⇒ detection floor ≈8pp). n=16 is now the ladder default.

### L5. Robustness claims need a masking guard
PGD-50→100 gap ≤ 1.0 pp = genuine; ≤2.5 = borderline (nondeterminism floor ~1.5pp); >2.5 = masking risk. Every validated model passed or was flagged (sbr3: −1.1pp → borderline, flagged 2026-09-15). The Athalye lesson (corpus [8]) is operationalized, not just cited.

### L6. Provenance discipline: donor rows, byte verification, no silent re-eval
Comparator rows (baseline, D) are reused across sweeps as DONOR rows, byte-verified against their source CSV at load time (`scripts/consistency_assert.py`; the impossible post-hoc aggregate check was removed 2026-09-15 with regression tests). UNAVAILABLE comparator ⇒ explicit UNAVAILABLE cells, never fabricated (rule 1b, enforced by the report builder).
**Also:** every verdict records git SHA, checkpoint sha256, seeds, timestamps; best-vs-rolling state-dict verification exists because a session wipe once made the peak-val weights vanish while metadata claimed them (stage-1 `eval_target_note` — the 54.05% figure is banned from citation).

### L7. Diagnose before redesign: the isolation playbook
When the Stage-1 smoke health gate fired (Pi_D top-2 = car/airplane ≠ car/truck), the response was two bounded single-knockout arms (isoA halting-off, isoB recon-mod-off) with a pre-registered decision rule. Verdict: recon-mod was the driver (necessary + boundary-level sufficient); halting exonerated. The smoke→isoB contrast is the cleanest causal attribution in the record.

### L8. Negative results are deliverables
E2 (SBR legacy: clean −9.9, adv-neutral), E3 (T=6: no robustness gain), E1's robustness null for recon-mod, Shape-ResNet's phase-5 negative, CORnet-S's recurrence-alone failure, the sbr1 formula-artifact failure, the smoke gate firing — all recorded with numbers and left in the lineage. The dossier's §6 table encodes them as RETIRE/DEFER without narrative laundering.

---

## 2. Mechanism verdicts (the evidence-backed scorecard)

| Mechanism | Verdict | Key evidence | Gen-1 implication |
|---|---|---|---|
| **TRADES backbone + pseudo-labels** | KEEP (as reference/baseline) | 54.81 clean / 24.23 adv, 16 seeds, stable across sessions (with documented drift 20.4→23.71) | the floor; re-establish at ImageNet scale before any mechanism claims |
| **AIS-v1 halting-only (relocated Eq.-II)** | KEEP/MODIFY | +8.5pp adv (n.s. at n=8), masking-free, telemetry healthy | keep the mechanism; run D2 to test whether *genuine* EIG beats the relocated-gradient surrogate |
| **Recon-mod (precision-modulated recon)** | KEEP/MODIFY (optional clean knob) | clean +3.10 (16/16 seed wins), adv −0.9 n.s.; confirmed driver of Pi_D reordering in isolation | never in the headline config; consider as a clean-accuracy lever only |
| **Soft entropy-gated halting** | KEEP (re-evaluate at sbr4) | exonerated by isolation; no harm vs fixed-T | sbr4 replaces its signal (slot entropy → evidence uncertainty) — the interaction test is pending |
| **T=6 foraging** | RETIRE from headline | adv −3.25, clean +2.29, both n.s.; D wins 13/16 adv | depth is not the lever; defer as ablation |
| **HPC edge-map L1 (w=0.1, per-group opt)** | KEEP | standalone CROSSOVER REAL (+7.33 at n=5); contributes inside D | keep; D3 (belief target) will test the *target choice*, motivated by E1's Lens dilution finding |
| **Multi-level/orientation HPC** | DEFER | never trained (extractor exists unwired) | only after level-0 fully understood |
| **SBR legacy N=1 wiring** | REPLACE | adv-neutral, clean −9.9 (E2); wiring can't express slot metrics | spatial binding is the only SBR wiring going forward |
| **SBR spatial binding (sbr0-4)** | KEEP (with caveat) | sbr0 structural gate 4/4 (entropy 0.98, cosine declining, probes above floor, ablation benign); BUT probes ~0.50 uniform ⇒ specialization unproven; sbr2/sbr3 show a clean↔robustness trade (−10.7 then +4.4 recovered) | sbr4 decides whether the uncertainty signal is real; treat the trade-off curve as the result, not a single point |
| **Fixed-ε fine-tune as recovery phase** | KEEP | sbr3 recovered ~41% of sbr2's adv concession in 20 epochs | a genuinely useful training-pattern discovery; carry into Gen 1 curriculum design |
| **AIS-v2 genuine EIG (D2)** | DEFER → RUN | code complete, zero runs | the cheapest disambiguation of *why* AIS-v1 helps |
| **Belief-HPC (D3)** | DEFER → RUN | code complete, zero runs | tests the pixel-vs-belief target hypothesis from E1's Lens result |
| **IWM** | DEFER | scaffold only; out of Gen-0 scope by design | nothing depends on it yet |
| **Memory / distributional belief / self-monitoring** | DEFER | clusters NOT STARTED | uncertainty calibration is the natural entry (c7) once sbr4's signal exists |

---

## 3. Process lessons (how the record stayed trustworthy)

1. **State machine as single source of truth.** `scripts/stage_state_machine.py` + HF-synced roadmap means any Colab session death resumes exactly; hand-editing stage statuses is forbidden by the artifact itself. Cost: the checked-in JSON goes stale (dossier §18.2) — acceptable, but consumers must read runtime.
2. **Smoke gates with calibrated thresholds.** gaze-shift ≥0.05 (raised from 0.01 after measuring the dead state at ~0.007 and known-good at 0.28–0.39), steps-std ≥0.02, frac-halted ≥0.02 — numbers derived from calibration runs, not vibes. Gates fired once (smoke Pi_D) and correctly forced diagnosis before spend.
3. **Paired per-seed reporting.** 16 seeds × same 300-sample subsets ⇒ paired comparisons; win/loss counts (E1 16/16 clean) are more informative than mean±std alone. Keep and extend (sign tests / Wilcoxon belong in Gen-1 writeups; none were run in Gen 0 — the 2σ rule is a z-approximation).
4. **Compute accounting is real.** ~31.5 min/adv cell on T4 ⇒ ~8.5h per 16-seed leg ⇒ the eval, not the training, is the budget driver. Everything resume-safe (local CSV + HF merge + donor preservation) exists because sessions die.
5. **Honesty conventions are load-bearing.** The labeling rule (never call AIS-v1 "information gain"), the implementation-honesty field in the literature corpus (Direct/Adapted/Not implemented per paper), UNAVAILABLE cells in reports, and "FAIL is a complete, valid, reportable outcome" in gate notes — these are why the record can be audited at all.

---

## 4. What Gen 0 should hand to Gen 1 (decision summary)

1. **The headline:** D (clean 54.96±2.37, adv@0.094 **34.02±3.24** PGD-100, 16 seeds) > baseline (54.81 / 24.23) by ~9.8–11.6pp adv at ε=0.094 (2σ-clean, masking-free), at clean parity, with the ablation row (E1/E2/E3) explaining what does NOT contribute.
2. **The trade-off curve:** baseline → sbr1/sbr2 (clean) → sbr3 (recovery) is the honest Pareto picture; sbr4 + D2 + D3 complete it. Finish sbr2 seeds 52–56 for matched n before writing.
3. **The infrastructure:** multi-group optimizer, gate protocol, donor-verification, state machine — all directly reusable, all tested.
4. **The debts (fix before external eyes):** docstring param counts, stale Gen-1 report, the "3 seeds" eval note, gaze-correlation phrasing drift, and the sbr3 PGD-50<PGD-100 borderline that needs the borderline label rather than GENUINE in any generated report.
5. **The open question that matters most:** does the uncertainty signal (sbr4) separate clean from adversarial inputs? Every downstream design choice in Gen 1 (halting, abstention, selective classification) inherits from that answer. Log uncertainty distributions per condition in the sbr4 eval — it is the cheapest experiment with the largest information gain remaining in Gen 0.
