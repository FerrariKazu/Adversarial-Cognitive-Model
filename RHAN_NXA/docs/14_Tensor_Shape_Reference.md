# 14 — Tensor Shape Reference

*Level 2 reading. The authoritative shapes table; anything not
specified by the plan is marked UNKNOWN, not invented.*

---

## Notation table

```text
B               batch size
C               number of classes
D_z             global embedding dimension
D_s             structural embedding dimension
K               number of AIS candidate locations
T               number of glimpses
K_structure     number of structural slots (kept distinct from K on purpose)

z_t             (B, D_z)
S_t             None | (B, K_structure, D_s)
U_t/e_t         (B, C)
A_t             gaze history + current glimpse index
```

> **Notation warning (deliberate):** the plan's own notation uses `K`
> for *both* the AIS candidate count and, in the belief-state type
> signature, the structural slot dimension (`S_t: Optional[(B, K,
> D_s)]`). These are **unrelated quantities** — the AIS candidate
> count (4–8) has nothing to do with any future slot count. This
> documentation writes the structural one as `K_structure` to keep
> them from being confused; when reading `MASTER_PLAN.md` verbatim,
> disambiguate by context. If the two ever coexist in code, they must
> use distinct names.

## Per-component shapes

| Tensor | Shape | Notes |
|---|---|---|
| `z_t` | `(B, D_z)` | pooled CLS-equivalent output of the within-glimpse recurrent transformer; dense, continuous, differentiable |
| `S_t` | `None` or `(B, K_structure, D_s)` | **None in the Gen-1 core build**; when present, an instance of the existing BeliefState ABC's structured variant |
| `e_t` | `(B, C)` | Dirichlet evidence, non-negative via softplus |
| `alpha_t` | `(B, C)` | `e_t + 1` |
| `U_scalar` | `()` / per-sample scalar | `C / sum(alpha_t)` |
| predicted/observed glimpse features | token-level, same format as the encoder's patch-embedding output | exact token grid/shape = substrate-dependent; **UNKNOWN in this documentation scope** (the plan fixes the *space*, not the numbers) |
| `E_t` | same space as the predicted/observed features | computed as mismatch between non-detached prediction and detached observation |
| `A_t` | `gaze_history: list of (B, 2)` up to length `T`, plus `current_glimpse_idx: int` | a structured record; not a learned tensor |
| candidate scores (AIS) | `(B, K)` | one score per candidate; produced by the shared predictor + Dirichlet-entropy scoring |
| selection weights (training) | `(B, K)` | soft distribution over candidates (e.g. Gumbel-softmax / straight-through); argmax at inference |
| classification readout | `(B, C)` logits from the evidential head | readout of the final belief |

## Shape walk of one glimpse (batch `B` fixed)

```text
glimpse extraction      → image crop at gaze          (B, C_in, h, w)   [substrate-specific]
compact ViT embed       → token features              (B, N_tokens, D_z) [N_tokens substrate-specific]
tied refinement ×(2–3)  → refined token features      (B, N_tokens, D_z)
pool                    → z candidate                 (B, D_z)
predictor at gaze a     → predicted local features    (B, N_tokens, D_z)  [same format as embed]
observed local features → (detached)                  (B, N_tokens, D_z)
E_t                     → error signal                (B, N_tokens, D_z)
UpdateNet(z_t, E_t)     → update term                 (B, D_z)
z_{t+1} = z_t + Π_t · update                          (B, D_z)
evidential head         → e_t, alpha_t                (B, C)
AIS candidate scoring   → scores                      (B, K)
selection               → next gaze                   (B, 2) coordinates; recorded in A_t
```

Only `z_t`, `e_t/alpha_t`, the candidate scores, and `A_t` are
specified at numeric-precision level by the plan. Token-grid sizes,
image resolutions, and crop sizes are substrate parameters — do not
copy numbers from this document into code; take them from the
substrate's configuration when it exists.

## UNKNOWN / not-specified in this scope

| Quantity | Status |
|---|---|
| numeric value of `D_z` | UNKNOWN here — locked once the substrate (Agent C) sets it |
| numeric value of `D_s` | n/a while `S_t = None`; deferred with structure |
| `N_tokens`, resolutions, crop sizes | UNKNOWN here — substrate parameters |
| exact within-glimpse iteration count | 2–3 range specified; exact value a substrate/implementation decision |
| loss terms and their weights | outside this documentation pass (master-plan Section 8+) |

## Consistency rules these shapes enforce

- The predictor's output shape **must equal** the observed
  patch-embedding shape — that equality *is* the "comparable space by
  construction" guarantee (`08_Prediction_Error.md`).
- `E_t`'s shape equals the predictor's output shape by construction;
  `UpdateNet` — not shape arithmetic — is what maps it into `z_t`'s
  update space.
- `A_t`'s history length grows by one per glimpse and never exceeds
  `T`.
- Any code that lets a structural slot count share a variable name
  with the AIS candidate count violates this reference.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A — BeliefState (all shapes and types verbatim in
> `MASTER_PLAN.md`); Part 1.B (E_t space); Part 1.E (K = 4–8). The
> `K_structure` renaming is a documentation-side disambiguation,
> flagged as such.
