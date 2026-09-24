# 12 — Architecture Data Flow (Implementation View)

*Level 2 reading. For implementers: what consumes what, where module
boundaries are, and what each component may assume.*

---

## The core data flow

```text
image
  ↓
glimpse / visual observation
  ↓
compact ViT
  ↓
within-glimpse recurrent refinement        (tied block, 2–3 steps)
  ↓
local token/patch features
  ↓
global pooled representation z_t
  ↓
BeliefState B_t
  ├── U_t
  ├── E_t
  └── A_t
  ↓
prediction / candidate scoring
  ↓
AIS-v2
  ↓
next gaze
  ↓
next glimpse
  ↓
repeat
  ↓
classification readout
```

Make clear where each module begins and ends:

| Boundary | Before it | After it |
|---|---|---|
| glimpse extraction | the image | the crop/patch at the current gaze |
| compact ViT + refinement | glimpse pixels | token features + pooled `z` candidate |
| belief update (`F`) | current `B_t`, fresh `E_t` | next `B_{t+1}` |
| AIS-v2 | belief + candidate set | the next gaze location |
| readout | final `B_T` | label + evidence/uncertainty |

## Component contract table

| Component | Consumes | Produces | May assume | Forbidden from assuming |
|---|---|---|---|---|
| Compact ViT substrate | glimpse pixels | token features, pooled vector | fixed `D_z`; patch-embedding output is `E_t`'s target space | that it will see the whole image at once |
| Within-glimpse refinement | token features | refined token features | weights are tied across iterations | that iteration count changes parameter count |
| Shared glimpse predictor | belief summary + gaze location | predicted local (token) features | same encoder/format as observation | that it may be duplicated for AIS use — **one predictor, two consumers** |
| `E_t` computation | predicted features, observed features (detached) | error signal | observed side is ground truth | that `E_t` may be detached before the update |
| `UpdateNet` | `(z_t, E_t)` | the update to `z_t` | learned mapping; own optimizer group; pre-flight `|dW|` check | that `E_t` is directly addable to `z_t` |
| `U_t` (evidential head) | belief content | `e_t (B,C)`, `alpha_t`, uncertainty scalar | softplus non-negativity; `alpha = e + 1` | that it measures belief-content uncertainty (it is readout-level) |
| AIS-v2 | `U_t`, candidate locations, shared predictor | next gaze; selection gradients (training) | K cheap predictor passes, never K backbone passes | that center-biased behavior is acceptable |
| `A_t` record | gaze choices | gaze history + step index | it is a record, not learned | that it should carry gradients |
| Classification readout | final belief | label (+ evidence) | it is a readout of perceptual state | that its objectives define the belief's content |

## State: persistent vs temporary

| Kind | Items | Lifetime |
|---|---|---|
| **Persistent** (checkpointed) | model parameters only — backbone, tied block, predictor, `UpdateNet`, evidential head, readout | across training |
| **Temporary** (per image) | the entire belief trajectory `B_1 … B_T`, token features, candidate scores | one forward pass, then discarded |

> There is **no** `B_t` persistence across images in Gen-1 — not as an
> object, not as a cache, not as memory. This is a specification
> decision (`04_Belief_State.md`), and any code that caches belief
> state across images is implementing a mechanism Gen-1 has not
> specified.

## The None-handling rule, in implementation terms

`S_t = None` is the *actual default configuration*. Therefore:

- every constructor of `B_t` must be able to produce the None
  structural variant;
- every consumer (`drift_to`, `as_tensor`, classifier head, update
  function `F`) must have an explicit `S_t is None` branch;
- tests must call the None branch **directly** (not merely happen to
  exercise it via defaults).

Silently substituting a zero tensor where a branch is required is a
specification violation, not an optimization.

## Configuration parameters visible at this level

| Parameter | Value in Gen-1 core | Set by |
|---|---|---|
| `T` (glimpses) | 4 | architecture (locked for first build) |
| `K` (AIS candidates) | 4–8 | AIS design (locked range) |
| within-glimpse refinement steps | 2–3 | substrate/implementation |
| `D_z` | substrate embedding dim | Agent C / substrate |
| `D_s` | n/a — `S_t` is None | (deferred with structure) |
| `C` | dataset class count | dataset |

Loss terms, optimizer schedules, and gate thresholds are **outside
this documentation pass** (master-plan Section 8+).

## What each implementer's module owes the others

- **Substrate owner:** patch-embedding output format must be exactly
  what the predictor predicts and the observation provides — the
  E_t-by-construction guarantee depends on it.
- **Predictor owner:** one module, exported once; both the update path
  and the AIS path import it. A second predictor implementation is a
  bug even if it trains better.
- **Update owner:** respect the optimizer-group isolation and the
  pre-flight `|dW|` check; never detach `E_t`.
- **AIS owner:** enforce the K-cheap-passes constraint at the
  interface; emit gaze-location distributions for center-bias
  reporting.
- **Head owner:** `U_t` is the *only* uncertainty representation; do
  not add a parallel scoring head.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A (None-propagation, serialization), 1.B (predictor
> unification, UpdateNet contract), 1.C (tied recurrence), 1.E
> (compute constraint). Boundary details are faithful restatements,
> not additions.
