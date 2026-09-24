# 04 — The Belief State: `B_t = (z_t, S_t, U_t, E_t, A_t)`

*Level 1 reading. Builds on `03_Perception_As_Investigation.md`.*

---

## In one sentence

The belief state is the structured snapshot of everything RHAN-NXA
currently "thinks" about the image — carried and updated across the
T glimpses of one investigation.

## The intuition

RHAN-NXA needs somewhere to represent what it currently believes about
the image. Not just a guess at the label — a *state*: what it thinks
it sees, how sure it is, how surprised it just was, and where it has
looked. Every decision the loop makes (where to look next, how much to
trust new evidence, how to revise itself) reads from and writes to
this state.

## A concrete example (the running example)

An image where an animal is partly hidden behind foliage:

- After glimpse 1, the model's `z_t` holds an incomplete percept —
  fur texture, an ear shape. `U_t` is high: several interpretations
  remain plausible. `A_t` records where it looked.
- The model predicts what another glimpse would reveal (e.g. "if this
  is a deer, looking just left of center should show a specific
  muzzle/antler structure").
- At glimpse 2, observation arrives. The mismatch (or match) becomes
  `E_t`; precision decides how much it matters; `z_t` is updated;
  uncertainty drops (or doesn't); the next gaze is chosen where
  ambiguity remains.

Each glimpse produces one `B_t`. The sequence B_1 → B_2 → B_3 → B_4
**is** the investigation.

## The five components

| Component | Beginner meaning | Intermediate meaning | Chapter |
|---|---|---|---|
| `z_t` | what the model thinks the scene contains | dense, continuous, differentiable global content vector | `05_z_State.md` |
| `S_t` | optional structural detail | explicit structured state; **None** in the Gen-1 core | `06_Structure_State.md` |
| `U_t` | how uncertain the model is | Dirichlet-evidence uncertainty over the classification readout | `07_Uncertainty.md` |
| `E_t` | how surprising the latest evidence was | latent prediction error at the next glimpse's fixation | `08_Prediction_Error.md` |
| `A_t` | where the model has looked | gaze history + current glimpse index (a record, not a learned tensor) | this chapter |

## Formal definition and shapes

```text
B_t = (z_t, S_t, U_t, E_t, A_t)

z_t : (B, D_z)                    global content
S_t : None | (B, K, D_s)          structure (default None)
U_t : Dirichlet evidence e_t (B, C), alpha_t = e_t + 1
E_t : latent error signal (predicted vs observed glimpse features)
A_t : gaze_history: list of (B,2) coords up to length T,
      current_glimpse_idx: int
```

In plain English: the model's current perceptual state — what it
thinks it sees, optional structure, uncertainty, surprise, and where
it has looked.

(`B` = batch size, `C` = number of classes, `D_z` = backbone embedding
dimension. Full notation table: `14_Tensor_Shape_Reference.md`.)

## Implementation contract

These rules are part of the specification, not suggestions:

### None-propagation
Every consumer of the belief state — `drift_to`, `as_tensor`, the
classifier head, and the update function `F` itself — must have an
**explicit branch for `S_t is None`**, tested directly. `S_t = None`
is the actual default configuration for Gen-1's first several phases,
not a hypothetical. Code that crashes on `S_t=None`, or silently
substitutes a zero tensor where a branch is required, violates the
spec.

### Gradient behavior
- `z_t`: gradient-bearing, always.
- `U_t`: gradient-bearing, always.
- `S_t`: gradient-bearing only when not None.
- `E_t`: computed from a **detached** "observed" target and a
  **non-detached** "predicted" value. `E_t` itself must **never be
  detached** before use in the update. This class of bug was the
  single most repeated failure mode in Gen 0's history.
- `A_t`: carries no gradient — it is a coordinate record;
  differentiability lives in the policy that produced it.

Full rationale: `13_Gradient_Flow.md`.

### Serialization / lifetime
`B_t` is **not** checkpointed as a persistent object across images. It
exists only within one image's forward pass (T glimpses), then is
discarded. What gets checkpointed is the **model** — the parameters
that produce `B_t` — never a specific `B_t` instance. There is no
"B_t persistence across images" in Gen-1 at all. This also answers the
memory question: belief states are per-image temporaries.

## Scientific status

| Piece | Status | Why |
|---|---|---|
| `B_t` as the explicit typed state | **REQUIRED** | the architecture is defined by maintaining it |
| `S_t = None` (core build) | **REQUIRED** (as the default) | doubled Gen-0 evidence against the current slot implementation — `06_Structure_State.md` |
| `U_t` = class-readout Dirichlet | **REQUIRED** for Gen-1, with a documented scoping compromise | validated component exists; representation-level uncertainty has no training target — `07_Uncertainty.md` |
| `E_t` = latent next-glimpse error | **EXPERIMENTAL CANDIDATE** | best-reasoned default, D3's evidence confounded — `08_Prediction_Error.md` |
| `A_t` as a non-learned record | **REQUIRED** | differentiability belongs to the policy, not the record |

## What this does NOT mean

- `B_t` is not a database of facts about the world; it is a
  per-image, per-forward-pass computation.
- "Belief" is a computational state variable, not a claim about
  consciousness or mentality.
- `S_t = None` does not mean the model cannot represent objects — it
  means no *explicit structural state module* is active in the core
  build. See `17_Common_Misunderstandings.md`.

## Open questions

- Whether `z_t` should eventually carry representation-level
  uncertainty (rather than only the class-readout `U_t`) — **PENDING
  DECISION**, deferred past the first integrated build.
- Whether `S_t` returns later in any form — **EXPERIMENTAL CANDIDATE,
  DEFERRED**.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A — BeliefState (all contracts quoted there are
> authoritative if this chapter and the plan ever diverge).
