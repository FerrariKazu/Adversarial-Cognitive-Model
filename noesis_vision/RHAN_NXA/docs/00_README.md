# RHAN-NXA Documentation

**Welcome.** This directory is the canonical documentation for
**RHAN-NXA** — the Generation-1 architecture of the RHAN research
program. It is written to be read by a newcomer, an implementer, a
reviewer, and the original researcher six months from now.

This is **documentation only**. The architecture is *specified* here,
not implemented. Nothing in this directory is a claim that code exists.

---

## What is RHAN-NXA?

A normal image classifier looks at an image once and outputs a label.

RHAN-NXA instead treats perception as an **iterative investigation**:

> **look → form a belief → estimate uncertainty → predict what another
> observation might reveal → observe → compare prediction with
> observation → update the belief → decide where to look next.**

The model keeps an explicit **belief state** about the image, and
classification is only a *readout* of that belief — not the belief
itself.

## Why does it exist?

Because Generation 0 (the previous generation, STL-10 scale) showed
that adding more structured modules and representational capacity does
not automatically produce a new clean/robustness frontier — every
addition traded along the existing frontier instead of moving it. And
because the two experiments intended to test isolated mechanisms
(AIS-v2, belief-HPC) were unintentionally contaminated by legacy SBR,
so their isolated effects are **UNKNOWN**. Gen-1 restarts from a
smaller, explicitly-labeled core. Details: `02_Why_RHAN_NXA_Exists.md`
and `16_Gen0_Evidence_And_Confounds.md`.

## What is the central idea?

**Perception as an investigation**, run as a loop:

**Predict → Observe → Error → Precision → Update → Attention → Repeat**

Every component in the architecture is one step of this loop.
Details: `03_Perception_As_Investigation.md`.

## What does the model actually do?

For each image, it takes **T = 4 glimpses** (where to look is chosen
by the AIS-v2 mechanism). Each glimpse passes through a compact ViT
with weight-tied within-glimpse refinement, produces local token
features and a pooled global vector, and updates the belief state:

> **Note on scope:** this documentation now covers the full authorized
> plan — Parts 1.A–1.I (architecture, uncertainty, L_stab, memory,
> frontend), Parts 2–6 (training DAG, ablation matrix, agents,
> infrastructure, gates), and the Agent 0 contract. Agents A–J's own
> prompts are **not yet written** (they are gated on review of Agent
> 0's handoff artifact).

```text
RHAN-NXA
│
├── Visual substrate
│   └── Compact ViT
│
├── Recurrent computation
│   ├── within-glimpse recurrence (weight-tied block, 2–3 steps)
│   └── across-glimpse recurrence (T = 4 fixed glimpses)
│
├── Explicit perceptual belief
│   └── B_t = (z_t, S_t, U_t, E_t, A_t)
│       ├── z_t   global content vector        (B, D_z)
│       ├── S_t   structural state            DEFAULT: None
│       ├── U_t   uncertainty (Dirichlet)     (B, C)
│       ├── E_t   prediction error (latent)   predicted next-glimpse features
│       └── A_t   gaze history + step index
│
├── Predictive dynamics
│   └── latent glimpse predictor (one module, two consumers)
│
├── Active observation
│   └── AIS-v2 (K = 4–8 candidates, shared predictor)
│
└── Readout
    └── classification / evaluation
```

## What are the five parts of B_t?

| Part | One-line meaning | Chapter |
|---|---|---|
| `z_t` | what the model currently thinks the scene contains | `05_z_State.md` |
| `S_t` | optional structural/object state (**None** in Gen-1 core) | `06_Structure_State.md` |
| `U_t` | how uncertain the model currently is (class-readout uncertainty) | `07_Uncertainty.md` |
| `E_t` | how surprising the latest evidence was vs. prediction | `08_Prediction_Error.md` |
| `A_t` | where the model has looked / is looking next | `04_Belief_State.md` |

## What should I read next?

### If you are new to RHAN
Read `01` → `02` → `03` → `04` → `08` → `09` → `10` → `11`.

### If you are implementing RHAN
Read the above, then `12` → `13` → `14` → `15` → `19` → `24` → `26` → `27` → `30`.

### If you are evaluating RHAN
Read `02` → `15` → `16` → `20` → `24` → `25` → `28`.

### If you are researching RHAN
Read everything, plus `noesis_vision/RHAN_NXA/MASTER_PLAN.md` (the source of truth
this documentation was built from).

---

## Document index

| File | Contents |
|---|---|
| `01_What_Is_RHAN_NXA.md` | Plain-language definition; what problem it solves |
| `02_Why_RHAN_NXA_Exists.md` | Gen-0 history and the confound, for non-researchers |
| `03_Perception_As_Investigation.md` | The central loop, word by word |
| `04_Belief_State.md` | `B_t` and all five components |
| `05_z_State.md` | The global content vector |
| `06_Structure_State.md` | `S_t = None`, and why |
| `07_Uncertainty.md` | Dirichlet evidence and its known limitation |
| `08_Prediction_Error.md` | Latent next-glimpse prediction; what was rejected |
| `09_Recurrence.md` | Within-glimpse, across-glimpse, halting deferral |
| `10_AIS_v2.md` | Choosing where to look |
| `11_Complete_Perceptual_Loop.md` | Everything assembled, one worked example |
| `12_Architecture_Data_Flow.md` | Implementation-oriented data flow |
| `13_Gradient_Flow.md` | What learns, what doesn't, and why |
| `14_Tensor_Shape_Reference.md` | Exact shapes and notation |
| `15_Status_And_Decision_System.md` | The five status labels and how to use them |
| `16_Gen0_Evidence_And_Confounds.md` | The evidence base, graded |
| `17_Common_Misunderstandings.md` | What NOT to conclude |
| `18_Glossary.md` | Every term, defined |
| `19_Decision_Records.md` | Why each non-obvious choice was made |
| `20_Scope_Boundaries.md` | What is explicitly outside Gen-1 |
| `21_L_stab_Stability.md` | The staged belief-stability objective + responsiveness guard |
| `22_Memory.md` | What counts as memory, what waits, what's rejected |
| `23_V1_Frontend.md` | The fixed Gabor frontend, built last |
| `24_Training_Phase_DAG.md` | Build graph vs experiment graph, steps 1–11 |
| `25_Ablation_Matrix.md` | The ten arms and the identical-columns discipline |
| `26_Agent_Organization.md` | Agents 0–J, ownership, the two adjustments |
| `27_Infrastructure_Port_Table.md` | PORT VERBATIM / ADAPT / REJECT, with reasons |
| `28_Gates_and_Compute_Accounting.md` | G6–G10 with honest threshold statuses |
| `29_Literature_Classification.md` | KNOWN PRIOR / ENGINEERING ADAPTATION / SCIENTIFIC COMBINATION |
| `30_Agent0_Interface_Contract.md` | The first agent's full contract |
| `../MASTER_PLAN.md` | Source of truth (Part 0 + Parts 1.A–1.I + Parts 2–6 + Agent 0) |

---

> **One rule before you start:** every architectural component in this
> documentation carries a **status label** — REQUIRED, PENDING
> DECISION, EXPERIMENTAL CANDIDATE, DEFERRED, or REJECTED. These
> labels are part of the specification. If a document ever reads as if
> a hypothesis were a proven fact, that is a documentation bug — see
> `15_Status_And_Decision_System.md`.
