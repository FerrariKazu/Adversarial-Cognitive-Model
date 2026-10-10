# Gen-2 Foundation Package — Audit

Canonical audit for `gen2_foundation/` (G2-K9/K1/K4/K5/K6/K7/K8, the Gen-2
foundation slice only). Frozen scope: NO policy zoo (G2-K3), NO multi-scale
pyramid, NO memory, NO halting, NO Gen-3, NO lambda-family loss search.

---

## 1. Verification of existing repo symbols (Gen-1 baseline)

Baseline branch `stage2/nxa-pipeline-refactor` @ clean state; `main` = `4f18bd4`
(Stage 3: PGD-100 silent-crash fix). Existing Gen-1 interfaces are **reused,
never modified**. Each symbol below was read directly from disk and matched
against the spec.

| Spec symbol | Repo location | Verified | Notes |
|---|---|---|---|
| `RHANNXAConfig` | `noesis_vision/core/schema.py` | ✅ | dataclass, `schema_version=0.1.0`, `GATED_FLAGS`, `REJECTED_OUTRIGHT` |
| `CompactViT` | `noesis_vision/models/backbone.py` | ✅ | `patch_embed` is `nn.ModuleDict({"proj": Conv2d(3,384,14,stride=14)})` |
| `ConcreteUpdateNet` | `noesis_vision/predictive_coding/update_net.py` | ✅ | + `DELTA_BOUND=0.1` |
| `PrecisionFunction` | `noesis_vision/predictive_coding/precision.py` | ✅ | repo Gen-1 uses a **learned single MLP** (not the spec's six-arm family) |
| `OptimizerGroupRegistry` | `noesis_vision/core/multi_group_optimizer.py` | ✅ | wrapped by `recipe.py` → `torch.optim.AdamW` |
| `PredictorTarget` | `noesis_vision/predictive_coding/ema_predictor.py` | ✅ | nn.Module contract; its `state_dict()`/`load_state_dict()` is the source of the 6 pre-existing test failures |
| `SpatialErrorPool` | `noesis_vision/predictive_coding/spatial_error_pool.py` | ✅ | |
| Six-phase ladder | `training/train_generation1_foundation.py`, `scripts/stage_state_machine.py` | ✅ | |
| Eval/attack infra | `phase2_attacks/`, `evaluation/` | ✅ | reused, not replaced |

### Discrepancies that were **documented, not silently fixed**

1. **`precision.py` (new).** The repo's existing `PrecisionFunction` is a learned
   single MLP. The spec's six-arm family (const, one_minus_u / legacy `1-U`,
   learned_scalar, learned_spatial, channelwise, policyconditioned, collapse
   diagnostics) is **not** present anywhere in the repo. Added in
   `gen2_foundation/precision.py` exactly as specified. Legacy `1-U` is the
   **executable Gen-1 control** — the executable control arm, not a collapse of
   precision vs uncertainty. `PrecisionField` requires a `U` input that
   evaluation must supply.

2. **`recipe.py` (new).** Spec's SGD→AdamW recipe (with warmup + cosine) is
   **not** present. `build_adamw` / `AdamWGroups` wrap
   `OptimizerGroupRegistry` → `torch.optim.AdamW` with per-group weight decay and
   warmup + cosine schedule.

3. **`eot.py` (new).** Spec's EOT-PGD (EOT averaging over `n_eot` copies,
   deterministic `n_eot=1`) is **not** present in the repo. Added as EOT-PGD
   reference + `eot_sanity_loss` + `build_autoattack_loss_wrapper` (AutoAttack
   integration point; no silent replacement of existing infra).

4. **Gist contract — exact match.** Spec's gist contract (shared `patch_embed`,
   zero-gated fusion, 16 tokens) matches `CompactViT.patch_embed` exactly.
   `FixedGistEncoder` uses the shared `patch_embed["proj"]` (guarded for
   both `nn.ModuleDict` and plain-module shapes). Fix documented in `gist.py`.

---

## 2. Gating design (`flags.py`)

- `RHANGen2Config` — frozen dataclass; every field is a pre-registered experiment
  ID; defaults are legal control arms.
- Default config = **legal control arm** (`no_gist=True` default = Gen-1 parity
  control; `is_control_arm()` True).
- `GATED_FLAGS` — ~32 mechanism names → exact experiment IDs; flipping a
  mechanism without its exact registered ID raises at construction (Gated flag
  G10).
- `REJECTED_OUTRIGHT` — carried **verbatim** from Gen-1 (`schema.py`,
  `REJECTED_OUTRIGHT`), duplicated here so the package cannot silently re-admit a
  rejected mechanism.
- `G2_OUT_OF_SCOPE_IDS` = policy zoo, multi-scale, memory, halting, Gen-3,
  deferred.
- `G2_EXPERIMENT_IDS` — freeze of all valid foundation IDs (33 IDs).
- One-parameter-one-group enforcement is a **recipe-level invariant**, not a
  config field.

---

## 3. Rule dispositions

| Mechanism | Disposition | Detail |
|---|---|---|
| Gist (K1) | Frozen at foundation slice | shared `patch_embed` + zero-gated fusion + 16 tokens; Gen-1 parity default |
| Precision (K4) | Implemented per spec | six-arm family + legacy `1-U`; legacy `1-U` is the executable Gen-1 control |
| Update (K5/K6/K7) | Implemented per spec | `ObservedFeatureFusion`, `PositionSensitiveErrorPool`, `BeliefUpdaterV2`, `adapt_magnitude` |
| EMA (K7) | Gradient-free target-encoder | `EMATargetEncoder`; alpha∈[0,1] inclusive; polyak sync; `record_agreement`/`ema_lag` diagnostics |
| Recipe (K7/AdamW) | Implemented per spec | `build_adamw`, `AdamWGroups`, `PhaseRecipe`, `measure_step_magnitudes` |
| EOT / AutoAttack (K8) | Reference + integration point | `eot_pgd_attack`, `eot_sanity_loss`, `build_autoattack_loss_wrapper`; AutoAttack optional import |

---

## 4. README / requirements.txt / config/train_config.yaml resolution

**Decision:** resolved to the `main`/base versions (the clean, authoritative
versions of these files). These files had unresolved merge-conflict markers
(`<<<<<<<`) that blocked the commit of the Gen-2 package; the conflict-state was
dropped and the files were re-staged cleanly rather than left as broken merge
documents.

* **Why this is a real resolved ambiguity, not a silent choice.** These files
  were in an in-progress merge between `main` and another branch. Committing the
  Gen-2 package while those files held unresolved conflict markers would have
  (a) blocked the commit (Git refuses to commit with unmerged files), and
  (b) risked baking conflict markers into the committed tree. Both were unacceptable.
  Resolution: the clean base (`main`) versions are authoritative and were
  restored. The Gen-2 package is unaffected.

* The Gen-2 package itself makes **no** edits to `README.md`, `requirements.txt`,
  or `config/train_config.yaml`. These were resolved purely to unblock the commit;
  the resolution is recorded here so the next reader does not treat them as
  part of the implementation change.

---

## 5. Frozen scope confirmations

| Out-of-scope mechanism | Confirmed NOT in foundation slice |
|---|---|
| Policy zoo (G2-K3) | ✅ |
| Multi-scale pyramid | ✅ |
| Memory (spatial working/memory) | ✅ |
| Halting (variable glimpses) | ✅ |
| Gen-3 (world-state, object-scene, counterfactual, reasoning, graduated) | ✅ |
| Deferred (G2-K11 calibration, G2-K10 factor-disagreement) | ✅ |

Deferred items (G2-K10 etc.) are gated on the foundation slice reporting real
numbers: the foundation slice has **not** yet produced those numbers. Their
deferral is recorded in `G2_OUT_OF_SCOPE_IDS` and in the scope-boundary doc
(`noesis_vision/RHAN_NXA/docs/20_Scope_Boundaries.md`).

---

## 6. Test results

`gen2_foundation/tests/` — **52 passed, 0 failed** (see `tests/__init__.py`).

---

## 7. Commit provenance

- `9929b47` — commit of this package (modules + tests). Commits
  `README.md`/`requirements.txt`/`config/train_config.yaml` were additionally
  dropped from merge-conflict state to unblock the commit (see §4).
- The stale `noesis_vision/gen2_foundation/` duplicate was **deleted** (see §8).

## 8. Stale-copy resolution

The earlier `noesis_vision/gen2_foundation/AUDIT.md` was an accidental, untracked
copy-paste with **no git history** and is a duplicate of the top-level package's
file(s). It was deleted (rule: check history first — `git log --follow` on both
paths shows zero history for both, confirming accidental copy). No two packages
share a name under different parent paths.

> Note: the prior summary claimed `gen2_foundation/AUDIT.md` was part of the
> originally committed package; it was not. This file (`AUDIT.md`) documents that
> the audit content was created as a deliverable and is now committed here.
