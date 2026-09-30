# Applications (future layer — intentionally empty)

This directory is **architectural infrastructure only**. It reserves a home
for future RHAN/NOESIS applications so they never grow inside the research
core.

## The boundary

```text
research architecture (noesis_vision/, training/, evaluation/)
        ↓  consumes through defined model-level interfaces
reusable RHAN capabilities
        ↓  composition, product logic, UX
application-specific systems (HERE)
```

Applications must consume RHAN through the model-level interfaces
(`noesis_vision/core/schema.py` config + the belief-state / readout
interfaces under `noesis_vision/`), **not** by importing internal training
code (`training/`), cloud launchers, or evaluation harness internals.

## Status

- **NOT YET IMPLEMENTED** — no application code exists here yet, by design.
  Building applications now would be premature: the research engine must
  first be understandable and stable (see `docs/REPOSITORY_MAP.md` §19/§20).
- `cognitive_vision_lab/` (repo root) is an existing decoupled Streamlit
  benchmarking side project — a candidate future resident of this layer or
  of its own repository. It is independent of the RHAN core today; that
  isolation is deliberate and documented in the repository map (§4.15).

## Adding an application later

1. One subdirectory per application (`applications/<name>/`), with its own
   README stating which RHAN interfaces it consumes.
2. Depend on the `noesis_vision` package only — never on `training/`,
   `evaluation/`, `scripts/`, or `cloud_setup/`.
3. No experiment artifacts under `applications/` — artifacts belong in
   `report/` / HF per the provenance rules in `CONTRIBUTING.md`.
