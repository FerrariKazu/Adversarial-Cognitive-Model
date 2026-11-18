# 18 — Unreal Engine handoff

This is the contract for the simulation developer. It is written to be implementable without reading
the rest of the dossier, and it is deliberately smaller than "build a virtual factory".

## 1. What to build

A single small inspection laboratory:

- a table with a flat metal plate (roughly 20 × 20 cm) and 3–5 interchangeable test components;
- a camera on a two-position rail (front, left 30°, right 30°) able to zoom to a 2× crop;
- a light rig with three switchable azimuths (left 45°, centre, right 45°) at fixed elevation;
- a set of defect assets: scratch, dent, stain — injectable at parameterised positions, with a
  "pristine" variant;
- reflective material variants (brushed, polished) and a matte variant;
- an occluder bar that can be placed over 10–30 % of the plate;
- a Python-callable control interface (TCP/JSON or a local process API) implementing §3.

Out of scope: a game, a city, a full digital twin, physics-accurate wear, multiple rooms, humans,
animation, VR.

## 2. Fidelity requirements

| Property | Requirement | Why |
|---|---|---|
| light change | the specular reflection of a scratch-like dark object must move with the light; a real scratch must not | this is the discriminating signal |
| view change | parallax consistent with the camera rail; reflections move with viewpoint | prevents a fake cue from viewpoint |
| zoom | resolves fine scratch geometry without changing the anomaly's identity | resolution ≠ identifiability |
| materials | at least one high-gloss and one matte material per component | reflection scenes need gloss |
| defect injection | deterministic given `(scene_seed, defect_params)`; can be applied after an initial observation | S4 state change |
| noise | realistic sensor noise and slight exposure variation, seeded per observation | prevents perfect clean images |
| determinism | same seed + same action ⇒ bitwise-identical PNG (or a documented tolerance ≤ 1 LSB) | the whole analysis assumes replay |

Photorealism is **not** required. Consistent geometry and repeatable intervention are.

## 3. Interface

```python
class UnrealLab:
    def reset(self) -> Observation: ...
    def observe(self, action: dict) -> Observation: ...
    def legal_actions(self) -> list[dict]: ...
    def ground_truth(self) -> dict: ...
    def advance_scene(self, mutation: dict) -> Observation: ...
```

### Observation

```json
{
  "image_path": "frames/<episode>/<step>.png",
  "image_sha256": "...",
  "width": 1024, "height": 1024,
  "action": {"kind": "light", "view": 0, "light": 2, "roi": null},
  "step": 1,
  "scene_id": "lab_000123",
  "roi": [0.0, 0.0, 1.0, 1.0]
}
```

The agent reads the image and the action. It must **not** receive the cause, the defect mask, the
material label or the light object in this payload.

### Actions

| `kind` | Fields | Legality |
|---|---|---|
| `light` | `light ∈ {0,1,2}` | must differ from current light |
| `view` | `view ∈ {-1,0,1}` | must differ from current view |
| `zoom` | `roi = [x0,y0,x1,y1]` in normalised coordinates | crop ≥ 25 % of the plate; at most twice per episode |

Costs are assigned by the Python side; the Unreal build does not need to know them.

### Ground truth

```json
{
  "scene_id": "lab_000123",
  "cause": "defect",
  "defect": {"type": "scratch", "centre_px": [512, 480], "angle_rad": 0.6,
             "length_px": 210, "width_px": 6, "severity": 0.7},
  "material": "polished",
  "occluder": null,
  "light": {"azimuth_deg": 45, "elevation_deg": 30},
  "camera": {"view": 0, "zoom": 1.0},
  "seed": 123,
  "mutation_step": null
}
```

`cause` must be exactly one of `defect`, `reflection`, `shadow`, `none`. A scene labelled `reflection`
has **no** physical defect; the appearance must be produced by a reflective object in the
environment. A scene labelled `shadow` has no defect and no reflection; the appearance comes from
geometry and light.

## 4. Episode protocol

1. `reset()` returns step 0 at front view, centre light, zoom 1.
2. The agent sends at most 3 legal actions.
3. `ground_truth()` is callable at any time but the runner calls it only after the decision.
4. `advance_scene({"introduce_defect": {...}})` mutates the component and returns the next
   observation; used only for scenario S4.
5. Episode ends at budget, or when the agent commits.

## 5. Acceptance tests (must pass before the data are used)

| ID | Test | Pass criterion |
|---|---|---|
| A1 | repeat `reset()` with the same seed | identical PNGs (or ≤ 1 LSB) |
| A2 | same action twice | identical PNGs; a duplicate must add no new information |
| A3 | light change, defect scene | defect region unchanged in object coordinates (SSIM of the defect crop ≥ 0.98) |
| A4 | light change, reflection scene | the reflective streak centroid moves by ≥ 20 px at 1024² |
| A5 | view change, defect scene | defect translates by the parallax predicted from the rail geometry (±3 px) |
| A6 | zoom | crop matches the requested ROI; no resampling artefacts beyond a documented kernel |
| A7 | occluder | the occluded region is truly hidden (defect not recoverable from any pixel) |
| A8 | S4 mutation | before/after `ground_truth()` differs; the image changes only after the mutation |
| A9 | ground-truth leakage | the observation payload contains none of: cause, defect, material, light object |
| A10 | throughput | ≥ 2 frames/s at 1024² on the target machine, headless |

A9 is checked mechanically by parsing the observation JSON keys.

## 6. Data export

```
exports/
  lab_v1/
    scenes.jsonl          one line per scene (ground truth)
    episodes/
      <episode_id>/
        step_0.png, step_1.png, ...
        actions.json
        ground_truth.json
    manifest.json         build hash, Unreal version, material assets, seed list
```

The Python side has a loader (`rhan_bi.simulator.export`) that converts this layout into the same
`Observation` objects as the reference simulator, so every gate runs unchanged against Unreal data.

## 7. What the Python side guarantees

- the same action space and legality rules;
- the same ground-truth schema;
- an adapter (`UnrealLabClient`) with a timeout and a clear error if the server is down;
- no hidden dependence on the reference renderer: gates read through `LabProtocol` only.

## 8. What the developer should not do

- Do not encode the cause in the image in a way that a simple statistic reveals (e.g. always the same
  background for defects). The acceptance suite detects gross versions of this, not subtle ones.
- Do not render the reflection as a texture glued to the plate; it must move with the light.
- Do not "improve" the scene between actions except through `advance_scene`.
- Do not add post-processing that differs between views or lights (exposure, colour grading), beyond a
  documented, seeded sensor model applied uniformly.

## 9. Milestones

| Week | Deliverable |
|---|---|
| 1 | scene + plate + camera rail; A1, A2, A5 |
| 2 | light rig and materials; A3, A4, A6 |
| 3 | defect injection + occluder; A7, A8 |
| 4 | export pipeline, A9, A10, 200-scene pilot set |
| 5 | 1,000-scene set for Gates C/D, manifest, handover |

The reference simulator remains the primary path until A1–A10 pass; Unreal data is not mixed with
reference data in any single result table.
