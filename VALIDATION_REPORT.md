# Validation report — RHAN Technical Handbook v3.1

Canonical detailed report: `docs/rhan_nxa/07_deliverables/validation_report.md`.
Machine-readable record: `report/handbook_audit/validation.json`
(per-page measurements: `report/handbook_audit/page_audit.json`).

## Build

| Field | Value |
|---|---|
| PDF | `docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf` |
| Pages / size | 203 / 3,981,089 bytes (3.98 MB) |
| SHA-256 | `dfb25f3fa9c72d7b16ca1969c12321eb9538a4a13fdb4f6cafc4c6884a8045d7` |
| Base revision | `feature/gen2-rebuild` @ `d85163a` + uncommitted changes |
| Figures | 32 (8 carried, 5 prioritised v3.1, 19 suite): 21 `EVIDENCE`, 11 `CONCEPTUAL` |
| Figure placement | 24/24 managed references present and caption-identical to the manifest (`place_figures.py --check`); 32/32 captions found in the rendered PDF |
| Registry | 68 metrics: 64 `VERIFIED`, 2 `SUPPORTED_WITH_LIMITATIONS`, 2 `RETRACTED` |

## Checks run, with exit status

```
.venv/bin/python scripts/validate_rhan_handbook.py        # exit 0, RESULT: PASS
.venv/bin/python -m pytest tests/test_handbook_registry.py -q   # 18 passed
```

| Stage | Result |
|---|---|
| `scripts/handbook/validate_numbers.py` — registry, artifact hashes, tokens, claim graph, lineage, figures | PASS, 0 errors, 0 warnings |
| `docs/rhan_nxa/verify_claims.py` — recompute every cited number | PASS |
| `scripts/audit_handbook_numbers.py` — percentages | PASS |
| `scripts/audit_handbook_deltas.py` — percentage-point deltas | PASS |
| `scripts/audit_rhan_handbook_pdf.py` — 203 pages rasterised and measured | PASS |

Structural: 0 blank pages, 0 margin overflows, 356/356 contents entries resolve, 5/5 part dividers
carry their number, 0 unresolved generated-number tokens in the PDF text, and the PDF hash in
`validation.json` equals the file on disk.

## What was not verified

* No page or figure was inspected by eye; validation is programmatic and source-level.
* No experiment was re-run. Notebooks were not executed (the only one is a dispatch wrapper with
  stale logs), no APGD-CE evaluation was executed, and no GPU training or timing was performed.
* Mathematical proofs, statistical model assumptions and the correctness of the training algorithm
  itself are outside what these checks establish.
