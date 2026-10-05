# Appendix C — Infrastructure Port Table and Lineage Resolution

> *Authoritative disposition table for all inherited and newly developed infrastructure components in RHAN-NXA Generation-1.*

---

## 1. Overview and Porting Principles

RHAN-NXA Generation-1 builds upon algorithmic insights from the prior STL-10 research exploration (Generation-0). However, to prevent technical debt and legacy confounds from migrating into the new system, every inherited module was subjected to a formal disposition review.

No module is ported by momentum; no component is rewritten by subjective taste.

### Porting Dispositions Defined
1. **PORT VERBATIM**: Copied as-is without semantic or structural alteration. The module is empirically validated and zero justification exists to modify it.
2. **ADAPT**: Semantics and mathematical logic preserved, but interface bindings are updated to match ImageNet-100 dimensions, new tensor namespaces, or normalization conventions.
3. **REJECT**: Prohibited from being ported. Must not exist even as a fallback or dead branch.
4. **NEW**: First implementation in this repository of an established scientific formulation. Carries zero inherited validation and requires full unit test coverage.

---

## 2. Master Infrastructure Port Table

| Component / Module | Source in Gen-0 | Disposition | Rationale and Technical Scope |
|---|---|---|---|
| **Multi-Group Optimizer + Registry** | `core/optimizer.py` | **PORT VERBATIM** | Validated across 3+ Gen-0 components; enables strictly isolated parameter groups and learning rates. |
| **Gradient Isolation Pre-Flight (`\|\nabla W\|`)** | `tests/preflight.py` | **PORT VERBATIM** | Standing rule: zero-leakage must be verified analytically before any smoke run is trusted. |
| **Checkpoint Resume + Parity Assertion** | `core/checkpoint.py` | **PORT VERBATIM** | Learned the hard way: bit-for-bit parity check between best and rolling models is non-negotiable. |
| **Summary Table vs CSV Parity Assertion** | `core/metrics.py` | **PORT VERBATIM** | Enforces that markdown tables in reports match raw CSV metric logs identically. |
| **Comparator Model Registry** | `benchmarks/registry.py` | **ADAPT** | Rebinds checkpoint paths and evaluation hooks for ImageNet-100; preserves donor-row discipline. |
| **Adversarial Evaluation (PGD / AutoAttack)** | `eval/attacks.py` | **ADAPT** | Preserves exact norm-bounded attack routines; updates image normalization and resolution ($224 \times 224$). |
| **Evidential Head** | None (Literature) | **NEW** | First implementation in this codebase of Sensoy et al. (2018); not ported from Gen-0. |
| **AIS-v1 Relocated-Gaze Module** | `gaze/ais_v1.py` | **REJECT** | Mathematically and empirically superseded by AIS-v2; banned from entering the codebase. |
| **16-Slot Spatial Bottleneck Recurrence** | `models/sbr.py` | **REJECT** | Chance-level slot decoding ($0.44–0.51$); caused reproducible $-10$ pp clean accuracy collapse. |

---

## 3. Discrepancy Resolution: The `EvidentialHead` Status

In early development notes, informal text occasionally referred to the `EvidentialHead` as "ported verbatim." However, a rigorous code audit confirmed that **no EvidentialHead class existed in the Generation-0 codebase**.

In the authoritative Master Implementation Plan (Part 5), the disposition was formally corrected and locked to **NEW**:
- It is the first implementation in this project of the Dirichlet evidential deep learning formulation (Sensoy et al., NeurIPS 2018).
- It inherits no empirical validation from Gen-0.
- It was constructed from scratch by Agent D in [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py) and is covered by dedicated unit test suites verifying non-negative evidence softplus mapping and Dirichlet entropy calculations.

---

## 4. The No-Parallel-Port Rule

The Generation-0 directory `rhan_core/` is preserved strictly as a **read-only historical archive**:
1. No Gen-1 module may import from `rhan_core`.
2. All components must import strictly from `noesis_vision.*`.
3. If an unported utility is required, it must pass an explicit RFC review, be assigned an ADAPT or PORT VERBATIM disposition, and be migrated into `noesis_vision/` under contract tests.

This rule completely eliminates silent dependency contamination and ensures complete reproducibility of the Generation-1 architecture.
