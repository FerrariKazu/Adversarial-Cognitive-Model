"""RHAN-Bi — evidence-integrity active perception (research reference implementation).

The loop is: observe -> build an evidence record -> append it to a content-addressed ledger ->
recompute the belief -> choose the next observation (or commit / abstain).

Nothing here is a trained end-to-end system. The stages are separate and measurable by design;
see ``docs/07_TRAINING_PROGRAM.md``.
"""

__version__ = "1.0.0"

from rhan_bi.types import CAUSES, Action, Encoding, EvidenceRecord, Observation, QualityReport

__all__ = [
    "CAUSES",
    "Action",
    "Encoding",
    "EvidenceRecord",
    "Observation",
    "QualityReport",
    "__version__",
]
