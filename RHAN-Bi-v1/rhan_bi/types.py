"""Core data types that cross component boundaries.

Everything here is immutable. ``Observation`` deliberately has no cause field: the agent-facing
type must not be able to carry ground truth (docs/08 §4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhan_bi.util import canon, sha256_image, sha256_payload

CAUSES: tuple[str, ...] = ("defect", "reflection", "shadow", "none")
ACTION_KINDS: tuple[str, ...] = ("base", "light", "view", "zoom")
LEDGER_VERSION = "rhan-bi-ledger-v1"
FEATURE_VERSION = "features-v1"


@dataclass(frozen=True)
class Action:
    """A request to the environment. ``base`` is the reset observation and has cost 0."""

    kind: str = "base"
    view: int = 0
    light: int = 1
    roi: tuple[float, float, float, float] | None = None
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}")
        if self.view not in (-1, 0, 1):
            raise ValueError("view must be -1, 0 or 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "view": self.view,
            "light": self.light,
            "roi": list(self.roi) if self.roi is not None else None,
            "cost": self.cost,
        }

    def label(self) -> str:
        if self.kind == "base":
            return "base"
        if self.kind == "zoom":
            return "zoom"
        return f"{self.kind}@{self.view if self.kind == 'view' else self.light}"


@dataclass(frozen=True)
class Observation:
    """What the agent may see. No cause, no mask, no material, no light object."""

    image: np.ndarray
    action: Action
    step: int
    scene_id: str
    view: int
    light: int
    roi: tuple[float, float, float, float] | None = None
    image_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.image_sha256:
            object.__setattr__(self, "image_sha256", sha256_image(self.image))

    def meta(self) -> dict[str, Any]:
        h, w = self.image.shape[:2]
        return {
            "height": int(h),
            "width": int(w),
            "view": int(self.view),
            "light": int(self.light),
            "roi": list(self.roi) if self.roi is not None else None,
        }


@dataclass(frozen=True)
class Encoding:
    features: dict[str, float]
    maps: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)
    backbone: str = "handcrafted"


@dataclass(frozen=True)
class QualityReport:
    score: float
    flags: tuple[str, ...]
    components: dict[str, float]


@dataclass(frozen=True)
class EvidenceRecord:
    """One immutable, content-addressed observation (docs/03 §2)."""

    ledger_version: str
    episode_id: str
    scene_id: str
    step: int
    action: dict[str, Any]
    image_sha256: str
    features: dict[str, float]
    quality: dict[str, Any]
    llr: dict[str, float]
    pair: dict[str, Any] | None
    dependence: dict[str, Any]
    provenance: dict[str, Any]
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.record_id:
            object.__setattr__(self, "record_id", self.compute_id())

    @staticmethod
    def _payload_of(fields: dict[str, Any]) -> dict[str, Any]:
        payload = {k: v for k, v in fields.items() if k not in ("record_id",)}
        return canon(payload)

    def payload(self) -> dict[str, Any]:
        return self._payload_of(self.__dict__)

    def compute_id(self) -> str:
        return sha256_payload(self.payload())

    def verify(self) -> bool:
        return self.record_id == self.compute_id()

    def weight(self) -> float:
        return float(self.dependence.get("weight", 1.0))

    def total_llr(self) -> dict[str, float]:
        """Singleton LLR plus pair LLR, keyed by cause."""
        out = {c: float(self.llr.get(c, 0.0)) for c in CAUSES}
        if self.pair:
            for c in CAUSES:
                out[c] += float(self.pair.get("llr", {}).get(c, 0.0))
        return out

    def to_json(self) -> dict[str, Any]:
        return canon({**self.__dict__})

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EvidenceRecord":
        return cls(**data)
