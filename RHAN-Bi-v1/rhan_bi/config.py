"""Typed configuration. YAML is merged over these defaults; unknown keys are an error."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from rhan_bi.util import canonical_json

T = TypeVar("T")


@dataclass(frozen=True)
class SimConfig:
    canvas: int = 96
    plate_inset: int = 16
    noise: float = 0.02
    reflection_light_shift: float = 14.0
    reflection_view_shift: float = 6.0
    shadow_light_shift: float = 10.0
    view_parallax: float = 4.0
    occluder_rate: float = 0.20
    zoom_factor: float = 2.0
    seed_offset: int = 0


@dataclass(frozen=True)
class FeatureConfig:
    highpass_sigma: float = 4.0
    line_length: int = 15
    ridge_threshold: float = 3.0
    roi_margin: int = 4


@dataclass(frozen=True)
class EvidenceConfig:
    dependence_same_action: float = 1.00
    dependence_same_view: float = 0.55
    dependence_same_light: float = 0.60
    dependence_overlap_roi: float = 0.45
    dependence_scene: float = 0.10
    min_weight_quality: float = 0.50


@dataclass(frozen=True)
class BeliefConfig:
    prior: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    tau_commit: float = 0.80
    tau_debt: float = 0.35
    tau_abstain_margin: float = 0.10
    conflict_log: float = 1.0986122886681098  # log(3)
    debt_cap: float = 1.0


@dataclass(frozen=True)
class PlannerConfig:
    w_falsify: float = 1.0
    w_risk: float = 0.5
    lam_cost: float = 0.05
    lam_redundancy: float = 0.15
    lam_quality: float = 0.10
    n_samples: int = 64
    cost_light: float = 1.0
    cost_zoom: float = 1.0
    cost_view: float = 2.0


@dataclass(frozen=True)
class RunConfig:
    seed: int = 7
    n_train: int = 240
    n_cal: int = 120
    n_dev: int = 120
    n_test: int = 120
    n_gate_e: int = 120
    budget: int = 3
    results_dir: str = "results"
    tag: str = "demo"
    bootstrap: int = 2000
    workers: int = 1


@dataclass(frozen=True)
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            section = getattr(self, f.name)
            out[f.name] = {sf.name: getattr(section, sf.name) for sf in dataclasses.fields(section)}
        return out

    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        sections: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            section_cls = f.type if not isinstance(f.type, str) else None
            if section_cls is None:  # postponed annotations
                section_cls = {
                    "sim": SimConfig,
                    "features": FeatureConfig,
                    "evidence": EvidenceConfig,
                    "belief": BeliefConfig,
                    "planner": PlannerConfig,
                    "run": RunConfig,
                }[f.name]
            raw = data.get(f.name, {})
            if not isinstance(raw, dict):
                raise ValueError(f"config section {f.name!r} must be a mapping")
            names = {sf.name for sf in dataclasses.fields(section_cls)}
            bad = set(raw) - names
            if bad:
                raise ValueError(f"unknown keys in {f.name}: {sorted(bad)}")
            if f.name == "belief" and "prior" in raw:
                raw = {**raw, "prior": tuple(float(x) for x in raw["prior"])}
            sections[f.name] = section_cls(**raw)
        return cls(**sections)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        data = yaml.safe_load(p.read_text()) or {}
    if overrides:
        for section, values in overrides.items():
            merged = dict(data.get(section, {}))
            merged.update(values)
            data[section] = merged
    return Config.from_dict(data)


def default_config() -> Config:
    return Config()


def dump_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=True))
