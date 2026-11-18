"""Reference inspection laboratory (docs/08)."""

from rhan_bi.simulator.lab import IllegalAction, InspectionLab, LabProtocol
from rhan_bi.simulator.scene import AnomalySpec, Occluder, Scene, make_scene, mutate, render

__all__ = [
    "AnomalySpec",
    "IllegalAction",
    "InspectionLab",
    "LabProtocol",
    "Occluder",
    "Scene",
    "make_scene",
    "mutate",
    "render",
]
