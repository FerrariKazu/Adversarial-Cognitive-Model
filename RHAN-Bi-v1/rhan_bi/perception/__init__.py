"""Perception front-end: handcrafted anomaly features (default) and an optional DINOv2 path."""

from rhan_bi.perception.backbone import Backbone, DinoV2Backbone, HandcraftedBackbone
from rhan_bi.perception.features import FEATURE_NAMES, plate_roi, z_extract

__all__ = ["Backbone", "DinoV2Backbone", "FEATURE_NAMES", "HandcraftedBackbone", "plate_roi", "z_extract"]
