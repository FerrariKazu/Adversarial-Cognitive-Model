"""Gate-2 rules v1.1 amendment: supersession, scene-capacity correction, timestamp erratum, and md hash pin."""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def test_v1_1_supersedes_v1_and_caps_scenes_at_non_reused_pair_capacity():
    v1 = _load("prereg/gate2_rules.json")
    v1_1 = _load("prereg/gate2_rules_v1_1.json")

    assert v1_1["supersedes"]["file"] == "prereg/gate2_rules.json"
    assert v1_1["supersedes"]["sha256"] == hashlib.sha256((ROOT / "prereg/gate2_rules.json").read_bytes()).hexdigest()
    assert v1_1["scene_capacity"]["fit_pool"]["planned_null_scenes"] == 850
    assert v1_1["scene_capacity"]["calibration_pool"]["planned_scenes"] == 850
    assert v1_1["scene_capacity"]["test_pool"]["planned_scenes"] == 800
    assert v1_1["data_bindings"]["val_split"]["fit_pool_part_0_of_3"] == v1["data_bindings"]["val_split"]["fit_pool_part_0_of_3"]
    assert v1_1["data_bindings"]["val_split"]["test_pool_part_2_of_3"] == v1["data_bindings"]["val_split"]["test_pool_part_2_of_3"]
    assert v1_1["erratum_01"]["actual_freeze_commit_utc"] == "2026-11-17T00:00:31Z"


def test_v1_1_pins_the_gate2_markdown_hash():
    v1_1 = _load("prereg/gate2_rules_v1_1.json")
    md = ROOT / "docs/research/RHAN_IM_GATE2_PREREG_DRAFT.md"
    assert v1_1["code_locking"]["gate2_md_draft"]["file"] == "docs/research/RHAN_IM_GATE2_PREREG_DRAFT.md"
    assert v1_1["code_locking"]["gate2_md_draft"]["sha256"] == hashlib.sha256(md.read_bytes()).hexdigest()
