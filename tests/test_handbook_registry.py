"""Tests for the v3.1 handbook data pipeline.

These tests check traceability and arithmetic, not scientific truth.  They run
without a GPU and without touching ``report/``; the only heavy dependency is
matplotlib, imported through ``style`` for the palette test.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "rhan_nxa" / "data"
sys.path.insert(0, str(ROOT / "scripts" / "handbook"))

import build_numbers  # noqa: E402
import render_numbers  # noqa: E402


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics(registry: dict) -> dict:
    return registry["metrics"]


def test_registry_satisfies_its_constraints(registry: dict) -> None:
    assert build_numbers.validate(registry) == []


def test_registry_is_not_stale(registry: dict) -> None:
    """A hand-edited registry that no longer matches the artifacts must fail."""
    fresh = {
        "schema_version": build_numbers.SCHEMA_VERSION,
        "registry_version": build_numbers.REGISTRY_VERSION,
        "project": "RHAN",
        "generated_at": None,
        "code_revision": None,
        "metrics": build_numbers.build_metrics(),
    }
    assert fresh["metrics"] == registry["metrics"]


def test_duplicate_ids_are_impossible_but_detected(metrics: dict) -> None:
    doc = {"schema_version": build_numbers.SCHEMA_VERSION,
           "registry_version": build_numbers.REGISTRY_VERSION,
           "project": "RHAN", "metrics": dict(metrics)}
    bad = dict(metrics["sweep.ff112.clean"])
    bad["display"] = "not the formatted value"
    doc["metrics"]["sweep.ff112.clean"] = bad
    assert any("display" in e for e in build_numbers.validate(doc))


def test_missing_provenance_is_rejected(metrics: dict) -> None:
    doc = {"schema_version": build_numbers.SCHEMA_VERSION,
           "registry_version": build_numbers.REGISTRY_VERSION,
           "project": "RHAN", "metrics": dict(metrics)}
    bad = dict(metrics["sweep.ff112.clean"])
    bad["source_artifacts"] = []
    bad["source_code"] = []
    doc["metrics"]["sweep.ff112.clean"] = bad
    assert any("without provenance" in e for e in build_numbers.validate(doc))


def test_invalid_unit_is_rejected(metrics: dict) -> None:
    doc = {"schema_version": build_numbers.SCHEMA_VERSION,
           "registry_version": build_numbers.REGISTRY_VERSION,
           "project": "RHAN", "metrics": dict(metrics)}
    bad = dict(metrics["sweep.ff112.clean"])
    bad["unit"] = "furlongs"
    doc["metrics"]["sweep.ff112.clean"] = bad
    assert any("unknown unit" in e for e in build_numbers.validate(doc))


def test_retracted_values_say_why(metrics: dict) -> None:
    retracted = [m for m in metrics.values() if m["status"] == "RETRACTED"]
    assert retracted, "the invalid ff_112 APGD rows must stay in the registry"
    for m in retracted:
        assert m["limitations"], "a retracted value must carry its reason"


def test_display_round_trips(metrics: dict) -> None:
    for mid, m in metrics.items():
        expect = build_numbers.fmt(m["value"], m["unit"], m["display_precision"],
                                   bool(m.get("signed_display")))
        assert m["display"] == expect, mid


def test_derived_values_close(metrics: dict) -> None:
    g = lambda k: metrics[k]["value"]  # noqa: E731
    assert g("sweep.ff112.drop_ppg") == pytest.approx(
        (g("sweep.ff112.clean") - g("sweep.ff112.pgd10_0094")) * 100, abs=0.02)
    assert g("cost.effective_delta") == pytest.approx(
        g("cost.effective_trainable_rec") - g("cost.effective_trainable_ff"), abs=0.5)
    assert g("lineage.epochs_rec_total") == pytest.approx(
        g("lineage.epochs_gist") + g("lineage.epochs_rec_phase"), abs=0.5)
    assert g("cost.macs_ratio") == pytest.approx(
        g("cost.macs_ff") / g("cost.macs_rec"), rel=1e-9)


def test_placeholder_resolution(metrics: dict) -> None:
    assert render_numbers.resolve("{{N:sweep.ff112.clean}}", metrics) == "73.79 %"
    with pytest.raises(render_numbers.UnknownMetric):
        render_numbers.resolve("{{N:does.not.exist}}", metrics)


def test_no_unresolved_placeholder_in_the_manuscript() -> None:
    metrics = json.loads((DATA / "numbers.json").read_text())["metrics"]
    for path in render_numbers.manuscript_files():
        text = path.read_text(encoding="utf-8")
        assert "{{N:" not in render_numbers.TOKEN.sub("", text), path
        for m in render_numbers.TOKEN.finditer(text):
            assert m.group(1) in metrics, f"{path}: {m.group(1)}"


def test_executive_overview_does_not_report_a_retracted_value() -> None:
    metrics = json.loads((DATA / "numbers.json").read_text())["metrics"]
    text = (ROOT / "docs/rhan_nxa/00_front_matter/01_executive_overview.md").read_text()
    for m in metrics.values():
        if m["status"] != "RETRACTED":
            continue
        i = text.find(m["display"])
        if i >= 0:
            window = text[max(0, i - 400):i + 400].lower()
            assert "retract" in window or "invalid" in window


def test_claims_reference_real_metrics(metrics: dict) -> None:
    claims = json.loads((DATA / "claims.json").read_text())["claims"]
    for c in claims:
        for mid in c["metric_ids"]:
            assert mid in metrics, f"{c['id']} cites {mid}"


def test_claims_do_not_promote_retracted_metrics(metrics: dict) -> None:
    claims = json.loads((DATA / "claims.json").read_text())["claims"]
    for c in claims:
        for mid in c["metric_ids"]:
            if metrics[mid]["status"] == "RETRACTED":
                assert c["status"] not in {"VERIFIED", "SUPPORTED_WITH_LIMITATIONS"}


def test_lineage_edges_and_cumulative_epochs() -> None:
    man = json.loads((DATA / "training_lineage.json").read_text())
    nodes = {n["id"]: n for n in man["nodes"]}
    for e in man["edges"]:
        assert e["from"] in nodes and e["to"] in nodes
    for nid, cum in man["cumulative_epochs"].items():
        assert cum >= nodes[nid]["epochs"]


def test_experiment_manifest_hashes_match_the_artifacts() -> None:
    """A re-run artifact must invalidate the registry rather than pass silently."""
    import hashlib

    man = json.loads((DATA / "experiment_manifest.json").read_text())
    assert man["experiments"]
    for e in man["experiments"]:
        path = ROOT / e["path"]
        assert path.exists(), e["path"]
        assert e["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), e["id"]


def test_figures_have_status_file_and_alt_text() -> None:
    man = json.loads((DATA / "figure_manifest.json").read_text())
    # The canonical manifest merges three generators: the priority figures plus
    # the carried legacy records, the suite, and the chapter openers / mechanism
    # figures. The total must equal the sum, and the chapter share must equal
    # 30 openers plus however many `fig_*` generators `main` actually calls —
    # never a number typed here after the fact.
    legacy = json.loads((ROOT / "docs/rhan_nxa/figures/figure_manifest.json").read_text())
    suite = json.loads((DATA / "figure_suite_manifest.json").read_text())
    chapter = json.loads((DATA / "figure_chapter_manifest.json").read_text())
    published = {f["id"] for f in suite["figures"]} | {
        f["id"] for f in chapter["figures"]}
    carried = [f for f in legacy["figures"] if f["id"] not in published]
    # Count the priority generators main() actually calls rather than a literal.
    psrc = (ROOT / "docs/rhan_nxa/figures/make_priority_figures.py").read_text()
    pmain = re.search(r"^def main\(\) -> int:(.*?)^if __name__", psrc,
                      re.S | re.M).group(1)
    priority = re.findall(r"^\s+(fig_\w+)\(\)", pmain, re.M)
    expected = (len(priority) + len(carried) + len(suite["figures"])
                + len(chapter["figures"]))
    assert man["figure_count"] == len(man["figures"]) == expected
    openers = [f for f in chapter["figures"] if f["id"].startswith("OPEN-CH")]
    assert len(openers) == 30
    assert {f["id"] for f in openers} == {f"OPEN-CH{c:02d}" for c in range(1, 31)}
    src = (ROOT / "docs/rhan_nxa/figures/make_chapter_figures.py").read_text()
    call_list = re.search(r"for fn in \((.*?)\):", src, re.S).group(1)
    assert len(chapter["figures"]) == 30 + len(re.findall(r"fig_\w+", call_list))
    # every figure must be referenced by a chapter, not just registered
    chapter_text = " ".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "docs/rhan_nxa").rglob("*.md"))
    for f in man["figures"]:
        assert f["file"].split("/")[-1] in chapter_text, f["id"]
    ids = [f["id"] for f in man["figures"]]
    assert len(ids) == len(set(ids))
    for f in man["figures"]:
        assert (ROOT / f["file"]).exists(), f["file"]
        assert f["status"] in {"EVIDENCE", "CONCEPTUAL"}
        assert f["alt_text"].strip()
        assert f["embedded_number"] is None


def test_palette_matches_the_legacy_generator() -> None:
    """Two generators, one visual identity: the constants must not drift."""
    src = (ROOT / "docs/rhan_nxa/figures/make_figures.py").read_text()
    leg = dict(re.findall(r'^(FF|REC|GEN1)\s*=\s*"([^"]+)"', src, re.M))
    spec = importlib.util.spec_from_file_location(
        "fig_style", ROOT / "docs/rhan_nxa/figures/style.py")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    assert style.ARM_COLORS["ff_112"] == leg["FF"]
    assert style.ARM_COLORS["recurrence_only"] == leg["REC"]
    assert style.ARM_COLORS["gen1_core"] == leg["GEN1"]


def test_every_arm_has_one_colour_and_one_marker() -> None:
    """An arm drawn in two chapters must be the same colour in both.

    The style module used to carry colours only for the five arms that had a
    robustness number; the other four fell back to a muted grey through
    ``get(arm, MUTED)``, which is how one arm becomes two greys in two figures.
    """
    spec = importlib.util.spec_from_file_location(
        "fig_style_arms", ROOT / "docs/rhan_nxa/figures/style.py")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    arms = {"ff_56", "ff_112", "g2_gist_only", "g2_gist_only_cont", "recurrence_only",
            "belief_no_f", "belief_with_f", "ais_v2_swap", "gen1_core"}
    missing_color = sorted(a for a in arms if a not in style.ARM_COLORS)
    missing_marker = sorted(a for a in arms if a not in style.ARM_MARKERS)
    assert not missing_color, missing_color
    assert not missing_marker, missing_marker
    assert len({style.ARM_COLORS[a] for a in arms}) == len(arms)


def test_epsilon_and_input_range_arithmetic() -> None:
    """The numbers quoted in Chapter 22.1.1 must follow from the constants."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean_std = sum(std) / 3
    eps_pixel = 0.094 * mean_std * 255
    assert eps_pixel == pytest.approx(5.42, abs=0.02)
    assert (8 / 255) / mean_std == pytest.approx(0.1389, abs=1e-3)
    lows = [(0 - m) / s for m, s in zip(mean, std)]
    highs = [(1 - m) / s for m, s in zip(mean, std)]
    assert min(lows) == pytest.approx(-2.118, abs=1e-3)
    assert max(highs) == pytest.approx(2.640, abs=1e-3)
    # the implemented clamp is strictly larger than the valid range
    assert all(-4.0 < lo for lo in lows) and all(4.0 > hi for hi in highs)
