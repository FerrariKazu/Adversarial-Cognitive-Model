"""PROPOSED tests for change C4 (report-only slope-heterogeneity script). SYNTHETIC: hand-built maps and a tiny self-test run; nothing here is a measurement of the real sensor."""
import importlib.util
import json
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_gate1_script import load_script, tiny_args  # noqa: E402


def load_het():
    spec = importlib.util.spec_from_file_location("het", ROOT / "scripts" / "rhan_im" / "gate1_slope_heterogeneity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_lower_median_is_torch_median_and_differs_from_the_ordinary_median_for_even_counts():
    h = load_het()
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert h.lower_median(t) == 2.0 == float(t.median()) and h.middle_median(t) == 2.5
    assert h.lower_median(torch.tensor([1.0, 2.0, 3.0])) == h.middle_median(torch.tensor([1.0, 2.0, 3.0])) == 2.0
    assert math.isnan(h.lower_median(torch.zeros(0)))


def test_map_stats_on_a_hand_checkable_map():
    h = load_het()
    m = torch.ones(4, 4, dtype=torch.double)
    m[0, 0] = float("nan")                                  # an unmeasured position
    m[3, 3] = 2.0                                           # outside [0.8, 1.25]
    s = h.map_stats(m)
    assert s["n_measured"] == 15 and s["n_positions"] == 16
    assert s["fraction_outside_band"] == pytest.approx(1 / 15) and s["max"] == 2.0 and s["min"] == 1.0
    assert s["lower_median"] == 1.0 and s["bottom_right_lower_median"] == 1.0 and math.isnan(h.map_stats(torch.full((4, 4), float("nan")))["lower_median"])
    assert s["sd"] == pytest.approx(float(torch.cat([torch.ones(14), torch.tensor([2.0])]).double().std()))


def test_the_reference_is_labelled_optimistic_and_is_not_the_sensor_noise_floor():
    h = load_het()
    assert "optimistic" in h.LABEL and "not the sensor's calibrated noise floor" in h.LABEL
    assert "OPTIMISTIC FINITE-SAMPLE REFERENCE" in h.__doc__ and "NOT the calibrated noise floor" in h.__doc__


def test_reference_for_an_equal_variance_gaussian_centres_on_slope_one_and_scatter_shrinks_with_n_pos():
    h = load_het()
    tok = 4
    d = torch.full((tok, tok), 1.5, dtype=torch.double)
    few = h.reference(d, torch.full((tok, tok), 40.0), 4000, 12, torch.Generator().manual_seed(1))
    many = h.reference(d, torch.full((tok, tok), 400.0), 4000, 12, torch.Generator().manual_seed(1))
    assert few["lower_median"]["p50"] == pytest.approx(1.0, abs=0.12) and many["lower_median"]["p50"] == pytest.approx(1.0, abs=0.05)
    assert many["across_position_sd"]["p50"] < few["across_position_sd"]["p50"]
    assert few["n_null_per_position_assumed"] == 4000 and few["label"] == h.LABEL
    again = h.reference(d, torch.full((tok, tok), 40.0), 4000, 12, torch.Generator().manual_seed(1))
    assert again == few                                                                                 # seeded -> reproducible


def test_positions_with_too_few_positives_or_no_d_are_skipped_in_the_reference():
    h = load_het()
    tok = 3
    d = torch.full((tok, tok), 1.0, dtype=torch.double); d[0, 0] = float("nan")
    n = torch.full((tok, tok), 30.0); n[1, 1] = 1.0
    r = h.reference(d, n, 500, 3, torch.Generator().manual_seed(0))
    assert r["lower_median"] is not None                                                                # still computable from the 7 simulated positions


def test_analyse_run_reads_a_self_test_run_json_and_changes_nothing_in_it(tmp_path, monkeypatch, capsys):
    m = load_script()
    run = m.run(tiny_args(seed=2, n_test=300, n_cal=200)); run.pop("_provenance_sidecar", None)
    p = tmp_path / "run.json"
    p.write_text(json.dumps(run, default=lambda o: None))
    before = p.read_bytes()
    h = load_het()
    out = tmp_path / "het.json"
    h.main(["--run", str(p), "--n-ref", "3", "--reference-n-null", "300", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["status"].startswith("REPORT-ONLY") and res["runs"][0]["verdict_uses_this"] is False
    assert list(res["runs"][0]["cells"]) == [res["runs"][0]["operating_cell"]]                          # --cells op is the default
    cell = next(iter(res["runs"][0]["cells"].values()))
    key = res["runs"][0]["operating_cell"]
    assert set(cell) == {"observed", "reference"} and cell["observed"]["n_positions"] == len(run["native"][key]["zroc_slope"]) ** 2
    assert p.read_bytes() == before                                                                      # input untouched
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        h.main(["--run", str(p), "--n-ref", "2", "--reference-n-null", "100", "--out", str(out)])
    capsys.readouterr()                                                                                  # discard the output of the calls above
    h.main(["--run", str(p), "--cells", "all", "--n-ref", "2", "--reference-n-null", "100"])
    printed = json.loads(capsys.readouterr().out)                                                        # no --out: the JSON goes to stdout only
    assert set(printed["runs"][0]["cells"]) == set(run["native"])                                        # --cells all: every cell of `native`


def test_the_script_has_no_threshold_decision_and_the_gate_does_not_import_it():
    src = (ROOT / "scripts" / "rhan_im" / "gate1_slope_heterogeneity.py").read_text()
    for banned in ("pass_all", "sdt_adequate", "assert ", "verdict_draft", "evaluate"):                  # no gate rule is evaluated here
        assert banned not in src
    from rhan_im import gate1 as g1
    assert "scripts/rhan_im/gate1_slope_heterogeneity.py" not in g1.MEASUREMENT_CODE_FILES              # runs are not required to agree on it: no verdict path
    for f in ("rhan_im/gate1.py", "scripts/rhan_im/gate1_prereg.py", "scripts/rhan_im/gate1_measure_dprime.py"):
        text = (ROOT / f).read_text()
        assert "import gate1_slope_heterogeneity" not in text and "from gate1_slope_heterogeneity" not in text
        assert "gate1_slope_heterogeneity" not in text.replace('"scripts/rhan_im/gate1_slope_heterogeneity.py"]   # C4: the report-only analysis is hashed too', "")   # only the hash list names it
