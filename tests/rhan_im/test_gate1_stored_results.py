"""The stored operating-characteristics file (docs/research/RHAN_IM_gate1_operating_characteristics.json) against the CURRENT decision rule.

The tables in the pre-registration draft and in RHAN_IM_RESULTS.md are generated from that file. If `rhan_im.gate1.verdict` changes, the
file must be regenerated or the documents are describing a rule that no longer exists; this test makes that visible instead of silent.
It re-judges every stored replicate summary with the stored thresholds and compares every stored verdict field, and checks that the stored
per-condition rates are the means of the stored replicate verdicts.
"""
import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rhan_im import gate1 as g1  # noqa: E402

OC = ROOT / "docs" / "research" / "RHAN_IM_gate1_operating_characteristics.json"


@pytest.fixture(scope="module")
def oc():
    assert OC.exists(), f"{OC} is missing: the pre-registration draft and RHAN_IM_RESULTS.md quote it"
    return json.loads(OC.read_text())


def _same(old, new):
    if isinstance(old, bool) or isinstance(new, bool) or isinstance(old, str) or isinstance(new, str):
        return old == new
    old = float("nan") if old is None else float(old)             # strict JSON stores NaN as null
    new = float("nan") if new is None else float(new)
    return (math.isnan(old) and math.isnan(new)) or abs(old - new) <= 1e-9 * max(1.0, abs(old))


def _thresholds(oc):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in oc["thresholds"].items()}


def test_the_stored_thresholds_are_the_current_defaults(oc):
    assert _thresholds(oc) == {k: (tuple(v) if isinstance(v, list) else v) for k, v in g1.DEFAULT_THRESHOLDS.items()}


def test_every_stored_verdict_is_what_the_current_rule_gives_for_the_stored_summary(oc):
    thr = _thresholds(oc)
    block = float(oc["plan"]["op_size"]) ** 2                                      # the older summaries do not store the block's token count
    checked, differences = 0, []
    for name, cond in oc["conditions"].items():
        for i, rep in enumerate(cond["per_replicate"]):
            summary = {"op_block_tokens": block, **rep["summary"]}
            now = g1.verdict(summary, thr)
            for key, old in rep["verdict"].items():
                if not _same(old, now[key]):
                    differences.append((name, i, key, old, now[key]))
            checked += 1
    assert checked >= 60
    assert not differences, f"the stored operating characteristics no longer follow the rule (regenerate them): {differences[:5]}"


def test_the_stored_rates_are_the_means_of_the_stored_replicate_verdicts(oc):
    for name, cond in oc["conditions"].items():
        reps = cond["per_replicate"]
        for rate, key in (("rate_sdt_adequate", "sdt_adequate"), ("rate_informative", "informative"), ("rate_looking_closer_pays", "looking_closer_pays"),
                          ("rate_pass_all", "pass_all")):
            assert cond[rate] == pytest.approx(sum(bool(r["verdict"][key]) for r in reps) / len(reps)), (name, rate)
