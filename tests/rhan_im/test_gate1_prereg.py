"""The gate-1 freeze / aggregate tool: hashes, refusal to freeze an unknown or dirty tree, frozen thresholds, aggregation rule,
deviations (which make the aggregate INVALID, never a quiet PASS), unmeasured quantities (INCONCLUSIVE), checked secondary runs."""
import hashlib
import importlib.util
import json
import math
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rhan_im import gate1 as g1  # noqa: E402

_spec = importlib.util.spec_from_file_location("gate1_prereg", ROOT / "scripts" / "rhan_im" / "gate1_prereg.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

PRIMARY = b"fake primary checkpoint bytes"
CODE = {f: g1.file_sha256(ROOT / f) for f in g1.MEASUREMENT_CODE_FILES}                      # what a run on this checkout records


def fake_listing(n_files_per_class):
    rel = sorted(f"{c}/{i}.jpg" for c in ("a", "b") for i in range(n_files_per_class))        # the fixture's fake ImageFolder, names only
    return {"sha256": hashlib.sha256("\n".join(rel).encode()).hexdigest(), "n_files": len(rel)}


LISTING = {"train_path_listing": fake_listing(3), "val_path_listing": fake_listing(2)}


def make_data(tmp_path):
    data = tmp_path / "imagenet100"
    for split, n in (("train", 3), ("val", 2)):
        for c in ("a", "b"):
            (data / split / c).mkdir(parents=True)
            for i in range(n):
                (data / split / c / f"{i}.jpg").write_bytes(b"x")
    return data


@pytest.fixture()
def frozen(tmp_path):
    ck = tmp_path / "ff112_seed17.pth"
    ck.write_bytes(PRIMARY)
    sec = tmp_path / "ff112_seed42.pth"
    sec.write_bytes(b"other")
    return P.freeze(ck, [sec], make_data(tmp_path), allow_dirty=True, allow_other_primary=True), ck


def run(seed, sha, name="ff112_seed17.pth", over=None, cfg=None, code=None):
    summary = dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=3.0, d_gist=1.0, rho_null=0.05,
                   vif_null_op=1.0, vif_pos_op=1.0, op_block_tokens=16.0)
    summary.update(over or {})
    return {"config": {**P.PLANNED_CONFIG, "seed": seed, **(cfg or {})}, "trunk": {"sha256": sha, "checkpoint": name},
            "code_sha256": dict(CODE if code is None else code), "data_listing": json.loads(json.dumps(LISTING)), "summary": summary}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def three(pre=None, **kw):
    return [run(s, sha(PRIMARY), **kw) for s in (17, 42, 99)]


# ------------------------------------------------------------------------------------------------ freeze
def test_freeze_records_hashes_thresholds_and_listings(frozen):
    pre, ck = frozen
    assert pre["checkpoint_hashes"] == {"ff112_seed17.pth": sha(PRIMARY), "ff112_seed42.pth": sha(b"other")}
    assert pre["status"] == "FROZEN" and pre["protocol"]["primary_seeds"] == [17, 42, 99]
    for f in P.CODE_FILES:
        assert pre["code_and_data_hashes"][f] == sha((ROOT / f).read_bytes())
    assert "gen2_foundation/backbone.py" in pre["code_and_data_hashes"]                      # the trunk definition is part of the measurement
    assert "noesis_vision/models/recurrent_block.py" in pre["code_and_data_hashes"]          # ... and so is the module it imports its blocks from
    assert "scripts/rhan_im/gate1_prereg.py" in pre["code_and_data_hashes"]                  # ... and the code that applies the decision rule
    assert pre["code_and_data_hashes"]["train_path_listing"]["n_files"] == 6 and pre["code_and_data_hashes"]["val_path_listing"]["n_files"] == 4
    assert {k: pre["code_and_data_hashes"][k] for k in LISTING} == LISTING                  # the listing the fixture computes by hand is the tool's
    assert pre["rules"]["thresholds"] == json.loads(json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in g1.DEFAULT_THRESHOLDS.items()}))


def test_freeze_refuses_a_dirty_measurement_tree_and_records_the_commit_when_clean(tmp_path):
    repo = tmp_path / "repo"
    for f in P.CODE_FILES:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, repo / f)
    git = lambda *a: subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *a], cwd=repo, check=True, capture_output=True, text=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "baseline")
    ck = tmp_path / "c.pth"
    ck.write_bytes(PRIMARY)
    data = make_data(tmp_path)
    ok = P.freeze(ck, [], data, repo_root=repo, allow_other_primary=True)
    assert ok["code_dirty_at_freeze"] is False and ok["git_head"] == git("rev-parse", "HEAD").stdout.strip()
    (repo / "rhan_im" / "gate1.py").write_text((repo / "rhan_im" / "gate1.py").read_text() + "\n# edited after review\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        P.freeze(ck, [], data, repo_root=repo, allow_other_primary=True)
    assert P.freeze(ck, [], data, allow_dirty=True, repo_root=repo, allow_other_primary=True)["code_dirty_at_freeze"] is True


def test_freeze_does_not_read_an_unreadable_git_state_as_a_clean_tree(tmp_path, monkeypatch):
    """A reviewer's finding: a failed git call (no .git on a Colab upload, git missing) was read as 'nothing uncommitted'."""
    ck = tmp_path / "c.pth"
    ck.write_bytes(PRIMARY)
    data = make_data(tmp_path)
    monkeypatch.setattr(P, "git", lambda *a, **k: None)                                        # what git() returns when git cannot answer
    with pytest.raises(SystemExit, match="git state cannot be read"):
        P.freeze(ck, [], data, allow_other_primary=True)
    pre = P.freeze(ck, [], data, allow_dirty=True, allow_other_primary=True)
    assert pre["code_dirty_at_freeze"] is None and pre["git_head"] is None                     # recorded as unknown, not as clean


def test_git_helper_returns_none_when_git_cannot_answer(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(P.subprocess, "run", boom)
    assert P.git("status", cwd=tmp_path) is None
    monkeypatch.undo()
    err = subprocess.CalledProcessError(128, "git")
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(err))
    assert P.git("rev-parse", "HEAD", cwd=tmp_path) is None


def test_freeze_asserts_the_documented_primary_checkpoint_and_requires_the_data_root(tmp_path, monkeypatch):
    assert len(P.DOCUMENTED_PRIMARY_SHA256) == 64 and P.DOCUMENTED_PRIMARY_SHA256.startswith("d00fb750")   # foundation_ff_112_seed17_final_d00fb750.pth
    ck = tmp_path / "other.pth"
    ck.write_bytes(PRIMARY)
    data = make_data(tmp_path)
    with pytest.raises(SystemExit, match="documented ff_112 seed-17"):
        P.freeze(ck, [], data, allow_dirty=True)
    assert P.freeze(ck, [], data, allow_dirty=True, allow_other_primary=True)["protocol"]["primary_is_the_documented_checkpoint"] is False
    monkeypatch.setattr(P, "DOCUMENTED_PRIMARY_SHA256", sha(PRIMARY))
    assert P.freeze(ck, [], data, allow_dirty=True)["protocol"]["primary_is_the_documented_checkpoint"] is True
    with pytest.raises(SystemExit, match="--data-root is required"):
        P.freeze(ck, [], None, allow_dirty=True)
    twin = tmp_path / "sub"
    twin.mkdir()
    (twin / "other.pth").write_bytes(b"x")
    with pytest.raises(SystemExit, match="distinct"):
        P.freeze(ck, [twin / "other.pth"], data, allow_dirty=True)


# ------------------------------------------------------------------------------------------------ aggregate
def test_three_passing_runs_pass_with_no_deviation(frozen):
    pre, _ = frozen
    res = P.aggregate(pre, three())
    assert res["gate1"] == "PASS" and res["pass_counts"]["pass_all"] == 3 and res["deviations"] == [] and res["unmeasured"] == {}
    assert res["combination_rule"] == "sum" and res["token_evidence"] == "independent" and res["vif_to_apply"] == 1.0
    assert res["next_step"][0].startswith("M2")


def test_two_of_three_is_a_pass_and_one_of_three_is_a_fail_with_the_failed_criterion_named(frozen):
    pre, _ = frozen
    weak = {"d_native_centre": 1.0, "d_gist": 0.3}                                      # not informative, but still 3x better than gist
    two = P.aggregate(pre, [run(17, sha(PRIMARY)), run(42, sha(PRIMARY)), run(99, sha(PRIMARY), over=weak)])
    assert two["gate1"] == "PASS" and two["pass_counts"]["pass_all"] == 2 and two["failed_criteria"] == []
    one = P.aggregate(pre, [run(17, sha(PRIMARY)), run(42, sha(PRIMARY), over=weak), run(99, sha(PRIMARY), over=weak)])
    assert one["gate1"] == "FAIL" and one["failed_criteria"] == ["informative"] and "S3" in one["next_step"][0]
    slope = P.aggregate(pre, [run(s, sha(PRIMARY), over={"zroc_slope_median": 0.6}) for s in (17, 42, 99)])
    assert slope["gate1"] == "FAIL" and slope["failed_criteria"] == ["sdt_adequate"] and "heteroscedastic" in slope["next_step"][0]


def test_combination_rule_is_sum_only_if_every_run_says_sum(frozen):
    pre, _ = frozen
    res = P.aggregate(pre, [run(17, sha(PRIMARY)), run(42, sha(PRIMARY), over={"rho_null": 0.4}), run(99, sha(PRIMARY))])
    assert res["combination_rule"] == "best"


def test_vif_correction_uses_the_median_over_runs_of_the_null_vif(frozen):
    pre, _ = frozen
    runs = [run(17, sha(PRIMARY), over={"vif_null_op": 3.0, "vif_pos_op": 3.1}),
            run(42, sha(PRIMARY), over={"vif_null_op": 1.5, "vif_pos_op": 4.0}),         # a large VIF_pos alone must NOT drive the correction
            run(99, sha(PRIMARY), over={"vif_null_op": 6.0, "vif_pos_op": 5.0})]
    res = P.aggregate(pre, runs)
    assert res["vif_null_median"] == 3.0 and res["vif_to_apply"] == 3.0 and res["token_evidence"].startswith("correct")
    low = P.aggregate(pre, [run(s, sha(PRIMARY), over={"vif_null_op": 1.1, "vif_pos_op": 1.2}) for s in (17, 42, 99)])
    assert low["token_evidence"] == "independent" and low["vif_to_apply"] == 1.0 and low["shared_difficulty"] is False


def test_an_unmeasured_vif_or_block_size_makes_the_aggregate_inconclusive_never_a_pass(frozen):
    """A reviewer's finding: an unmeasured VIF_null read as 'independent tokens' and could only help a pass."""
    pre, _ = frozen
    nan = P.aggregate(pre, [run(s, sha(PRIMARY), over={"vif_null_op": float("nan")}) for s in (17, 42, 99)])
    assert nan["gate1"].startswith("INCONCLUSIVE") and nan["gate1_if_deviations_and_unmeasured_quantities_were_ignored"] == "FAIL"
    assert nan["token_evidence"] == "unmeasured" and math.isnan(nan["vif_to_apply"])
    assert nan["unmeasured"] == {"17": ["vif_null_op"], "42": ["vif_null_op"], "99": ["vif_null_op"]}
    assert nan["next_step"][0].startswith("a required quantity was not measured")
    one = P.aggregate(pre, [run(17, sha(PRIMARY), over={"vif_null_op": float("nan")}), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
    assert one["gate1"].startswith("INCONCLUSIVE") and one["unmeasured"] == {"17": ["vif_null_op"]}     # two clean passes do not rescue it
    blk = P.aggregate(pre, [run(s, sha(PRIMARY), over={"op_block_tokens": float("nan")}) for s in (17, 42, 99)])
    assert blk["gate1"].startswith("INCONCLUSIVE") and blk["unmeasured"]["17"] == ["op_block_tokens"]


def test_shared_difficulty_needs_a_majority_of_runs_is_never_shrunk_and_reports_the_spread(frozen):
    pre, _ = frozen
    n, d, cv = 16, 0.6, 0.4
    c = d * d * cv * cv
    vp = (1.0 + n * c) / (1 + c)                                                        # the positive-scene VIF that a spread of 0.4 implies
    hit = {"vif_null_op": 1.0, "vif_pos_op": vp, "d_native_centre": d, "op_block_tokens": float(n)}
    two = P.aggregate(pre, [run(17, sha(PRIMARY), over=hit), run(42, sha(PRIMARY), over=hit), run(99, sha(PRIMARY))])
    assert two["shared_difficulty"] is True and two["shared_difficulty_runs"] == 2 and two["shared_difficulty_assessed_runs"] == 3
    assert two["difficulty_cv_estimate"] == pytest.approx(cv, rel=1e-6)
    assert two["token_evidence"] == "independent" and two["vif_to_apply"] == 1.0         # flagged, not shrunk
    assert any("scale-mixture" in x for x in two["next_step"])
    one = P.aggregate(pre, [run(17, sha(PRIMARY), over=hit), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
    assert one["shared_difficulty"] is False and one["shared_difficulty_runs"] == 1
    both = {"vif_null_op": 2.5, "vif_pos_op": (2.5 + n * c) / (1 + c), "d_native_centre": d, "op_block_tokens": float(n)}   # correlated noise too
    res = P.aggregate(pre, [run(s, sha(PRIMARY), over=both) for s in (17, 42, 99)])
    assert res["token_evidence"].startswith("correct") and res["vif_to_apply"] == 2.5
    assert res["shared_difficulty"] is True and res["difficulty_cv_estimate"] == pytest.approx(cv, rel=1e-6)
    unassessed = P.aggregate(pre, [run(s, sha(PRIMARY), over={"vif_pos_op": float("nan")}) for s in (17, 42, 99)])
    assert unassessed["shared_difficulty"] is False and unassessed["shared_difficulty_assessed_runs"] == 0   # not assessed is reported as such


def test_the_frozen_thresholds_decide_not_the_ones_in_the_code_today(frozen):
    pre, _ = frozen
    pre["rules"]["thresholds"]["d_centre_min"] = 5.0
    res = P.aggregate(pre, three())                                                     # d = 3.0 passes the code's 1.5, not the frozen 5.0
    assert res["gate1"].startswith("FAIL") and res["failed_criteria"] == ["informative"]


def test_every_difference_from_the_frozen_protocol_is_listed_and_makes_the_aggregate_invalid(frozen):
    pre, _ = frozen
    res = P.aggregate(pre, [run(17, sha(PRIMARY), cfg={"n_test": 3000}), run(42, "0" * 64), run(7, sha(PRIMARY)), run(99, sha(PRIMARY), cfg={"self_test": True})])
    text = "\n".join(res["deviations"])
    assert res["gate1"].startswith("INVALID")
    for needle in ("n_test = 3000", "is not the frozen primary checkpoint", "primary seeds", "4 primary runs", "self-test"):
        assert needle in text, needle
    pre["code_and_data_hashes"]["rhan_im/gate1.py"] = "0" * 64
    bad = P.aggregate(pre, three())
    assert bad["gate1"].startswith("INVALID") and any("code file differs from the frozen hash: rhan_im/gate1.py" in d for d in bad["deviations"])
    assert any("ran with a different rhan_im/gate1.py" in d for d in bad["deviations"])      # and the run's own recorded hash disagrees too


def test_a_self_test_a_wrong_checkpoint_or_a_missing_run_never_gives_a_pass(frozen):
    """A reviewer's finding: these used to give 'PASS (WITH DEVIATIONS)'. The verdict the rule would have given is kept, separately."""
    pre, _ = frozen
    for runs, needle in ((three(cfg={"self_test": True}), "self-test"),
                         ([run(s, sha(b"another checkpoint")) for s in (17, 42, 99)], "is not the frozen primary checkpoint"),
                         ([run(17, sha(PRIMARY)), run(42, sha(PRIMARY))], "2 primary runs"),
                         ([run(s, sha(PRIMARY)) for s in (17, 17, 99)], "primary seeds")):
        res = P.aggregate(pre, runs)
        assert res["gate1"].startswith("INVALID") and res["gate1_if_deviations_and_unmeasured_quantities_were_ignored"] == "PASS"
        assert any(needle in d for d in res["deviations"]) and res["next_step"] == [P.NEXT_STEP["invalid"]]


def test_a_missing_configuration_key_is_a_deviation(frozen):
    """A reviewer's finding: `if k in cfg and cfg[k] != v` made an absent key invisible."""
    pre, _ = frozen
    r = run(17, sha(PRIMARY))
    del r["config"]["n_test"]
    res = P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
    assert res["gate1"].startswith("INVALID") and any("config key 'n_test' is missing" in d for d in res["deviations"])


def test_a_run_on_other_code_or_other_data_is_a_deviation(frozen):
    pre, _ = frozen
    other = {**CODE, "noesis_vision/models/recurrent_block.py": "0" * 64}
    res = P.aggregate(pre, [run(17, sha(PRIMARY), code=other), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
    assert res["gate1"].startswith("INVALID") and any("ran with a different noesis_vision/models/recurrent_block.py" in d for d in res["deviations"])
    r = run(17, sha(PRIMARY))
    del r["code_sha256"]
    assert any("did not record the hashes of the code" in d for d in P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])
    r = run(17, sha(PRIMARY))
    r["data_listing"]["val_path_listing"] = {"sha256": "0" * 64, "n_files": 4}
    assert any("val_path_listing differs from the frozen listing" in d for d in P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])
    r = run(17, sha(PRIMARY))
    r["data_listing"] = None                                                            # e.g. a self-test run, or an older script
    assert any("train_path_listing was not recorded" in d for d in P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])


def test_a_frozen_file_that_does_not_pin_what_the_tool_checks_is_a_deviation(frozen):
    pre, _ = frozen
    del pre["code_and_data_hashes"]["noesis_vision/models/foveation.py"]
    del pre["code_and_data_hashes"]["val_path_listing"]
    res = P.aggregate(pre, three())
    text = "\n".join(res["deviations"])
    assert res["gate1"].startswith("INVALID") and "does not pin noesis_vision/models/foveation.py" in text and "does not pin val_path_listing" in text


def test_secondary_runs_are_checked_like_primary_runs_and_an_invalid_one_is_ignored(frozen):
    pre, _ = frozen
    prim = three()
    weak = {"d_native_centre": 1.0, "d_gist": 0.3}
    res = P.aggregate(pre, prim, [run(17, sha(b"other"), name="ff112_seed42.pth", over=weak)])
    assert res["secondary_trunks"][0]["valid"] and res["trunk_dependent"] is True and res["gate1"] == "PASS"
    tampered = P.aggregate(pre, prim, [run(17, sha(b"tampered"), name="ff112_seed42.pth", over=weak)])
    s = tampered["secondary_trunks"][0]
    assert not s["valid"] and any("is not the frozen hash of ff112_seed42.pth" in d for d in s["deviations"])
    assert tampered["trunk_dependent"] is False and tampered["gate1"] == "PASS"            # an invalid secondary neither fails nor blocks the primary verdict
    unknown = P.aggregate(pre, prim, [run(17, sha(b"x"), name="not_frozen.pth")])["secondary_trunks"][0]
    assert not unknown["valid"] and any("not in the frozen checkpoint list" in d for d in unknown["deviations"])
    selftest = P.aggregate(pre, prim, [run(17, sha(b"other"), name="ff112_seed42.pth", cfg={"self_test": True}, over=weak)])["secondary_trunks"][0]
    assert not selftest["valid"] and any("self-test" in d for d in selftest["deviations"])
    cfgbad = P.aggregate(pre, prim, [run(17, sha(b"other"), name="ff112_seed42.pth", cfg={"n_test": 3000})])["secondary_trunks"][0]
    assert not cfgbad["valid"]
    nan = P.aggregate(pre, prim, [run(17, sha(b"other"), name="ff112_seed42.pth", over={"vif_null_op": float("nan")})])["secondary_trunks"][0]
    assert not nan["valid"] and nan["unmeasured"] == ["vif_null_op"]
    ok = P.aggregate(pre, prim, [run(17, sha(b"other"), name="ff112_seed42.pth")])
    assert ok["trunk_dependent"] is False


def test_a_frozen_file_is_never_overwritten(tmp_path, monkeypatch):
    ck = tmp_path / "c.pth"
    ck.write_bytes(PRIMARY)
    out = tmp_path / "prereg.json"
    out.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["x", "freeze", "--primary", str(ck), "--data-root", str(make_data(tmp_path)), "--allow-dirty", "--allow-other-primary", "--out", str(out)])
    with pytest.raises(SystemExit, match="never overwritten"):
        P.main()
    assert out.read_text() == "{}"


# ------------------------------------------------------------------------------------------------ the hash list is complete
def test_the_hash_list_covers_every_repository_module_the_measurement_loads():
    """A reviewer found the trunk's block definitions (noesis_vision/models/recurrent_block.py) missing from the freeze. Import the
    measurement script in a fresh interpreter, list the repository files it loads, and require each to be hashed or to be on an
    explicit list of modules that are loaded as a side effect of package __init__ files and do not enter the measurement."""
    code = (
        "import sys, json, pathlib, importlib.util\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "spec = importlib.util.spec_from_file_location('gate1_measure', root / 'scripts/rhan_im/gate1_measure_dprime.py')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['gate1_measure'] = m; spec.loader.exec_module(m)\n"
        "files = set()\n"
        "for mod in list(sys.modules.values()):\n"
        "    f = getattr(mod, '__file__', None)\n"
        "    if not f or not pathlib.Path(f).is_absolute():\n"
        "        continue\n"
        "    try:\n"
        "        files.add(str(pathlib.Path(f).resolve().relative_to(root)))\n"
        "    except ValueError:\n"
        "        pass\n"
        "print(json.dumps(sorted(files)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, check=True, cwd=str(ROOT))
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    side_effect_only = {                                                                  # imported by gen2_foundation/__init__.py and rhan_im/__init__.py, not used by the measurement
        "gen2_foundation/__init__.py", "gen2_foundation/ema.py", "gen2_foundation/eot.py", "gen2_foundation/flags.py", "gen2_foundation/gist.py",
        "gen2_foundation/precision.py", "gen2_foundation/recipe.py", "gen2_foundation/update_net_v2.py",
        "rhan_im/__init__.py", "rhan_im/belief.py", "rhan_im/policy.py", "rhan_im/stopping.py",
    }
    unhashed = sorted(f for f in loaded if f not in g1.MEASUREMENT_CODE_FILES and f not in side_effect_only)
    assert unhashed == [], f"loaded by the measurement but not hashed into the freeze: {unhashed}"
    assert set(g1.MEASUREMENT_CODE_FILES) <= loaded, sorted(set(g1.MEASUREMENT_CODE_FILES) - loaded)   # nothing listed that the measurement never loads
