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
import torch

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

# the files gate1_freeze_tools_v2.py would write for the fixture's two-folder "dataset" (a = target, b = background); built by hand here
SPLIT = {"status": "FINAL", "seed": 5, "targets": [{"wnid": "n00000001", "dirname": "a"}], "backgrounds": [{"wnid": "n00000002", "dirname": "b"}]}
SPLIT["hashes"] = {"class_split_sha256": g1.class_split_sha256(5, SPLIT["targets"], SPLIT["backgrounds"])}
POOL_LISTS = ({"a": ["0.jpg"], "b": ["0.jpg"]}, {"a": ["1.jpg"], "b": ["1.jpg"]})
POOLS = {"status": "FINAL", "rule_id": "category-hash-halves-v1", "class_split": SPLIT["hashes"]["class_split_sha256"],
         "calibration_pool_part_0_of_2": POOL_LISTS[0], "test_pool_part_1_of_2": POOL_LISTS[1],
         "listing_binding": {"val": {"repo_listing_sha256": LISTING["val_path_listing"]["sha256"]}}}
POOLS["pool_assignment_sha256"] = hashlib.sha256(g1.canonical_json_bytes({
    "rule_id": POOLS["rule_id"], "class_split": POOLS["class_split"], "val_repo_listing_sha256": LISTING["val_path_listing"]["sha256"],
    "calibration": POOL_LISTS[0], "test": POOL_LISTS[1]})).hexdigest()
CLASS_LISTS = g1.class_lists_sha256(["a"], ["b"])


def cf(tmp_path):
    """keyword arguments for P.freeze: the two class files of the fixture dataset"""
    return dict(zip(("class_split", "pools"), write_class_files(tmp_path)))


def write_class_files(tmp_path, split=None, pools=None):
    paths = []
    for name, obj in (("class_split.json", split or SPLIT), ("pools.json", pools or POOLS)):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        paths.append(p)
    return paths


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
    cs, pl = write_class_files(tmp_path)
    return P.freeze(ck, [sec], make_data(tmp_path), allow_dirty=True, allow_other_primary=True, class_split=cs, pools=pl), ck


def run(seed, sha, name="ff112_seed17.pth", over=None, cfg=None, code=None):
    summary = dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=3.0, d_gist=1.0, rho_null=0.05,
                   vif_null_op=1.0, vif_pos_op=1.0, op_block_tokens=16.0)
    summary.update(over or {})
    c = {**P.PLANNED_CONFIG, "seed": seed, "head_init_seed": seed, **(cfg or {})}
    return {"config": c, "trunk": {"sha256": sha, "checkpoint": name},
            "code_sha256": dict(CODE if code is None else code), "data_listing": json.loads(json.dumps(LISTING)), "summary": summary,
            "class_split": {"mode": "class_split_file", "class_split_sha256": SPLIT["hashes"]["class_split_sha256"], "class_lists_sha256": CLASS_LISTS},
            "pools": {"pool_assignment_sha256": POOLS["pool_assignment_sha256"]},
            "head": {"head_init_seed": c["head_init_seed"], "final_weights_sha256": "c" * 64},
            "provenance": {"sidecar_sha256": "a" * 64, "file": f"run{seed}.provenance.json", "link_sha256": g1.provenance_link_sha256(g1.provenance_link(
                c["seed"], c["head_init_seed"], sha, CLASS_LISTS, POOLS["pool_assignment_sha256"], c["n_cal"], c["n_test"], min(c["n_pairs"], c["n_test"])))},
            "environment": {"python": "3.13", "torch": "2.x"}}


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
    ok = P.freeze(ck, [], data, repo_root=repo, allow_other_primary=True, **cf(tmp_path))
    assert ok["code_dirty_at_freeze"] is False and ok["git_head"] == git("rev-parse", "HEAD").stdout.strip()
    (repo / "rhan_im" / "gate1.py").write_text((repo / "rhan_im" / "gate1.py").read_text() + "\n# edited after review\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        P.freeze(ck, [], data, repo_root=repo, allow_other_primary=True)
    assert P.freeze(ck, [], data, allow_dirty=True, repo_root=repo, allow_other_primary=True, **cf(tmp_path))["code_dirty_at_freeze"] is True


def test_freeze_does_not_read_an_unreadable_git_state_as_a_clean_tree(tmp_path, monkeypatch):
    """A reviewer's finding: a failed git call (no .git on a Colab upload, git missing) was read as 'nothing uncommitted'."""
    ck = tmp_path / "c.pth"
    ck.write_bytes(PRIMARY)
    data = make_data(tmp_path)
    monkeypatch.setattr(P, "git", lambda *a, **k: None)                                        # what git() returns when git cannot answer
    with pytest.raises(SystemExit, match="git state cannot be read"):
        P.freeze(ck, [], data, allow_other_primary=True)
    pre = P.freeze(ck, [], data, allow_dirty=True, allow_other_primary=True, **cf(tmp_path))
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
    assert P.freeze(ck, [], data, allow_dirty=True, allow_other_primary=True, **cf(tmp_path))["protocol"]["primary_is_the_documented_checkpoint"] is False
    monkeypatch.setattr(P, "DOCUMENTED_PRIMARY_SHA256", sha(PRIMARY))
    assert P.freeze(ck, [], data, allow_dirty=True, **cf(tmp_path))["protocol"]["primary_is_the_documented_checkpoint"] is True
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
    cs, pl = write_class_files(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "freeze", "--primary", str(ck), "--data-root", str(make_data(tmp_path)), "--class-split", str(cs), "--pools", str(pl),
                                      "--allow-dirty", "--allow-other-primary", "--out", str(out)])
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
        "        rel = pathlib.Path(f).resolve().relative_to(root)\n"
        "        if 'site-packages' in rel.parts or '.venv' in rel.parts or 'dist-packages' in rel.parts:\n"
        "            continue\n"
        "        files.add(str(rel))\n"
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


# ------------------------------------------------------------------------------------------------ unreadable runs, seed types, other unmeasured quantities
def test_a_run_that_cannot_be_read_is_a_deviation_not_a_crash_and_not_a_vote(frozen):
    """Found by my own mutation fuzz: a missing summary key or a string where a number belongs raised a traceback instead of a label."""
    pre, _ = frozen
    breakers = (lambda r: r.pop("summary"), lambda r: r["summary"].pop("d_gist"), lambda r: r["summary"].pop("op_block_tokens"),
                lambda r: r["summary"].update(d_gist="x"), lambda r: r["summary"].update(vif_null_op=[1.0]), lambda r: r["summary"].update(d_native_centre=True),
                lambda r: r["summary"].update(d_gist_full="1.0"), lambda r: r.update(config=[1, 2]), lambda r: r.update(trunk=None))
    for i, breaker in enumerate(breakers):
        runs = three()
        breaker(runs[1])
        res = P.aggregate(pre, runs, repo_root=ROOT)
        assert res["gate1"].startswith("INVALID") and res["deviations"], i
        assert res["gate1_if_deviations_and_unmeasured_quantities_were_ignored"] in ("PASS", "FAIL")
    res = P.aggregate(pre, three()[:2] + ["not a run"], repo_root=ROOT)
    assert res["gate1"].startswith("INVALID") and any("not a JSON object" in d for d in res["deviations"])
    broken = run(17, sha(b"other"), name="ff112_seed42.pth")
    del broken["summary"]
    sec = P.aggregate(pre, three(), [broken, "junk"], repo_root=ROOT)["secondary_trunks"]
    assert [s["valid"] for s in sec] == [False, False] and sec[0]["pass_all"] is False


def test_seeds_must_be_the_frozen_integers(frozen):
    pre, _ = frozen
    for seeds in (("17", "42", "99"), (17.0, 42.0, 99.0), (True, 42, 99), (17, 42, None)):
        res = P.aggregate(pre, [run(s, sha(PRIMARY)) for s in seeds], repo_root=ROOT)
        assert res["gate1"].startswith("INVALID") and any("primary seeds" in d for d in res["deviations"]), seeds


def test_a_missing_centre_gist_or_null_sd_in_one_run_is_not_out_voted(frozen):
    """'Could not be measured' is not 'failed', and two clean runs do not rescue a third whose measurement is missing."""
    pre, _ = frozen
    for key, name in (("d_native_centre", "d_native_centre"), ("d_gist", "d_gist"), ("null_sd_median", "null_sd_median")):
        for bad in (None, float("nan"), float("inf")):
            res = P.aggregate(pre, [run(17, sha(PRIMARY), over={key: bad}), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))], repo_root=ROOT)
            assert res["gate1"].startswith("INCONCLUSIVE") and res["unmeasured"] == {"17": [name]}, (key, bad)
    res = P.aggregate(pre, [run(17, sha(PRIMARY), over={"zroc_slope_median": float("nan"), "pos_sd_median": float("nan")}), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))], repo_root=ROOT)
    assert res["gate1"].startswith("INCONCLUSIVE") and res["unmeasured"] == {"17": ["zroc_slope_median"]}
    ok = P.aggregate(pre, [run(17, sha(PRIMARY), over={"zroc_slope_median": float("nan")}), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))], repo_root=ROOT)   # the sd-ratio fallback measures it
    assert ok["gate1"] == "PASS" and ok["unmeasured"] == {}


# ------------------------------------------------------------------------------------------------ C1, C2, C5, C7, C8 (proposed)
def test_freeze_pins_the_class_split_the_pools_and_the_secondary_seed(frozen):
    pre, _ = frozen
    p = pre["protocol"]
    assert p["class_split"]["class_split_sha256"] == SPLIT["hashes"]["class_split_sha256"] and p["class_split"]["class_lists_sha256"] == CLASS_LISTS
    assert p["pools"]["pool_assignment_sha256"] == POOLS["pool_assignment_sha256"] and p["secondary_seed"] == 17
    assert "head_init_seed" in p["head_init_seed_rule"] and "original_split_report_only" not in p and "not part of Gate 1" in p["original_split_run"]
    assert "scripts/rhan_im/gate1_slope_heterogeneity.py" in pre["code_and_data_hashes"]


def test_freeze_refuses_without_split_or_pools_or_with_a_non_final_or_inconsistent_one(tmp_path):
    ck = tmp_path / "ff112_seed17.pth"; ck.write_bytes(PRIMARY)
    data = make_data(tmp_path)
    cs, pl = write_class_files(tmp_path)
    def go(**kw):
        return P.freeze(ck, [], data, allow_dirty=True, allow_other_primary=True, **{"class_split": cs, "pools": pl, **kw})
    with pytest.raises(SystemExit, match="required"):
        P.freeze(ck, [], data, allow_dirty=True, allow_other_primary=True)
    with pytest.raises(SystemExit, match="required"):
        go(pools=None)
    cand = {**SPLIT, "status": "CANDIDATE"}
    cs2, _ = write_class_files(tmp_path / ".." / tmp_path.name, split=cand)
    with pytest.raises(SystemExit, match="FINAL"):
        go(class_split=cs2)
    bad = json.loads(json.dumps(SPLIT)); bad["hashes"]["class_split_sha256"] = "0" * 64
    cs3, _ = write_class_files(tmp_path, split=bad)
    with pytest.raises(SystemExit, match="recomputed"):
        go(class_split=cs3)
    other_listing = json.loads(json.dumps(POOLS)); other_listing["listing_binding"]["val"]["repo_listing_sha256"] = "1" * 64
    other_listing["pool_assignment_sha256"] = hashlib.sha256(g1.canonical_json_bytes({
        "rule_id": POOLS["rule_id"], "class_split": POOLS["class_split"], "val_repo_listing_sha256": "1" * 64, "calibration": POOL_LISTS[0], "test": POOL_LISTS[1]})).hexdigest()
    _, pl2 = write_class_files(tmp_path, pools=other_listing)
    with pytest.raises(SystemExit, match="different val listing"):
        go(pools=pl2)
    with pytest.raises(TypeError):                                                                     # the original-split option no longer exists (OI-06)
        go(legacy_split=cs)


def test_the_class_split_the_pools_and_the_head_seed_of_every_primary_run_are_checked(frozen):
    pre, _ = frozen
    r = run(17, sha(PRIMARY)); r["class_split"]["class_split_sha256"] = "0" * 64
    r2 = run(42, sha(PRIMARY)); r2["class_split"]["class_lists_sha256"] = "0" * 64
    r3 = run(99, sha(PRIMARY)); r3["pools"]["pool_assignment_sha256"] = "0" * 64
    res = P.aggregate(pre, [r, r2, r3])
    text = "\n".join(res["deviations"])
    assert res["gate1"].startswith("INVALID")
    for needle in ("class_split_sha256", "class_lists_sha256", "pool_assignment_sha256"):
        assert needle in text, needle
    for mode in ("legacy_sorted_dirnames", None, "self_test"):                                         # no fallback mode is accepted
        wrong = run(17, sha(PRIMARY)); wrong["class_split"]["mode"] = mode
        assert any("class split mode" in d for d in P.aggregate(pre, [wrong, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"]), mode
    no_pools = run(17, sha(PRIMARY)); no_pools["pools"] = None
    assert any("pool_assignment_sha256" in d for d in P.aggregate(pre, [no_pools, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])
    for seed_cfg in ({"head_init_seed": 0}, {"head_init_seed": 18}, {"head_init_seed": None}, {"head_init_seed": True}):
        res = P.aggregate(pre, [run(17, sha(PRIMARY), cfg=seed_cfg), run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
        assert any("head_init_seed" in d and "measurement seed" in d for d in res["deviations"]), seed_cfg
    missing = run(17, sha(PRIMARY)); del missing["config"]["head_init_seed"]
    assert any("head_init_seed None" in d for d in P.aggregate(pre, [missing, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])
    inconsistent = run(17, sha(PRIMARY)); inconsistent["head"]["head_init_seed"] = 0
    assert any("head record missing or inconsistent" in d for d in P.aggregate(pre, [inconsistent, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])


def test_a_run_without_provenance_hashes_or_runtime_record_is_a_deviation(frozen):
    pre, _ = frozen
    for key, needle in (("provenance", "provenance sidecar hash"), ("environment", "runtime / dependency metadata"), ("head", "head record")):
        r = run(17, sha(PRIMARY)); del r[key]
        res = P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
        assert res["gate1"].startswith("INVALID") and any(needle in d for d in res["deviations"]), key
    r = run(17, sha(PRIMARY)); r["provenance"]["sidecar_sha256"] = "xyz"
    assert any("provenance sidecar hash" in d for d in P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])


def test_a_frozen_file_without_the_new_pins_is_a_deviation(frozen):
    pre, _ = frozen
    for k in ("class_split", "pools", "secondary_seed"):
        q = json.loads(json.dumps(pre)); del q["protocol"][k]
        assert any(f"does not pin protocol.{k}" in d for d in P.aggregate(q, three())["deviations"]), k


def test_secondary_runs_must_use_the_frozen_secondary_seed(frozen):
    pre, _ = frozen
    prim = three()
    ok = P.aggregate(pre, prim, [run(17, sha(b"other"), name="ff112_seed42.pth")])["secondary_trunks"][0]
    assert ok["valid"]
    bad = P.aggregate(pre, prim, [run(42, sha(b"other"), name="ff112_seed42.pth")])["secondary_trunks"][0]
    assert not bad["valid"] and any("measurement seed 42, frozen 17" in d for d in bad["deviations"])


def test_there_is_no_original_split_run_in_the_aggregator_or_the_freeze():
    """OI-06: the original-split report-only run is not part of Gate 1; the options and the output section are gone."""
    import inspect
    assert "original_split" not in inspect.signature(P.aggregate).parameters and "legacy_split" not in inspect.signature(P.freeze).parameters
    assert "kind" not in inspect.signature(P.run_deviations).parameters
    src = (ROOT / "scripts" / "rhan_im" / "gate1_prereg.py").read_text()
    assert "--legacy-split" not in src and "--original-split-run" not in src and "original_split_report_only" not in src


def test_the_provenance_linkage_hash_must_be_the_hash_of_the_runs_own_protocol_fields(frozen):
    pre, _ = frozen
    ok = P.aggregate(pre, three())
    assert not any("linkage" in d for d in ok["deviations"])
    mutations = {
        "seed": lambda r: r["config"].__setitem__("n_cal", 3001),
        "checkpoint": lambda r: r["trunk"].__setitem__("sha256", "1" * 64),
        "class lists": lambda r: r["class_split"].__setitem__("class_lists_sha256", "2" * 64),
        "pools": lambda r: r["pools"].__setitem__("pool_assignment_sha256", "3" * 64),
        "n_test": lambda r: r["config"].__setitem__("n_test", 5999),
        "n_pairs": lambda r: r["config"].__setitem__("n_pairs", 1999),
        "link itself": lambda r: r["provenance"].__setitem__("link_sha256", "4" * 64),
    }
    for name, mutate in mutations.items():
        r = run(17, sha(PRIMARY)); mutate(r)
        res = P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])
        assert any("provenance linkage hash" in d for d in res["deviations"]), name
    r = run(17, sha(PRIMARY)); del r["config"]["n_test"]
    assert any("could not be recomputed" in d for d in P.aggregate(pre, [r, run(42, sha(PRIMARY)), run(99, sha(PRIMARY))])["deviations"])


def test_the_sidecar_file_next_to_a_run_is_checked(tmp_path, frozen):
    pre, _ = frozen
    r = run(17, sha(PRIMARY))
    path = tmp_path / "run17.json"
    assert P.sidecar_file_problems(r, path, "seed 17") and "missing" in P.sidecar_file_problems(r, path, "seed 17")[0]
    side = tmp_path / "run17.provenance.json"
    side.write_bytes(b'{"x":1}')
    assert "does not hash" in P.sidecar_file_problems(r, path, "seed 17")[0]
    r["provenance"]["sidecar_sha256"] = hashlib.sha256(b'{"x":1}').hexdigest()
    assert P.sidecar_file_problems(r, path, "seed 17") == []
    for bad in (None, "", "../run17.provenance.json", "sub/x.json"):                                 # no file named, or a name that leaves the run's folder
        r2 = json.loads(json.dumps(r)); r2["provenance"]["file"] = bad
        assert "names no provenance sidecar" in P.sidecar_file_problems(r2, path, "seed 17")[0], bad
    r3 = json.loads(json.dumps(r)); del r3["provenance"]
    assert "names no provenance sidecar" in P.sidecar_file_problems(r3, path, "seed 17")[0]
    # the aggregator turns the problems into deviations: primary and secondary alike
    res = P.aggregate(pre, three(), [run(17, sha(b"other"), name="ff112_seed42.pth")],
                      sidecar_problems={"seed 42": ["seed 42: provenance sidecar file x is missing next to the run JSON"],
                                        "secondary ff112_seed42.pth": ["secondary ff112_seed42.pth: provenance sidecar file y is missing next to the run JSON"]})
    assert res["gate1"].startswith("INVALID") and any("sidecar file x is missing" in d for d in res["deviations"])
    assert not res["secondary_trunks"][0]["valid"] and any("sidecar file y is missing" in d for d in res["secondary_trunks"][0]["deviations"])


def test_aggregate_main_reads_the_sidecar_files_from_disk(tmp_path, frozen, monkeypatch, capsys):
    pre, _ = frozen
    prereg = tmp_path / "prereg.json"; prereg.write_text(json.dumps(pre))
    paths = []
    for seed in (17, 42, 99):
        r = run(seed, sha(PRIMARY))
        side_bytes = f"sidecar {seed}".encode()
        r["provenance"]["sidecar_sha256"] = hashlib.sha256(side_bytes).hexdigest()
        (tmp_path / r["provenance"]["file"]).write_bytes(side_bytes)
        rp = tmp_path / f"run{seed}.json"; rp.write_text(json.dumps(r)); paths.append(rp)
    out = tmp_path / "verdict.json"
    argv = ["x", "aggregate", "--prereg", str(prereg), "--runs", *map(str, paths), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    P.main()
    assert not any("sidecar" in d for d in json.loads(out.read_text())["deviations"])
    (tmp_path / "run42.provenance.json").unlink()                                                      # delete one sidecar: the verdict must say so
    out.unlink()
    P.main()
    res = json.loads(out.read_text())
    assert res["gate1"].startswith("INVALID") and any("seed 42" in d and "sidecar file" in d and "missing" in d for d in res["deviations"])
    (tmp_path / "run42.provenance.json").write_bytes(b"altered")
    out.unlink()
    P.main()
    assert any("seed 42" in d and "does not hash" in d for d in json.loads(out.read_text())["deviations"])


def test_freeze_v2_with_s3_provenance(tmp_path):
    base_ck = tmp_path / "base.pth"
    base_ck.write_bytes(PRIMARY)
    base_sha = hashlib.sha256(PRIMARY).hexdigest()

    ck = tmp_path / "tuned_s3.pth"
    torch.save({"base_checkpoint_sha256": base_sha, "phase": "ff_112_s3", "arm": "s3_tuned"}, ck)

    s3_script = tmp_path / "train_s3.py"
    s3_script.write_bytes(b"s3 script content")
    s3_split = tmp_path / "s3_split.json"
    s3_split.write_bytes(b'{"train": [], "heldout": []}')
    data = make_data(tmp_path)
    cs, pl = write_class_files(tmp_path)

    # Missing any of the 3 S3 components must fail
    with pytest.raises(SystemExit, match="required for rhan_im_gate1_v2"):
        P.freeze(primary=ck, secondary=[], data_root=data, allow_dirty=True, class_split=cs, pools=pl,
                 protocol_version="rhan_im_gate1_v2", s3_script=None, s3_split=s3_split, s3_base_ckpt=base_ck)
    with pytest.raises(SystemExit, match="required for rhan_im_gate1_v2"):
        P.freeze(primary=ck, secondary=[], data_root=data, allow_dirty=True, class_split=cs, pools=pl,
                 protocol_version="rhan_im_gate1_v2", s3_script=s3_script, s3_split=None, s3_base_ckpt=base_ck)
    with pytest.raises(SystemExit, match="required for rhan_im_gate1_v2"):
        P.freeze(primary=ck, secondary=[], data_root=data, allow_dirty=True, class_split=cs, pools=pl,
                 protocol_version="rhan_im_gate1_v2", s3_script=s3_script, s3_split=s3_split, s3_base_ckpt=None)

    # Mismatched embedded base_checkpoint_sha256 must fail
    bad_ck = tmp_path / "bad_tuned.pth"
    torch.save({"base_checkpoint_sha256": "0" * 64}, bad_ck)
    with pytest.raises(SystemExit, match="does not match --s3-base-ckpt sha256"):
        P.freeze(primary=bad_ck, secondary=[], data_root=data, allow_dirty=True, class_split=cs, pools=pl,
                 protocol_version="rhan_im_gate1_v2", s3_script=s3_script, s3_split=s3_split, s3_base_ckpt=base_ck)

    res = P.freeze(
        primary=ck,
        secondary=[],
        data_root=data,
        allow_dirty=True,
        allow_other_primary=True,
        class_split=cs,
        pools=pl,
        protocol_version="rhan_im_gate1_v2",
        s3_script=s3_script,
        s3_split=s3_split,
        s3_base_ckpt=base_ck,
    )

    assert res["protocol_version"] == "rhan_im_gate1_v2"
    assert res["protocol"]["primary_checkpoint"] == "tuned_s3.pth"
    assert res["protocol"]["secondary_checkpoints"] == []
    assert "s3_provenance" in res
    assert res["s3_provenance"]["s3_script"]["sha256"] == hashlib.sha256(b"s3 script content").hexdigest()
    assert res["s3_provenance"]["s3_split"]["sha256"] == hashlib.sha256(b'{"train": [], "heldout": []}').hexdigest()
    assert res["s3_provenance"]["s3_base_checkpoint"]["file"] == "base.pth"
    assert res["s3_provenance"]["s3_base_checkpoint"]["sha256"] == base_sha


def test_v2_end_to_end_freeze_self_test_and_aggregate(tmp_path):
    """Dry-run of the v2 path end to end: freeze v2 with empty secondary, run self-test, aggregate."""
    import types
    import scripts.rhan_im.gate1_measure_dprime as m

    base_ck = tmp_path / "base.pth"
    base_ck.write_bytes(PRIMARY)
    base_sha = hashlib.sha256(PRIMARY).hexdigest()

    ck = tmp_path / "foundation_ff_112_s3_tuned.pth"
    torch.save({"base_checkpoint_sha256": base_sha, "phase": "ff_112_s3", "arm": "s3_tuned"}, ck)
    ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()

    s3_script = tmp_path / "train_s3.py"
    s3_script.write_bytes(b"print('s3')")
    s3_split = tmp_path / "s3_split.json"
    s3_split.write_bytes(b'{"train": [], "heldout": []}')
    data = make_data(tmp_path)
    cs, pl = write_class_files(tmp_path)

    # 1. Freeze v2
    pre_v2 = P.freeze(
        primary=ck,
        secondary=[],
        data_root=data,
        allow_dirty=True,
        class_split=cs,
        pools=pl,
        protocol_version="rhan_im_gate1_v2",
        s3_script=s3_script,
        s3_split=s3_split,
        s3_base_ckpt=base_ck,
    )
    assert pre_v2["protocol"]["secondary_checkpoints"] == []

    # 2. Run self-test runs
    self_runs = []
    for s in (17, 42, 99):
        args = types.SimpleNamespace(
            self_test=True, ckpt=None, data_root=None, device="cpu", canvas=112, window=56,
            stride_units=2, classes=4, sizes=[2, 3], alphas=[1.0], op_size=2, p_null=0.3,
            steps=2, batch=4, lr=2e-3, n_cal=10, n_test=10, n_pairs=5, seed=s, out=None,
            class_split=None, pools=None, chunk_scenes=10,
        )
        r = m.run(args)
        self_runs.append(r)

    # Aggregate self-test runs -> must be INVALID due to self-test, but aggregate completes cleanly
    agg_self = P.aggregate(pre_v2, self_runs, secondary=None, repo_root=ROOT)
    assert agg_self["gate1"].startswith("INVALID")
    assert any("self-test" in d for d in agg_self["deviations"])
    assert agg_self["runs"] == 3

    # 3. Clean mock runs matching the v2 freeze
    mock_runs = [run(s, ck_sha, name="foundation_ff_112_s3_tuned.pth") for s in (17, 42, 99)]
    # Write matching sidecar files so sidecar check passes
    sidecar_probs = {}
    agg_clean = P.aggregate(pre_v2, mock_runs, secondary=None, repo_root=ROOT, sidecar_problems=sidecar_probs)
    assert agg_clean["gate1"] == "PASS"
    assert agg_clean["deviations"] == []
    assert "secondary_trunks" not in agg_clean


