"""PROPOSED tests for the Gate-1 freeze corrections C1 (explicit class split, mandatory), C2 (head-init seed = measurement seed), C3 (operating-cell gist
overlap), C5 (runtime record), C6 (one gist minimum), C7 (per-scene provenance sidecar) and C8 (pools-file binding, mandatory).

SYNTHETIC: tiny fake ImageFolders and synthetic scores. They show the code does what the change says; they say nothing about the real dataset or checkpoint."""
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rhan_im import gate1 as g1  # noqa: E402
from test_gate1_script import load_script, make_fake_imagefolder, tiny_args  # noqa: E402


def chash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()      # independent of g1.canonical_json_bytes


def write_split(tmp, targets, backgrounds, seed=7, status="FINAL", name="split.json"):
    t = [{"wnid": f"n{100 + i:08d}", "dirname": d} for i, d in enumerate(targets)]
    b = [{"wnid": f"n{200 + i:08d}", "dirname": d} for i, d in enumerate(backgrounds)]
    key = lambda L: [{"wnid": r["wnid"], "dirname": r["dirname"]} for r in sorted(L, key=lambda r: r["wnid"])]
    h = chash({"seed": seed, "targets": key(t), "backgrounds": key(b)})                                    # the definition, written out by hand
    p = tmp / name
    p.write_text(json.dumps({"status": status, "seed": seed, "targets": t, "backgrounds": b, "hashes": {"class_split_sha256": h}}))
    return p, h


def write_pools(tmp, root, split_hash, name="pools.json"):
    classes = sorted(p.name for p in (root / "val").iterdir())
    cal, test = {}, {}
    for c in classes:
        names = sorted(p.name for p in (root / "val" / c).iterdir())
        cal[c], test[c] = names[0::2], names[1::2]
    val_hash = g1.listing_hash(root / "val")["sha256"]
    ah = chash({"rule_id": "r", "class_split": split_hash, "val_repo_listing_sha256": val_hash, "calibration": cal, "test": test})
    p = tmp / name
    p.write_text(json.dumps({"status": "FINAL", "rule_id": "r", "class_split": split_hash, "calibration_pool_part_0_of_2": cal, "test_pool_part_1_of_2": test,
                             "listing_binding": {"val": {"repo_listing_sha256": val_hash}}, "pool_assignment_sha256": ah}))
    return p, ah, cal, test


def real_args(root, **kw):
    base = dict(self_test=False, data_root=str(root), classes=4, steps=4, batch=8, n_cal=60, n_test=80, n_pairs=20, sizes=[2, 3])
    base.update(kw)
    return tiny_args(**base)


@pytest.fixture()
def tree(tmp_path):
    root = make_fake_imagefolder(tmp_path / "data", n_classes=8, n_files=6)
    return root


# ------------------------------------------------------------------------------------------------ C1
def test_explicit_class_split_replaces_the_sorted_name_rule_and_is_recorded(tree, tmp_path):
    m = load_script()
    classes = sorted(p.name for p in (tree / "train").iterdir())
    targets, bgs = [classes[i] for i in (1, 3, 5, 7)], [classes[i] for i in (0, 2, 4, 6)]            # NOT the first half
    sp, h = write_split(tmp_path, targets, bgs)
    pp, _, _, _ = write_pools(tmp_path, tree, h)
    r = m.run(real_args(tree, class_split=str(sp), pools=str(pp)))
    cs = r["class_split"]
    assert cs["mode"] == "class_split_file" and cs["class_split_sha256"] == h and cs["target_folders"] == targets and cs["background_folders"] == bgs
    assert cs["class_lists_sha256"] == chash({"targets": targets, "backgrounds": bgs}) and cs["file_sha256"] == hashlib.sha256(sp.read_bytes()).hexdigest()


def test_class_split_and_pools_are_mandatory_there_is_no_fallback(tree, tmp_path):
    """OI-05 / OI-06: no silent default. The original sorted-name split is gone from the measurement script."""
    m = load_script()
    classes = sorted(p.name for p in (tree / "train").iterdir())
    with pytest.raises(SystemExit, match="--class-split is required"):
        m.run(real_args(tree))                                                                          # used to fall back to the sorted-name halves
    sp, h = write_split(tmp_path, classes[:4], classes[4:])
    with pytest.raises(SystemExit, match="--pools is required"):
        m.run(real_args(tree, class_split=str(sp)))
    with pytest.raises(SystemExit, match="--class-split is required"):
        m.run(real_args(tree, class_split="", pools="x"))                                                # an empty path is not a split
    src = (ROOT / "scripts" / "rhan_im" / "gate1_measure_dprime.py").read_text()
    assert "legacy_sorted_dirnames" not in src


def test_class_split_file_is_refused_unless_it_is_final_consistent_and_exactly_the_folders_present(tree, tmp_path):
    m = load_script()
    classes = sorted(p.name for p in (tree / "train").iterdir())
    t, b = classes[:4], classes[4:]
    good, h = write_split(tmp_path, t, b)
    pp, _, _, _ = write_pools(tmp_path, tree, h)
    assert m.run(real_args(tree, class_split=str(good), pools=str(pp)))["class_split"]["class_split_sha256"] == h
    cand, _ = write_split(tmp_path, t, b, status="CANDIDATE", name="cand.json")
    with pytest.raises(SystemExit, match="FINAL"):
        m.run(real_args(tree, class_split=str(cand)))
    tampered = json.loads(good.read_text()); tampered["targets"][0]["dirname"] = classes[7]; tampered["targets"][3]["dirname"] = classes[0]
    (tmp_path / "t.json").write_text(json.dumps(tampered))
    with pytest.raises(SystemExit):
        m.run(real_args(tree, class_split=str(tmp_path / "t.json")))
    wrong_hash = json.loads(good.read_text()); wrong_hash["hashes"]["class_split_sha256"] = "0" * 64
    (tmp_path / "w.json").write_text(json.dumps(wrong_hash))
    with pytest.raises(SystemExit, match="recomputed"):
        m.run(real_args(tree, class_split=str(tmp_path / "w.json")))
    dup, _ = write_split(tmp_path, t, t, name="dup.json")                                              # a folder in both roles
    with pytest.raises(SystemExit, match="disjoint"):
        m.run(real_args(tree, class_split=str(dup)))
    (tree / "train" / "c99_extra").mkdir()                                                             # a folder the split does not list
    with pytest.raises(SystemExit, match="not exactly the folders"):
        m.run(real_args(tree, class_split=str(good)))
    with pytest.raises(SystemExit, match="targets"):                                                   # --classes disagrees with the file
        (tree / "train" / "c99_extra").rmdir()
        m.run(real_args(tree, class_split=str(good), classes=3))


def test_class_split_and_pools_are_refused_in_a_self_test(tmp_path):
    m = load_script()
    with pytest.raises(SystemExit):
        m.run(tiny_args(class_split=str(tmp_path / "x.json")))


def test_hash_helpers_match_a_by_hand_computation():
    t = [{"wnid": "n2", "dirname": "b"}, {"wnid": "n1", "dirname": "a"}]
    bg = [{"wnid": "n3", "dirname": "c"}]
    assert g1.class_split_sha256(9, t, bg) == chash({"seed": 9, "targets": [{"dirname": "a", "wnid": "n1"}, {"dirname": "b", "wnid": "n2"}], "backgrounds": [{"dirname": "c", "wnid": "n3"}]})
    assert g1.class_lists_sha256(["b", "a"], ["c"]) == chash({"targets": ["a", "b"], "backgrounds": ["c"]})
    assert g1.canonical_json_bytes({"é": 1, "a": [1, 2]}) == b'{"a":[1,2],"\\u00e9":1}'
    with pytest.raises(ValueError):
        g1.canonical_json_bytes({"x": float("nan")})


# ------------------------------------------------------------------------------------------------ C2
def test_head_init_seed_is_the_measurement_seed_recorded_and_reproducible(monkeypatch):
    m = load_script()
    a = m.run(tiny_args(seed=3)); a.pop("_provenance_sidecar")
    assert a["config"]["head_init_seed"] == 3 and a["head"]["head_init_seed"] == 3                     # = the run's own seed, written down
    b = m.run(tiny_args(seed=3, head_init_seed=3)); b.pop("_provenance_sidecar")
    assert a["head"] == b["head"]                                                                      # stating it changes nothing, bit for bit (CPU, this torch)
    again = m.run(tiny_args(seed=3)); again.pop("_provenance_sidecar")
    assert again["head"]["final_weights_sha256"] == a["head"]["final_weights_sha256"]                  # reproducible run to run on this machine
    other = m.run(tiny_args(seed=4)); other.pop("_provenance_sidecar")                                 # another measurement seed -> another head initialisation
    assert other["head"]["final_weights_sha256"] != a["head"]["final_weights_sha256"] and other["head"]["head_init_seed"] == 4
    for bad in (0, 4, -3):                                                                             # no other value is accepted (OI-01 option A): no override, no hidden 0
        with pytest.raises(SystemExit, match="head_init_seed is defined as the measurement seed"):
            m.run(tiny_args(seed=3, head_init_seed=bad))
    monkeypatch.setattr(sys, "argv", ["x", "--self-test", "--head-init-seed", "3"])
    with pytest.raises(SystemExit):                                                                    # the command line has no such option any more
        m.main()


def test_the_three_primary_seeds_no_longer_share_one_head_initialisation():
    from rhan_im.window_head import EvidenceHead  # noqa: F401  (the head class the init belongs to)
    batch = lambda step: (torch.zeros(4, 384), torch.zeros(4, dtype=torch.long))
    heads = {}
    for s in (17, 42, 99):
        h, _ = g1.train_evidence_head(lambda step: (torch.randn(8, 5, 384), torch.randint(0, 3, (8, 5))), 384, 3, steps=0, lr=1e-3, seed=s)
        heads[s] = torch.cat([p.detach().flatten() for p in h.parameters()])
    assert not torch.equal(heads[17], heads[42]) and not torch.equal(heads[42], heads[99])
    h0, _ = g1.train_evidence_head(lambda step: (torch.randn(8, 5, 384), torch.randint(0, 3, (8, 5))), 384, 3, steps=0, lr=1e-3)
    h0b, _ = g1.train_evidence_head(lambda step: (torch.randn(8, 5, 384), torch.randint(0, 3, (8, 5))), 384, 3, steps=0, lr=1e-3, seed=0)
    assert torch.equal(torch.cat([p.detach().flatten() for p in h0.parameters()]), torch.cat([p.detach().flatten() for p in h0b.parameters()]))   # the old default was seed 0


# ------------------------------------------------------------------------------------------------ C3 and C6
OC = None


def oc():
    global OC
    if OC is None:
        spec = importlib.util.spec_from_file_location("oc_script", ROOT / "scripts" / "rhan_im" / "gate1_operating_characteristics.py")
        OC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(OC)
    return OC


def synth(n_test=1200, n_cal=800, seed=3):
    from rhan_im.layout import UnitLayout
    o = oc()
    plan = dict(o.PLAN, n_cal=n_cal, n_test=n_test, classes=6)
    lay = UnitLayout(canvas=plan["canvas"], patch=plan["patch"], window=plan["window"], target_units=plan["op_size"])
    cand = lay.candidates(plan["stride_units"])
    gen = torch.Generator().manual_seed(seed)
    cond = o.CONDITIONS["ideal_equal_variance"]
    cal = o.synth_split(plan["n_cal"], lay, cand, cond, gen, plan["classes"])
    tst = o.synth_split(plan["n_test"], lay, cand, cond, gen, plan["classes"])
    tst["native2"] = torch.zeros(0, lay.tok ** 2, plan["classes"]); tst["tl2"] = torch.zeros(0, 2, dtype=torch.long)
    return lay, cal, tst, plan


def test_d_gist_full_ignores_target_scenes_of_other_cells():
    m = load_script()
    lay, cal, tst, plan = synth(n_test=6000)
    base = m.analyse(cal, tst, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)["gist_by_overlap"]
    sc = tst["sc"]
    op = (sc["size_units"] == plan["op_size"]) & (sc["alpha"] == plan["alphas"][0])
    outside = (~op) & (sc["label"] >= 0)                                                               # positive scenes of OTHER cells
    assert bool(outside.any())
    poisoned = {**tst, "gist": tst["gist"].clone()}
    poisoned["gist"][outside] += 25.0                                                                  # absurd evidence everywhere in other cells
    after = m.analyse(cal, poisoned, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)["gist_by_overlap"]
    assert after["fully_covered_token"] == base["fully_covered_token"] and after["partially_covered_token"] == base["partially_covered_token"]
    assert base["n_full_instances"] == int(((tst["ov_gist"] >= 0.99) & (op & (sc["label"] >= 0))[:, None]).sum())      # counted by hand
    assert base["n_partial_instances"] == int((((tst["ov_gist"] > 0) & (tst["ov_gist"] < 0.99)) & (op & (sc["label"] >= 0))[:, None]).sum())
    assert base["partially_covered_token"] is not None and base["partially_covered_token"] < base["fully_covered_token"]   # partial overlap is weaker evidence
    assert base["fully_covered_token"] == pytest.approx(1.2, abs=0.25)                                 # the synthetic sensor's d_gist
    assert "operating cell size4_alpha1.0" in base["scope"]


def test_d_gist_full_null_reference_is_every_null_scene_whatever_its_placeholder_cell():
    """OI-02 option A. A null scene has no pasted target, so the size / opacity that scene_geometry draws for it are placeholders: they must not
    restrict the null reference. Null scenes whose placeholder lies OUTSIDE the operating cell still feed the null mean."""
    m = load_script()
    lay, cal, tst, plan = synth(n_test=6000)
    base = m.analyse(cal, tst, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)["gist_by_overlap"]
    sc = tst["sc"]
    op = (sc["size_units"] == plan["op_size"]) & (sc["alpha"] == plan["alphas"][0])
    null_other = (sc["label"] < 0) & (~op)                                                             # null scenes with a placeholder outside the operating cell
    null_in = (sc["label"] < 0) & op
    assert int(null_other.sum()) > 100 and int(null_in.sum()) > 30                                      # both kinds exist in the synthetic set
    poisoned = {**tst, "gist": tst["gist"].clone()}
    poisoned["gist"][null_other] += 25.0                                                                # absurd evidence in those null scenes only
    after = m.analyse(cal, poisoned, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)["gist_by_overlap"]
    assert after["fully_covered_token"] < base["fully_covered_token"] - 0.5                             # they are in the null reference: its mean rose, so d fell
    assert after["n_full_instances"] == base["n_full_instances"]                                        # the positive instances are untouched
    assert "all null scenes" in base["scope"]


def test_d_gist_full_is_unmeasured_below_the_minimum_number_of_instances():
    m = load_script()
    lay, cal, tst, plan = synth(n_test=120, n_cal=300)
    g = m.analyse(cal, tst, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)["gist_by_overlap"]
    assert g["n_full_instances"] < g["min_instances"] == 30 and g["fully_covered_token"] is None
    assert g["n_partial_instances"] >= 30 and g["partially_covered_token"] is not None                  # each quantity is guarded by its OWN instance count
    lay, cal, tst, plan = synth(n_test=40, n_cal=300)
    g = m.analyse(cal, tst, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)["gist_by_overlap"]
    assert g["n_partial_instances"] < 30 and g["partially_covered_token"] is None                      # the same minimum on the report-only partial-overlap quantity


def test_every_gist_call_uses_the_same_minimum_as_the_native_look(monkeypatch):
    m = load_script()
    lay, cal, tst, plan = synth(n_test=600, n_cal=300)
    seen = []
    real_mt, real_bv = g1.measure_tokens, g1.block_vif
    monkeypatch.setattr(g1, "measure_tokens", lambda *a, **k: (seen.append(("measure_tokens", k.get("min_pos"))), real_mt(*a, **k))[1])
    monkeypatch.setattr(g1, "block_vif", lambda *a, **k: (seen.append(("block_vif", k.get("min_pos"))), real_bv(*a, **k))[1])
    m.analyse(cal, tst, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)
    explicit = [v for _, v in seen if v is not None]
    assert explicit and set(explicit) == {30}                                                          # was 20 at two sites
    assert "min_pos=20" not in (ROOT / "scripts" / "rhan_im" / "gate1_measure_dprime.py").read_text()
    import inspect
    assert inspect.signature(real_mt).parameters["min_pos"].default == 30                              # the native default the gist now matches
    assert inspect.signature(real_bv).parameters["min_pos"].default == 30


# ------------------------------------------------------------------------------------------------ C5, C7
def test_run_records_the_runtime_environment():
    m = load_script()
    r = m.run(tiny_args(seed=1)); r.pop("_provenance_sidecar")
    env = r["environment"]
    assert env["torch"] == torch.__version__ and env["python"] and env["platform"] and env["device"] == "cpu" and "deterministic_algorithms" in env and "num_threads" in env


def test_provenance_sidecar_lists_every_scene_and_matches_the_pools(tree, tmp_path):
    m = load_script()
    classes = sorted(p.name for p in (tree / "train").iterdir())
    t, b = classes[1::2], classes[0::2]
    sp, h = write_split(tmp_path, t, b)
    pp, ah, cal, test = write_pools(tmp_path, tree, h)
    r = m.run(real_args(tree, class_split=str(sp), pools=str(pp)))
    side = r.pop("_provenance_sidecar")
    assert r["pools"]["pool_assignment_sha256"] == ah
    for role, pool, n in (("calibration", cal, 60), ("test", test, 80)):
        d = side[role]
        assert d["n_scenes"] == n and len(d["background"]) == n and len(d["target"]) == n and len(d["label"]) == n and len(d["window1_topleft_units"]) == n
        for bg in d["background"]:
            assert bg[0] in b and bg[1] in pool[bg[0]]                                                 # background class never a target; file from THIS pool
        for lab, tg in zip(d["label"], d["target"]):
            if lab < 0:
                assert tg is None
            else:
                assert tg[0] == t[lab] and tg[1] in pool[tg[0]]
    assert len(side["test"]["window2_topleft_units"]) == 20 and side["calibration"]["window2_topleft_units"] == []
    other = "test" if "x" else None
    cal_files = {(c, f) for c, fs in cal.items() for f in fs}
    assert not ({tuple(x) for x in side["test"]["background"]} & cal_files)                            # the pools stay disjoint inside the scenes
    link = {"seed": 0, "head_init_seed": 0, "checkpoint_sha256": r["trunk"].get("sha256"), "class_lists_sha256": r["class_split"]["class_lists_sha256"],
            "pool_assignment_sha256": ah, "n_cal": 60, "n_test": 80, "n_pairs": 20}
    assert side["link"] == link and r["provenance"]["link_sha256"] == chash(link)                       # written out by hand, independent of g1.provenance_link
    assert r["provenance"]["sidecar_sha256"] == hashlib.sha256(g1.canonical_json_bytes(side)).hexdigest()
    assert r["provenance"]["sidecar_bytes"] == len(g1.canonical_json_bytes(side)) > 0 and r["provenance"]["build_and_hash_seconds"] >= 0   # OI-04: overhead is observable


def test_provenance_does_not_change_which_images_are_drawn(tree):
    """The recording must not move the random streams: a bank that records draws picks exactly the files the old expressions picked."""
    m = load_script()
    classes = sorted(p.name for p in (tree / "val").iterdir())
    new = m.FolderBank(tree, "val", classes[:4], classes[4:], seed=5, part=(1, 2))
    old = m.FolderBank(tree, "val", classes[:4], classes[4:], seed=5, part=(1, 2))
    gen = torch.Generator().manual_seed(0)
    for i in range(30):
        if i % 3:
            new.background(28, gen)
            want = old.rng.choice(old.files[old.rng.choice(old.bg_classes)])                             # the pre-change expression
            assert new.last == (want.parent.name, want.name)
        else:
            new.target(i % 4, 28, gen)
            want = old.rng.choice(old.files[old.target_classes[i % 4]])
            assert new.last == (want.parent.name, want.name)


def test_main_writes_the_sidecar_next_to_the_run_json_and_the_hash_matches_its_bytes(tmp_path, monkeypatch):
    m = load_script()
    out = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["x", "--self-test", "--seed", "4", "--out", str(out)])
    monkeypatch.setattr(m.torch, "set_num_threads", lambda n: None)
    m.main()
    run = json.loads(out.read_text())
    side_path = tmp_path / run["provenance"]["file"]
    assert side_path.name == "run.provenance.json" and "_provenance_sidecar" not in run
    assert hashlib.sha256(side_path.read_bytes()).hexdigest() == run["provenance"]["sidecar_sha256"]
    side = json.loads(side_path.read_bytes())
    assert hashlib.sha256(g1.canonical_json_bytes(side["link"])).hexdigest() == run["provenance"]["link_sha256"] == side["link_sha256"]


# ------------------------------------------------------------------------------------------------ C8
def test_pools_file_must_match_the_images_the_run_would_draw_from(tree, tmp_path):
    m = load_script()
    classes = sorted(p.name for p in (tree / "train").iterdir())
    sp, h = write_split(tmp_path, classes[:4], classes[4:])
    pp, ah, cal, test = write_pools(tmp_path, tree, h)
    assert m.run(real_args(tree, class_split=str(sp), pools=str(pp)))["pools"]["pool_assignment_sha256"] == ah
    with pytest.raises(SystemExit, match="--class-split is required"):
        m.run(real_args(tree, pools=str(pp)))
    other, h2 = write_split(tmp_path, classes[1:5], classes[5:] + classes[:1], name="other.json")
    with pytest.raises(SystemExit, match="not a FINAL pools file"):
        m.run(real_args(tree, class_split=str(other), pools=str(pp)))
    (tree / "val" / classes[0] / "img005.png").unlink()                                                 # the tree changed after `pools`
    with pytest.raises(SystemExit):
        m.run(real_args(tree, class_split=str(sp), pools=str(pp)))
    pp2, ah2, _, _ = write_pools(tmp_path, tree, h, name="pools2.json")
    d = json.loads(pp2.read_text()); d["pool_assignment_sha256"] = "0" * 64; pp2.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="recomputed"):
        m.run(real_args(tree, class_split=str(sp), pools=str(pp2)))


def test_a_pools_file_missing_a_class_is_a_refusal_not_a_keyerror(tree, tmp_path):
    """A consistently re-hashed pools file that lacks one class: check_pools must refuse (SystemExit), whatever read_pools accepts."""
    m = load_script()
    classes = sorted(p.name for p in (tree / "train").iterdir())
    sp, h = write_split(tmp_path, classes[:4], classes[4:])
    pp, ah, cal, test = write_pools(tmp_path, tree, h)
    d = json.loads(pp.read_text())
    gone = classes[0]
    del d["calibration_pool_part_0_of_2"][gone]
    cal2 = d["calibration_pool_part_0_of_2"]
    d["pool_assignment_sha256"] = chash({"rule_id": "r", "class_split": h, "val_repo_listing_sha256": d["listing_binding"]["val"]["repo_listing_sha256"],
                                         "calibration": cal2, "test": d["test_pool_part_1_of_2"]})        # hash is consistent with the edited content
    pp.write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        m.run(real_args(tree, class_split=str(sp), pools=str(pp)))
