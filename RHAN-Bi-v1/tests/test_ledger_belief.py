import dataclasses

import numpy as np
import pytest

from rhan_bi.belief import BeliefState
from rhan_bi.config import default_config
from rhan_bi.evidence import EvidenceBuilder
from rhan_bi.ledger import EvidenceLedger, LedgerIntegrityError
from rhan_bi.perception.backbone import HandcraftedBackbone
from rhan_bi.simulator.lab import InspectionLab
from rhan_bi.simulator.scene import make_scene
from rhan_bi.types import CAUSES, Action, EvidenceRecord


def _episode(cfg, index=1):
    backbone = HandcraftedBackbone(cfg.features)
    scene = make_scene("dev", index, cfg.sim)
    lab = InspectionLab(scene, cfg)
    ledger = EvidenceLedger()
    builder = EvidenceBuilder(cfg, backbone, model=None)
    obs = lab.reset()
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    obs = lab.observe(Action(kind="light", light=2, cost=1.0))
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    return scene, ledger


def test_hashes_verify_and_no_ground_truth_leak():
    cfg = default_config()
    _, ledger = _episode(cfg)
    assert ledger.verify()["ok"]
    assert all("cause" not in rec.to_json() for rec in ledger.records)


def test_tampering_is_detected():
    cfg = default_config()
    _, ledger = _episode(cfg)
    rec = ledger.records[0]
    tampered = dataclasses.replace(rec, step=99)
    assert not tampered.verify()
    with pytest.raises(LedgerIntegrityError):
        ledger.add(tampered)


def test_duplicate_add_rejected():
    cfg = default_config()
    _, ledger = _episode(cfg)
    with pytest.raises(LedgerIntegrityError):
        ledger.add(ledger.records[0])


def _record(llr_defect, llr_reflection, weight, pair_llr=None):
    llr = {c: 0.0 for c in CAUSES}
    llr["defect"] = llr_defect
    llr["reflection"] = llr_reflection
    pair = None
    if pair_llr is not None:
        pair = {"with_record_id": "x", "persistence": 0.1, "bright_shift": 1.0,
                "llr": {**{c: 0.0 for c in CAUSES}, **pair_llr}}
    return EvidenceRecord(
        ledger_version="rhan-bi-ledger-v1", episode_id="e", scene_id="s", step=1,
        action={"kind": "light", "view": 0, "light": 2, "roi": None, "cost": 1.0},
        image_sha256="0" * 64,
        features={k: 0.0 for k in ("ridge_dark", "ridge_bright", "soft_dark", "sharp_ratio",
                                   "glare", "occlusion", "contrast", "snr")},
        quality={"score": 1.0, "flags": [], "components": {}},
        llr=llr, pair=pair,
        dependence={"rho_max": 0.0, "relations": [], "dependency_weight": weight, "weight": weight},
        provenance={},
    )


def test_pair_evidence_can_overturn_the_leader():
    ledger = EvidenceLedger()
    first = _record(3.0, 0.0, 1.0)
    ledger.add(first)
    assert BeliefState.from_ledger(ledger).map_cause() == "defect"
    ledger.add(_record(3.0, 0.0, 1.0, pair_llr={"defect": -4.0, "reflection": 4.0}))
    assert BeliefState.from_ledger(ledger).map_cause() == "reflection"


def test_weight_reduces_influence():
    a, b = EvidenceLedger(), EvidenceLedger()
    a.add(_record(2.0, 0.0, 1.0))
    b.add(_record(2.0, 0.0, 0.25))
    pa = BeliefState.from_ledger(a).posterior["defect"]
    pb = BeliefState.from_ledger(b).posterior["defect"]
    assert pa > pb


def test_posterior_normalised_and_revision_logged():
    cfg = default_config()
    backbone = HandcraftedBackbone(cfg.features)
    scene = make_scene("dev", 1, cfg.sim)
    lab = InspectionLab(scene, cfg)
    ledger = EvidenceLedger()
    builder = EvidenceBuilder(cfg, backbone, model=None)

    obs = lab.reset()
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    state = BeliefState.from_ledger(ledger)
    assert abs(sum(state.posterior.values()) - 1.0) < 1e-9

    obs = lab.observe(Action(kind="light", light=2, cost=1.0))
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    state2 = BeliefState.from_ledger(ledger, prev=state)
    assert abs(sum(state2.posterior.values()) - 1.0) < 1e-9
    for revision in state2.revisions:
        assert set(revision) >= {"from", "to", "caused_by", "step"}
        assert revision["caused_by"] in {r.record_id for r in ledger.records}
