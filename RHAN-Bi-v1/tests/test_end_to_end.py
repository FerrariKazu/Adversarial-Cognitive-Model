import numpy as np

from rhan_bi.config import default_config
from rhan_bi.evaluation.runner import run_episode
from rhan_bi.perception.backbone import HandcraftedBackbone
from rhan_bi.training.data import collect_calibration
from rhan_bi.training.stages import fit_all
from rhan_bi.simulator.scene import make_scene


def _setup():
    cfg = default_config()
    cfg = cfg.__class__.from_dict({**cfg.to_dict(), "run": {**cfg.to_dict()["run"],
                                                             "budget": 2}})
    backbone = HandcraftedBackbone(cfg.features)
    cal = collect_calibration(8, cfg, backbone)
    model, forward = fit_all(cal, cfg, backbone)
    return cfg, backbone, model, forward


def test_episode_structure_and_reproducibility():
    cfg, backbone, model, forward = _setup()
    scene = make_scene("test", 1, cfg.sim)
    ep = run_episode(scene, "falsification", cfg, backbone, model, forward, seed=5)
    assert ep["n_observations"] == cfg.run.budget
    # serialised posteriors are rounded to 6 decimals, so the serialised sum is 1 to 1e-6
    assert abs(sum(ep["final_belief"]["posterior"].values()) - 1.0) < 1e-4
    assert ep["decision"] in ("defect", "reflection", "shadow", "none")
    assert ep["ground_truth"]["cause"] == scene.cause
    for rec in ep["records"]:
        assert len(rec["record_id"]) == 64
    ep2 = run_episode(scene, "falsification", cfg, backbone, model, forward, seed=5)
    assert ep["decision"] == ep2["decision"]
    assert ep["final_belief"]["posterior"] == ep2["final_belief"]["posterior"]


def test_oracle_policy_runs():
    cfg, backbone, model, forward = _setup()
    scene = make_scene("test", 0, cfg.sim)
    ep = run_episode(scene, "oracle", cfg, backbone, model, forward, seed=1)
    assert ep["n_observations"] == cfg.run.budget
    assert "oracle" not in ep["ground_truth"]
