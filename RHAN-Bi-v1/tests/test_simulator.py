import numpy as np
import pytest

from rhan_bi.config import default_config
from rhan_bi.simulator import IllegalAction, InspectionLab, make_scene, mutate, render
from rhan_bi.types import Action


def test_render_is_deterministic():
    cfg = default_config()
    scene = make_scene("dev", 1, cfg.sim)
    a = render(scene, view=0, light=1, cfg=cfg.sim, step=1, action_index=1)
    b = render(scene, view=0, light=1, cfg=cfg.sim, step=1, action_index=1)
    assert np.array_equal(a, b)


def test_defect_is_fixed_under_light_change():
    cfg = default_config()
    scene = make_scene("dev", 0, cfg.sim)  # index 0 is a defect scene
    assert scene.cause == "defect"
    c1 = scene.anomaly_centre(0, 0, cfg.sim)
    c2 = scene.anomaly_centre(0, 2, cfg.sim)
    assert c1 == c2


def test_reflection_moves_with_light():
    cfg = default_config()
    scene = make_scene("dev", 1, cfg.sim)  # index 1 is a reflection scene
    assert scene.cause == "reflection"
    c1 = scene.anomaly_centre(0, 0, cfg.sim)
    c2 = scene.anomaly_centre(0, 2, cfg.sim)
    assert abs(c2[0] - c1[0]) > 10


def test_observation_has_no_ground_truth():
    cfg = default_config()
    lab = InspectionLab(make_scene("dev", 2, cfg.sim), cfg)
    obs = lab.reset()
    assert not hasattr(obs, "cause")
    assert "cause" not in obs.meta()


def test_action_legality_and_budget():
    cfg = default_config()
    lab = InspectionLab(make_scene("dev", 0, cfg.sim), cfg, budget=1)
    lab.reset()
    with pytest.raises(IllegalAction):
        lab.observe(Action(kind="light", light=1))  # same light
    lab.observe(Action(kind="light", light=2))
    with pytest.raises(IllegalAction):
        lab.observe(Action(kind="zoom"))


def test_s4_mutation_changes_state():
    cfg = default_config()
    scene = make_scene("dev", 3, cfg.sim)  # none
    assert scene.cause == "none"
    lab = InspectionLab(scene, cfg)
    lab.reset()
    lab.advance_scene("introduce_defect")
    assert lab.ground_truth()["cause"] == "defect"
    with pytest.raises(ValueError):
        mutate(make_scene("dev", 0, cfg.sim), "introduce_defect")
