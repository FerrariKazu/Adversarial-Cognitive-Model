import numpy as np

from rhan_bi.belief import BeliefState
from rhan_bi.config import default_config
from rhan_bi.dependence import effective_n, rho, weight_for
from rhan_bi.likelihood import EvidenceModel
from rhan_bi.ledger import EvidenceLedger
from rhan_bi.perception.features import FEATURE_NAMES
from rhan_bi.planner import ForwardModel, Planner
from rhan_bi.types import Action


def test_rho_rules():
    cfg = default_config().evidence
    a = {"kind": "light", "view": 0, "light": 2, "roi": None}
    b = {"kind": "light", "view": 0, "light": 1, "roi": None}
    c = {"kind": "view", "view": 1, "light": 1, "roi": None}
    assert rho(a, a, cfg)[0] == 1.0
    assert rho(a, b, cfg)[0] == cfg.dependence_same_view
    assert rho(b, c, cfg)[0] == cfg.dependence_same_light


def test_weight_and_effective_n():
    cfg = default_config().evidence
    a = {"kind": "light", "view": 0, "light": 2, "roi": None}
    w, _ = weight_for(a, [], cfg, quality_score=1.0)
    assert w == 1.0
    class R:
        def __init__(self, action):
            self.action = action
            self.record_id = "r"
    w2, rels = weight_for(a, [R(a)], cfg, quality_score=1.0)
    assert w2 <= 0.5 + 1e-9
    assert effective_n([0.5, 0.5]) < 2.0


def _model_and_forward():
    rng = np.random.default_rng(0)
    names = FEATURE_NAMES
    null = {k: float(rng.normal(0, 1)) for k in names}
    rows = {c: [{k: float(null[k] + (2.0 if c == "defect" else 0.0)) for k in names}
                for _ in range(20)] for c in ("defect", "reflection", "shadow", "none")}
    pair = {"defect": [(0.9, 1.0)] * 20, "reflection": [(0.05, 1.0)] * 20,
            "shadow": [(0.2, 1.0)] * 20}
    model = EvidenceModel.fit(rows, pair)
    fmu = {}
    fsd = {}
    for c in ("defect", "reflection", "shadow", "none"):
        for kind in ("base", "light", "view", "zoom"):
            fmu[(c, kind)] = np.zeros(len(names))
            fsd[(c, kind)] = np.ones(len(names))
    fwd = ForwardModel(mu=fmu, sd=fsd,
                       pair_mu={"defect": np.array([0.9, 1.0]),
                                "reflection": np.array([0.05, 1.0]),
                                "shadow": np.array([0.2, 1.0])},
                       pair_sd={c: np.array([0.05, 0.1]) for c in ("defect", "reflection", "shadow")},
                       fitted=True)
    return model, fwd


def test_planner_returns_legal_deterministic_choice():
    cfg = default_config()
    model, fwd = _model_and_forward()
    planner = Planner(cfg, model, fwd)
    ledger = EvidenceLedger()
    belief = BeliefState(posterior={c: 0.25 for c in ("defect", "reflection", "shadow", "none")},
                         log_weights={c: 0.0 for c in ("defect", "reflection", "shadow", "none")})
    actions = [Action(kind="light", light=0, cost=1.0), Action(kind="light", light=2, cost=1.0),
               Action(kind="zoom", roi=(0.25, 0.25, 0.75, 0.75), cost=1.0)]
    a1, info1 = planner.choose(belief, ledger, actions, policy="falsification",
                               rng=np.random.default_rng(3))
    a2, _ = planner.choose(belief, ledger, actions, policy="falsification",
                           rng=np.random.default_rng(3))
    assert a1 in actions and a2 == a1
    assert "displacement" in info1["scores"][a1.label()]
    for policy in ("eig", "confidence", "random", "fixed_light_first"):
        chosen, _ = planner.choose(belief, ledger, actions, policy=policy, rng=np.random.default_rng(1))
        assert chosen in actions
