"""One episode of the loop: observe, record, believe, plan, commit (docs/02 §5)."""
from __future__ import annotations

from typing import Any

import zlib

import numpy as np

from rhan_bi.belief import BeliefState
from rhan_bi.config import Config
from rhan_bi.evidence import EvidenceBuilder
from rhan_bi.ledger import EvidenceLedger, LedgerIntegrityError
from rhan_bi.likelihood import EvidenceModel
from rhan_bi.perception.backbone import Backbone
from rhan_bi.planner import ForwardModel, Planner
from rhan_bi.simulator.lab import InspectionLab
from rhan_bi.simulator.scene import Scene
from rhan_bi.stopping import decide


def run_episode(
    scene: Scene,
    policy: str,
    config: Config,
    backbone: Backbone,
    model: EvidenceModel,
    forward: ForwardModel,
    budget: int | None = None,
    use_stopping: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    budget = int(budget if budget is not None else config.run.budget)
    lab = InspectionLab(scene, config, budget=budget)
    builder = EvidenceBuilder(config, backbone, model)
    planner = Planner(config, model, forward)
    rng = np.random.default_rng([int(scene.seed), int(seed), zlib.crc32(policy.encode()) & 0x7FFFFFFF])

    ledger = EvidenceLedger()
    obs = lab.reset()
    ledger.add(builder.build(obs, ledger, lab.episode_id), image=obs.image)
    belief = BeliefState.from_ledger(ledger, config.belief)

    trace: list[dict[str, Any]] = [{
        "step": 0,
        "action": obs.action.label(),
        "belief": belief.to_dict(),
        "predicted": None,
    }]
    stopped = None

    while lab.step < budget:
        actions = lab.legal_actions()
        action, info = planner.choose(
            belief, ledger, actions, policy=policy, rng=rng,
            ground_truth=scene.cause if policy == "oracle" else None,
        )
        if action is None:
            break
        try:
            obs = lab.observe(action)
        except ValueError as exc:
            raise LedgerIntegrityError(f"planner chose an illegal action: {exc}") from exc
        rec = builder.build(obs, ledger, lab.episode_id)
        ledger.add(rec, image=obs.image)
        previous = belief
        belief = BeliefState.from_ledger(ledger, config.belief, prev=previous)
        trace.append({
            "step": int(obs.step),
            "action": action.label(),
            "belief": belief.to_dict(),
            "predicted": info.get("scores", {}).get(action.label()),
        })

        if use_stopping:
            best_value = None
            if info.get("scores"):
                vals = [v["score"] for v in info["scores"].values() if isinstance(v, dict) and "score" in v]
                best_value = max(vals) if vals else None
            stopped = decide(belief, budget - lab.step, config.belief, best_value=best_value,
                             next_cost=config.planner.cost_light / config.planner.cost_view)
            if stopped.action != "continue":
                break

    if stopped is not None and stopped.action == "abstain":
        decision = "abstain"
        committed = None
        abstained = True
    else:
        committed = belief.map_cause()
        decision = committed
        abstained = False

    gt = lab.ground_truth()
    return {
        "scene_id": scene.scene_id,
        "seed": int(scene.seed),
        "cause": scene.cause,
        "scenario": scene.scenario,
        "policy": policy,
        "decision": decision,
        "committed": committed,
        "abstained": abstained,
        "correct": bool(committed == scene.cause),
        "n_observations": int(lab.step),
        "cost": float(sum(a.cost for a in lab.actions)),
        "final_belief": belief.to_dict(),
        "map_trajectory": [t["belief"]["map"] for t in trace],
        "confidence_trajectory": [t["belief"]["confidence"] for t in trace],
        "actions": [a.to_dict() for a in lab.actions],
        "n_eff": float(ledger.n_eff()),
        "n_records": len(ledger),
        "conflicts": belief.conflicts,
        "revisions": belief.revisions,
        "records": [r.to_json() for r in ledger.records],
        "ground_truth": gt,
        "trace": trace,
    }
