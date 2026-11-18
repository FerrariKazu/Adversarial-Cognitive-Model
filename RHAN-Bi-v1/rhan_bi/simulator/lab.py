"""The laboratory interface: observations, legality, budget, ground truth.

The lab enforces the action space so a policy cannot request an observation the environment cannot
produce (docs/08 §5). The agent-facing ``Observation`` never contains the cause.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from rhan_bi.config import Config
from rhan_bi.simulator.scene import Scene, mutate, render
from rhan_bi.types import Action, Observation


class IllegalAction(ValueError):
    pass


class LabProtocol(Protocol):
    def reset(self) -> Observation: ...
    def observe(self, action: Action) -> Observation: ...
    def legal_actions(self) -> list[Action]: ...
    def ground_truth(self) -> dict[str, Any]: ...
    def advance_scene(self, mutation: str) -> Observation: ...


class InspectionLab:
    """One scene, one episode, up to ``budget`` observations."""

    def __init__(self, scene: Scene, config: Config, budget: int | None = None):
        self.cfg = config
        self.scene = scene
        self.budget = int(budget if budget is not None else config.run.budget)
        self.episode_id = f"{scene.scene_id}-e0"
        self._reset_done = False
        self._step = 0
        self._view = 0
        self._light = 1
        self._zoom_box: tuple[float, float, float, float] | None = None
        self._zoom_count = 0
        self._actions: list[Action] = []
        self._observations: list[Observation] = []

    # ------------------------------------------------------------------ state
    @property
    def step(self) -> int:
        return self._step

    @property
    def view(self) -> int:
        return self._view

    @property
    def light(self) -> int:
        return self._light

    @property
    def actions(self) -> list[Action]:
        return list(self._actions)

    # ------------------------------------------------------------------ API
    def reset(self) -> Observation:
        if self._reset_done:
            raise IllegalAction("reset() already called; create a new lab for a new episode")
        self._reset_done = True
        action = Action(kind="base", view=0, light=1, roi=None, cost=0.0)
        self._actions.append(action)
        obs = self._make_observation(action)
        self._observations.append(obs)
        return obs

    def observe(self, action: Action) -> Observation:
        if not self._reset_done:
            raise IllegalAction("call reset() first")
        if self._step >= self.budget:
            raise IllegalAction(f"budget of {self.budget} observations exhausted")
        if not isinstance(action, Action):
            raise IllegalAction("action must be an Action")

        if action.kind == "light":
            if action.light == self._light:
                raise IllegalAction("light action must change the illumination")
            self._light = int(action.light)
        elif action.kind == "view":
            if action.view == self._view:
                raise IllegalAction("view action must change the camera")
            self._view = int(action.view)
        elif action.kind == "zoom":
            if self._zoom_count >= 2:
                raise IllegalAction("at most two zooms per episode")
            self._zoom_box = action.roi or (0.25, 0.25, 0.75, 0.75)
            self._zoom_count += 1
        else:
            raise IllegalAction(f"unknown action kind {action.kind!r}")

        cost = action.cost or self._cost_of(action)
        action = replace(action, cost=cost, view=self._view, light=self._light)
        self._step += 1
        self._actions.append(action)
        obs = self._make_observation(action)
        self._observations.append(obs)
        return obs

    def legal_actions(self) -> list[Action]:
        out: list[Action] = []
        for light in (0, 1, 2):
            if light != self._light:
                out.append(Action(kind="light", view=self._view, light=light,
                                  roi=self._zoom_box, cost=self.cfg.planner.cost_light))
        for view in (-1, 0, 1):
            if view != self._view:
                out.append(Action(kind="view", view=view, light=self._light,
                                  roi=self._zoom_box, cost=self.cfg.planner.cost_view))
        if self._zoom_count < 2:
            out.append(Action(kind="zoom", view=self._view, light=self._light,
                              roi=(0.25, 0.25, 0.75, 0.75), cost=self.cfg.planner.cost_zoom))
        return out

    def ground_truth(self) -> dict[str, Any]:
        gt = self.scene.ground_truth()
        gt["episode_id"] = self.episode_id
        gt["actions"] = [a.to_dict() for a in self._actions]
        gt["observations"] = len(self._observations)
        return gt

    def advance_scene(self, mutation: str) -> Observation:
        self.scene = mutate(self.scene, mutation, self.cfg.sim)
        return self._make_observation(Action(kind="base", view=self._view, light=self._light,
                                             roi=self._zoom_box, cost=0.0))

    # ------------------------------------------------------------------ helpers
    def _cost_of(self, action: Action) -> float:
        return {"light": self.cfg.planner.cost_light,
                "view": self.cfg.planner.cost_view,
                "zoom": self.cfg.planner.cost_zoom}.get(action.kind, 0.0)

    def _make_observation(self, action: Action) -> Observation:
        image = render(
            self.scene,
            view=self._view,
            light=self._light,
            cfg=self.cfg.sim,
            step=self._step,
            action_index=len(self._actions) - 1,
            zoom_box=self._zoom_box,
        )
        return Observation(
            image=image,
            action=action,
            step=self._step,
            scene_id=self.scene.scene_id,
            view=self._view,
            light=self._light,
            roi=self._zoom_box,
        )
