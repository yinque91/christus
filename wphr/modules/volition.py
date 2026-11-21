"""M7 Volition: goal competition under value, information gain and cost.

Implements the MASTER SPEC 9.2 scoring::

    goal_score = expected_value + information_gain + long_term_alignment
                 - resource_cost - risk_cost - opportunity_cost

V0.1 keeps a small explicit goal set. The selected goal is published to shared
state with its score so the choice is auditable, never implicit.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from wphr.core.module import BaseModule


class VolitionModule(BaseModule):
    name: ClassVar[str] = "volition"
    description: ClassVar[str] = "goal competition and selection under constraints"
    group: ClassVar[str] = "slow"

    def reset(self) -> None:
        self._state: dict[str, Any] = {
            "goal": "explore",
            "score": 0.0,
            "scores": {},
            "switches": 0,
            "history": [],
        }

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        affect = self.ctx.get_shared("affect.values", {}) or {}
        energy = float(self.ctx.get_shared("world.energy", 1.0))
        obs = self.ctx.get_shared("world.obs", []) or []
        if energy == 1.0 and len(obs) > 2:
            energy = float(obs[2])
        curiosity = float(affect.get("curiosity", 0.5))
        fatigue = float(affect.get("fatigue", 0.0))
        risk_sensitivity = float(affect.get("risk_sensitivity", 0.5))
        stress = float(affect.get("stress_load", 0.0))

        candidates = {
            "explore": {
                "value": 0.60,
                "info": 0.80 * curiosity,
                "long_term": 0.30,
                "cost": 0.20,
                "risk": 0.10 * risk_sensitivity,
                "opportunity": 0.05,
            },
            "recharge": {
                "value": 1.20 * (1.0 - energy) if energy < 0.5 else 0.15,
                "info": 0.05,
                "long_term": 0.40,
                "cost": 0.10,
                "risk": 0.05 * risk_sensitivity,
                "opportunity": 0.05,
            },
            "rest": {
                "value": 0.90 * fatigue,
                "info": 0.00,
                "long_term": 0.20,
                "cost": 0.05,
                "risk": 0.02,
                "opportunity": 0.10,
            },
            "consolidate": {
                "value": 0.25 + 0.30 * stress,
                "info": 0.20,
                "long_term": 0.50,
                "cost": 0.15,
                "risk": 0.02,
                "opportunity": 0.05,
            },
        }
        scores: dict[str, float] = {}
        goal_bias = self.ctx.get_shared("affect.goal_bias", {}) or {}
        for name, parts in candidates.items():
            score = (
                parts["value"]
                + parts["info"]
                + parts["long_term"]
                - parts["cost"]
                - parts["risk"]
                - parts["opportunity"]
            )
            score *= float(goal_bias.get(name, 1.0))
            scores[name] = round(float(score), 6)
        best = max(sorted(scores), key=lambda name: (scores[name], name == self._state["goal"]))
        if best != self._state["goal"]:
            self._state["switches"] += 1
        self._state["goal"] = best
        self._state["score"] = scores[best]
        self._state["scores"] = scores
        history = self._state["history"]
        history.append({"step": int(step), "goal": best, "score": scores[best]})
        if len(history) > 64:
            del history[:-64]
        self.ctx.set_shared("volition.goal", best, source=self.name)
        self.ctx.set_shared("volition.score", scores[best], source=self.name)
        self.ctx.set_shared("volition.scores", dict(scores), source=self.name)
        _ = np

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "goal": self._state["goal"],
                "score": round(float(self._state["score"]), 4),
                "switches": int(self._state["switches"]),
            }
        )
        return base
