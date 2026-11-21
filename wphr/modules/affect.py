"""M3 Affect/Motivation: functional internal regulation variables.

Every channel has an explicit range, a decay term, an event-driven impulse, and
a per-step maximum change. The values modulate exploration (through M2) and are
logged with every event snapshot. Names are engineering labels; they do not
assert subjective feelings.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from wphr.core.event import Event, EventType
from wphr.core.module import BaseModule


class AffectModule(BaseModule):
    name: ClassVar[str] = "affect"
    description: ClassVar[str] = "functional affect: decay, impulses, bounded regulation"
    group: ClassVar[str] = "affect"

    def reset(self) -> None:
        cfg = self.ctx.config.section("affect")
        self.ablation = str(cfg.get("ablation", "full"))
        self.spec: dict[str, dict[str, float]] = {
            name: {key: float(value) for key, value in channel.items()}
            for name, channel in cfg["channels"].items()
        }
        self.bias: dict[str, float] = {
            key: float(value) for key, value in cfg["exploration_bias"].items()
        }
        self._state: dict[str, Any] = {
            "values": {name: spec["init"] for name, spec in self.spec.items()},
            "impulses": {name: 0.0 for name in self.spec},
            "impulse_events": 0,
            "updates": 0,
        }

    # ------------------------------------------------------------------ event
    def on_event(self, event: Event) -> None:
        super().on_event(event)
        impulses = self._state["impulses"]
        self._state["impulse_events"] += 1
        if event.event_type == EventType.OUTCOME.value:
            outcome = event.actual_outcome or {}
            reward = float(outcome.get("reward", 0.0))
            err = float(event.prediction_error or 0.0)
            impulses["frustration"] += 0.12 * min(err, 2.0)
            if reward > 0:
                impulses["relief"] += 0.18
                impulses["frustration"] -= 0.10
                impulses["curiosity"] += 0.05
            else:
                impulses["frustration"] += 0.02
            impulses["fatigue"] += 0.004
        elif event.event_type == EventType.OBSERVATION.value:
            energy = float((event.observation or {}).get("energy", 1.0))
            if energy < 0.35:
                impulses["stress_load"] += 0.05 * (0.35 - energy) / 0.35
                impulses["risk_sensitivity"] += 0.03
        elif event.event_type == EventType.PREDICTION.value:
            # A real action-conditioned prediction error, not a reward value.
            # It can raise frustration/curiosity through the existing channels;
            # it never directly sets action scores.
            err = float(event.prediction_error or 0.0)
            impulses["frustration"] += 0.04 * min(err, 2.0)
            impulses["curiosity"] += 0.02 * min(err, 2.0)
        elif event.event_type == EventType.SOCIAL.value:
            impulses["social_motivation"] += 0.1

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        values = self._state["values"]
        impulses = self._state["impulses"]
        if self.ablation == "none":
            # Ablation control: no modulation at all. Channels stay at their
            # initial values and the exploration rate stays at the base rate,
            # so a full-vs-none comparison isolates the affect mechanism.
            for name in impulses:
                impulses[name] = 0.0
            self._state["impulse_events"] = 0
            self._state["updates"] += 1
            self.ctx.set_shared("affect.values", dict(values), source=self.name)
            self.ctx.set_shared("affect.exploration_bias", self.bias.get("base", 0.25), source=self.name)
            return
        # Impulses are averaged per event since the previous update. Summing
        # them instead lets the event *rate* drive the channel to an extreme,
        # which made risk_sensitivity and stress_load stick at 1.0 during the
        # V0.1 smoke test.
        events_since = max(1, int(self._state["impulse_events"]))
        averaged = {name: impulses[name] / events_since for name in impulses}
        for name, spec in self.spec.items():
            previous = values[name]
            # decay + event impulse, then a homeostatic pull toward the channel
            # baseline. Restitution is the explicit recovery mechanism required
            # by MASTER SPEC 7.1.
            decayed = spec["decay"] * previous + averaged[name]
            baseline = spec.get("baseline", spec["init"])
            restitution = spec.get("restitution", 0.0)
            homeostatic = decayed + restitution * (baseline - decayed)
            bounded = float(np.clip(homeostatic, spec["min"], spec["max"]))
            delta = float(np.clip(bounded - previous, -spec["max_change"], spec["max_change"]))
            values[name] = float(np.clip(previous + delta, spec["min"], spec["max"]))
            impulses[name] = 0.0
        self._state["impulse_events"] = 0
        self._state["updates"] += 1
        self.ctx.set_shared("affect.values", dict(values), source=self.name)
        self.ctx.set_shared(
            "affect.exploration_bias", self.exploration_bias(), source=self.name
        )
        self.ctx.set_shared("affect.attention_gain", self.attention_gain(), source=self.name)
        self.ctx.set_shared("affect.learning_gain", self.learning_gain(), source=self.name)
        self.ctx.set_shared(
            "affect.memory_importance_gain", self.memory_importance_gain(), source=self.name
        )
        self.ctx.set_shared("affect.goal_bias", self.goal_bias(), source=self.name)

    def exploration_bias(self) -> float:
        if self.ablation == "none":
            return float(self.bias.get("base", 0.25))
        values = self._state["values"]
        bias = (
            self.bias.get("base", 0.25)
            + self.bias.get("curiosity_gain", 0.35) * values.get("curiosity", 0.0)
            + self.bias.get("frustration_gain", 0.20) * values.get("frustration", 0.0)
            + self.bias.get("fatigue_gain", -0.15) * values.get("fatigue", 0.0)
        )
        return float(np.clip(bias, 0.0, 0.95))

    # ------------------------------------------------------- causal pathways
    # MASTER SPEC 7.2 lists five paths from affect into the rest of the system.
    # Each is a separate, measurable scalar so an ablation can remove them all.
    def attention_gain(self) -> float:
        """Path 1: attention. Scales how strongly percepts enter the core."""
        if self.ablation == "none":
            return 1.0
        return float(0.8 + 0.4 * self._state["values"].get("curiosity", 0.5))

    def learning_gain(self) -> float:
        """Path 2: learning. Scales the online predictor's learning rate."""
        if self.ablation == "none":
            return 1.0
        values = self._state["values"]
        return float(
            np.clip(0.6 + values.get("curiosity", 0.5) - 0.3 * values.get("fatigue", 0.0), 0.2, 1.6)
        )

    def memory_importance_gain(self) -> float:
        """Path 3: memory. Scales the importance attached to new rules."""
        if self.ablation == "none":
            return 1.0
        values = self._state["values"]
        return float(np.clip(0.6 + values.get("stress_load", 0.0) + 0.4 * values.get("relief", 0.0), 0.3, 1.8))

    def goal_bias(self) -> dict[str, float]:
        """Path 4: goals. Multipliers for the goal scoring in M7."""
        if self.ablation == "none":
            return {"explore": 1.0, "recharge": 1.0, "rest": 1.0, "consolidate": 1.0}
        values = self._state["values"]
        return {
            "explore": float(0.6 + values.get("curiosity", 0.5)),
            "recharge": float(1.0 + 0.8 * values.get("stress_load", 0.0)),
            "rest": float(0.5 + values.get("fatigue", 0.0)),
            "consolidate": float(0.8 + values.get("relief", 0.0)),
        }

    def reset_to_baseline(self) -> None:
        """Manual recovery: return every channel to its configured baseline."""
        for name, spec in self.spec.items():
            self._state["values"][name] = float(spec["init"])
            self._state["impulses"][name] = 0.0
        self._state["impulse_events"] = 0
        self._state["resets"] = int(self._state.get("resets", 0)) + 1

    def snapshot_values(self) -> dict[str, float]:
        return {name: round(value, 6) for name, value in self._state["values"].items()}

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "updates": int(self._state["updates"]),
                "exploration_bias": round(self.exploration_bias(), 6),
                **{f"affect_{k}": round(v, 4) for k, v in self._state["values"].items()},
            }
        )
        return base
