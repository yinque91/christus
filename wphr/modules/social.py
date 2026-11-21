"""M6 Social: other-agent behaviour model with explicit uncertainty.

The module only ever consumes *observations of another agent's behaviour*
(its position and last move). It never reads the agent's hidden cooperation
probability, and it never models the agent as if it had verified intentions.

Three things are produced and logged:

* a transition model over the observed agent's moves (next-move prediction),
* an intent hypothesis (\"cooperative\") with a confidence that grows with
  evidence and can be revised by contrary interaction outcomes,
* online prediction accuracy for the transition model and for a frequency
  baseline, so \"the model beats a trivial predictor\" is a measured claim.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from wphr.core.event import Event, EventType
from wphr.core.module import BaseModule

MOVE_NAMES = ("move_north", "move_east", "move_south", "move_west", "wait")
MOVE_VECTORS = {
    "move_north": (0.0, -1.0),
    "move_east": (1.0, 0.0),
    "move_south": (0.0, 1.0),
    "move_west": (-1.0, 0.0),
    "wait": (0.0, 0.0),
}


def _move_name(vector: tuple[float, float]) -> str:
    dx, dy = float(vector[0]), float(vector[1])
    for name, (vx, vy) in MOVE_VECTORS.items():
        if abs(dx - vx) < 1e-6 and abs(dy - vy) < 1e-6:
            return name
    return "wait"


class SocialModule(BaseModule):
    name: ClassVar[str] = "social"
    description: ClassVar[str] = "other-agent transition model, intent hypothesis with confidence"
    group: ClassVar[str] = "slow"

    def reset(self) -> None:
        cfg = self.ctx.config.get("social", {}) or {}
        self.mode = str(cfg.get("mode", "transition"))
        self.trust_lr = float(cfg.get("trust_lr", 0.1))
        self._state: dict[str, Any] = {
            "agents": {},
            "last_move": None,
            "pending_prediction": None,
            "predictions": 0,
            "correct": 0,
            "frequency_predictions": 0,
            "frequency_correct": 0,
            "global_moves": {name: 0 for name in MOVE_NAMES},
            "transitions": {},
            "interactions": 0,
            "hint": (0.0, 0.0),
        }

    # ------------------------------------------------------------------ event
    def on_event(self, event: Event) -> None:
        super().on_event(event)
        if event.event_type == EventType.OBSERVATION.value:
            # Per-step observation: the behaviour model has to learn from every
            # observation, not from the slow scheduler tick. The observation
            # carries the agent snippet directly, so this does not depend on
            # another module having published to shared state first.
            derived = (event.observation or {}).get("derived") or {}
            agent = derived.get("agent")
            if agent:
                self._observe_agent(agent)
                self._publish()
            return
        if event.event_type != EventType.OUTCOME.value:
            return
        outcome = event.actual_outcome or {}
        info = outcome.get("info") or {}
        social = info.get("social")
        if not social:
            return
        agent = self._state["agents"].setdefault(
            "nearest", {"cooperate": 0, "refuse": 0, "trust": 0.5}
        )
        self._state["interactions"] += 1
        if social.get("cooperative"):
            agent["cooperate"] += 1
            agent["trust"] = float(min(1.0, agent["trust"] + self.trust_lr))
        else:
            agent["refuse"] += 1
            agent["trust"] = float(max(0.0, agent["trust"] - self.trust_lr))

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        self._publish()

    def _observe_agent(self, observation: dict[str, Any]) -> None:
        move = _move_name(tuple(observation.get("last_move", (0.0, 0.0))))
        counts = self._state["global_moves"]
        counts[move] = counts.get(move, 0) + 1

        # score the pending predictions against the move actually observed
        pending = self._state["pending_prediction"]
        if pending:
            self._state["predictions"] += 1
            if pending["model"] == move:
                self._state["correct"] += 1
            self._state["frequency_predictions"] += 1
            if pending["frequency"] == move:
                self._state["frequency_correct"] += 1

        previous = self._state["last_move"]
        if previous is not None:
            key = f"{previous}->{move}"
            transitions = self._state["transitions"]
            transitions[key] = int(transitions.get(key, 0)) + 1

        predicted, basis = self._predict(move)
        frequency = max(MOVE_NAMES, key=lambda name: counts.get(name, 0))
        self._state["pending_prediction"] = {"model": predicted, "frequency": frequency}
        self._state["last_move"] = move
        self._state["hint"] = MOVE_VECTORS.get(predicted, (0.0, 0.0))
        agent = self._state["agents"].setdefault(
            "nearest", {"cooperate": 0, "refuse": 0, "trust": 0.5}
        )
        agent.update(
            {
                "last_move": move,
                "predicted_move": predicted,
                "prediction_basis": basis,
                "confidence": self._confidence(),
                "observed_moves": sum(counts.values()),
            }
        )

    def _predict(self, current_move: str) -> tuple[str, str]:
        transitions = self._state["transitions"]
        if self.mode == "frequency":
            counts = self._state["global_moves"]
            return max(MOVE_NAMES, key=lambda name: counts.get(name, 0)), "frequency"
        row = {
            key.split("->")[1]: count
            for key, count in transitions.items()
            if key.startswith(f"{current_move}->")
        }
        if row:
            return max(row, key=lambda name: row[name]), "transition"
        counts = self._state["global_moves"]
        return max(MOVE_NAMES, key=lambda name: counts.get(name, 0)), "fallback"

    def _confidence(self) -> float:
        observations = int(sum(self._state["global_moves"].values()))
        return float(round(1.0 - math.exp(-observations / 8.0), 6))

    def _publish(self) -> None:
        agent = self._state["agents"].get("nearest", {})
        intend = {
            "cooperate": int(agent.get("cooperate", 0)),
            "refuse": int(agent.get("refuse", 0)),
        }
        total = intend["cooperate"] + intend["refuse"]
        intent_probability = (intend["cooperate"] + 1.0) / (total + 2.0)
        intent_confidence = float(1.0 - math.exp(-total / 3.0))
        predictions = int(self._state["predictions"])
        accuracy = (
            float(self._state["correct"]) / predictions if predictions else 0.0
        )
        frequency_predictions = int(self._state["frequency_predictions"])
        frequency_accuracy = (
            float(self._state["frequency_correct"]) / frequency_predictions
            if frequency_predictions
            else 0.0
        )
        self.ctx.set_shared("social.agent_count", len(self._state["agents"]), source=self.name)
        self.ctx.set_shared(
            "social.trust_mean", float(agent.get("trust", 0.5)), source=self.name,
            confidence=intent_confidence,
        )
        self.ctx.set_shared(
            "social.intent",
            {
                "hypothesis": "cooperative",
                "probability": round(float(intent_probability), 6),
                "confidence": round(float(intent_confidence), 6),
                "evidence": intend,
            },
            source=self.name,
        )
        self.ctx.set_shared("social.predicted_move", agent.get("predicted_move", "wait"), source=self.name)
        self.ctx.set_shared("social.hint", list(self._state["hint"]), source=self.name)
        self.ctx.set_shared("social.prediction_accuracy", round(accuracy, 6), source=self.name)
        self.ctx.set_shared(
            "social.frequency_accuracy", round(frequency_accuracy, 6), source=self.name
        )
        self.ctx.set_shared(
            "social.predictions", predictions, source=self.name
        )

    # ------------------------------------------------------------- accessors
    def model(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "transitions": dict(self._state["transitions"]),
            "global_moves": dict(self._state["global_moves"]),
            "predictions": int(self._state["predictions"]),
            "correct": int(self._state["correct"]),
            "frequency_predictions": int(self._state["frequency_predictions"]),
            "frequency_correct": int(self._state["frequency_correct"]),
            "agents": self._state["agents"],
            "interactions": int(self._state["interactions"]),
        }

    def prediction_accuracy(self) -> float:
        predictions = int(self._state["predictions"])
        return float(self._state["correct"]) / predictions if predictions else 0.0

    def frequency_accuracy(self) -> float:
        predictions = int(self._state["frequency_predictions"])
        return float(self._state["frequency_correct"]) / predictions if predictions else 0.0

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "agents": len(self._state["agents"]),
                "interactions": int(self._state["interactions"]),
                "predictions": int(self._state["predictions"]),
                "accuracy": round(self.prediction_accuracy(), 6),
                "frequency_accuracy": round(self.frequency_accuracy(), 6),
            }
        )
        return base
