"""M1 World: uncertain world model with a learned one-step predictor.

The predictor is a bounded normalised-LMS linear model over the observation
vector (see :mod:`wphr.learning.prediction`). Two things are therefore real and
measurable rather than asserted: the prediction error decreases as the model
adapts, and the error is an explicit, separately logged quantity.

Everything published by this module carries its provenance. The derived cue and
patch come from the vision sensor; if the vision reading is invalid the module
publishes ``cue="none"`` plus the missing-sensor list instead of guessing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from wphr.core.event import Event, EventType
from wphr.core.module import BaseModule
from wphr.learning.prediction import LinearPredictor


class WorldModule(BaseModule):
    name: ClassVar[str] = "world"
    description: ClassVar[str] = "uncertain world model: learned one-step prediction + percept"
    group: ClassVar[str] = "fast"

    def reset(self) -> None:
        cfg = self.ctx.config.get("learning.prediction", {}) or {}
        self.predictor_cfg = {
            "lr": float(cfg.get("lr", 0.03)),
            "weight_bound": float(cfg.get("weight_bound", 4.0)),
            "output_bound": float(cfg.get("output_bound", 10.0)),
        }
        self.predictor: LinearPredictor | None = None
        self._state: dict[str, Any] = {
            "obs": [],
            "prediction": [],
            "error_ema": 0.0,
            "error_last": 0.0,
            "error_n": 0,
            "obs_count": 0,
            "prediction_updates": 0,
            "cue": "none",
            "patch": [],
            "scent": [0.0, 0.0],
            "loudness": 0.0,
            "valid_ratio": 1.0,
            "missing_sensors": [],
            "agent": None,
            "energy": 1.0,
        }

    # ------------------------------------------------------------------ event
    def on_event(self, event: Event) -> None:
        super().on_event(event)
        if event.event_type != EventType.OBSERVATION.value:
            return
        payload = event.observation or {}
        values = payload.get("values")
        if values is None:
            return
        if "energy" in payload:
            self._state["energy"] = float(payload["energy"])
        obs = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.predictor is None:
            self.predictor = LinearPredictor(obs.size, **self.predictor_cfg)
        elif self.predictor.dim != obs.size:
            self.predictor = LinearPredictor(obs.size, **self.predictor_cfg)

        previous = np.asarray(self._state["obs"], dtype=np.float64)
        if previous.size == obs.size and self._state["obs_count"] > 0:
            # affect modulates the online learning rate (MASTER SPEC 7.2 path 2)
            gain = float(self.ctx.get_shared("affect.learning_gain", 1.0))
            base_lr = self.predictor_cfg["lr"]
            self.predictor.lr = float(max(1e-4, min(0.5, base_lr * gain)))
            error = self.predictor.observe(previous, obs)
            self._state["learning_gain"] = gain
            self._state["error_last"] = error
            self._state["error_ema"] = self.predictor.error_ema
            self._state["error_n"] += 1
        self._state["prediction"] = self.predictor.predict(obs).tolist()
        self._state["prediction_updates"] = self.predictor.updates
        self._state["obs"] = obs.tolist()
        self._state["obs_count"] += 1

        derived = payload.get("derived") or {}
        if derived:
            self._state["cue"] = str(derived.get("cue", "none"))
            self._state["patch"] = [float(v) for v in derived.get("patch", [])]
            scent = derived.get("scent", (0.0, 0.0))
            self._state["scent"] = [float(scent[0]), float(scent[1])]
            self._state["loudness"] = float(derived.get("loudness", 0.0))
            self._state["valid_ratio"] = float(derived.get("valid_ratio", 1.0))
            self._state["missing_sensors"] = list(derived.get("missing", []))
            self._state["agent"] = derived.get("agent")

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        conf = self.confidence()
        self.ctx.set_shared("world.cue", self._state["cue"], source=self.name, confidence=conf)
        self.ctx.set_shared("world.patch", self._state["patch"], source=self.name, confidence=conf)
        self.ctx.set_shared("world.scent", self._state["scent"], source=self.name, confidence=conf)
        self.ctx.set_shared("world.prediction", self._state["prediction"], source=self.name, confidence=conf)
        self.ctx.set_shared("world.error_ema", self._state["error_ema"], source=self.name, confidence=conf)
        self.ctx.set_shared(
            "world.sensing_valid_ratio", self._state["valid_ratio"], source=self.name,
            confidence=conf,
        )
        self.ctx.set_shared("world.agent", self._state.get("agent"), source=self.name, confidence=conf)
        self.ctx.set_shared("world.energy", float(self._state["energy"]), source=self.name, confidence=conf)

    def confidence(self) -> float:
        """Confidence in the world model, degraded by error and by sensor dropouts."""
        error_term = 1.0 / (1.0 + float(self._state["error_ema"]))
        valid = float(self._state.get("valid_ratio", 1.0))
        return float(np.clip(0.5 * error_term + 0.5 * valid, 0.0, 1.0))

    @property
    def last_prediction_error(self) -> float:
        return float(self._state["error_last"])

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "obs_count": int(self._state["obs_count"]),
                "error_ema": round(float(self._state["error_ema"]), 6),
                "error_last": round(float(self._state["error_last"]), 6),
                "prediction_updates": int(self._state["prediction_updates"]),
                "cue": self._state["cue"],
                "valid_ratio": round(float(self._state["valid_ratio"]), 4),
            }
        )
        return base

    # ------------------------------------------------------------ checkpoint
    def snapshot(self) -> dict[str, Any]:
        payload = super().snapshot()
        if self.predictor is not None:
            payload["predictor"] = {
                "W": self.predictor.W.tolist(),
                "b": self.predictor.b.tolist(),
                "updates": self.predictor.updates,
                "error_ema": self.predictor.error_ema,
            }
        return payload

    def restore(self, payload: dict[str, Any]) -> None:
        super().restore(payload)
        predictor = payload.get("predictor")
        if predictor:
            self.predictor = LinearPredictor(len(predictor["b"]), **self.predictor_cfg)
            self.predictor.restore(
                {
                    "W": np.asarray(predictor["W"], dtype=np.float64),
                    "b": np.asarray(predictor["b"], dtype=np.float64),
                }
            )
            self.predictor.updates = int(predictor.get("updates", 0))
            self.predictor.error_ema = float(predictor.get("error_ema", 0.0))
