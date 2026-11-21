"""Small, auditable action readout from the existing SM-CTRN state.

The readout is deliberately tiny: one linear projection of ``[h; c]`` (264
inputs) plus a bias, trained online by normalised LMS to predict the observed
ball offset ``scent_x``. Its scalar prediction is mapped to a fixed left/right
motor bias on top of Practice's action scores. It is *not* a new recurrent
core, a hidden layer, or a replacement policy.

Modes (configured under ``christus.core_readout``):

* ``active``  - read the current ``[h; c]`` and learn from the observed scent.
* ``masked``  - feed a zero vector; output is zero and weights are not updated.
* ``shuffled``- feed a state from the recent past; same learning rule. This
  preserves the readout capacity and training signal while destroying the
  temporal correspondence between state and current observation.

The module is registered by :class:`christus.runtime.ChristusSubject` and is
only consulted through ``Subject.action_hook`` after ``core.step``. SAFE leaves
it disabled, so the default runtime path is unchanged.
"""
from __future__ import annotations

import numpy as np

from wphr.core.event import EventType
from wphr.core.module import BaseModule
from wphr.modules.practice import ACTIONS, ACTION_INDEX

EAST = ACTION_INDEX["move_east"]
WEST = ACTION_INDEX["move_west"]


class CoreActionReadout(BaseModule):
    name = "core_readout"
    description = "small linear h/c -> left/right motor bias readout"
    group = "fast"

    # ------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        cfg = self.ctx.config.get("christus.core_readout", {}) or {}
        self.mode = str(cfg.get("mode", "active"))
        if self.mode not in ("active", "masked", "shuffled"):
            self.mode = "active"
        self.gain = float(cfg.get("gain", 2.0))
        self.lr = float(cfg.get("lr", 0.05))
        self.deadzone = float(cfg.get("deadzone", 0.02))
        self.shuffle_buffer = max(2, int(cfg.get("shuffle_buffer", 1000)))
        # ``permute`` (default): randomly permute the 264 state components each
        # step. This preserves the value distribution and the readout capacity
        # while destroying the dimension<->observation correspondence. A
        # short-lag time shuffle is not a valid control because the ball offset
        # is autocorrelated; ``time`` is retained only for diagnostics.
        self.shuffle_kind = str(cfg.get("shuffle_kind", "permute"))
        self.shuffle_min_lag = max(1, int(cfg.get("shuffle_min_lag", 300)))
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.core = None
        self.practice = None
        self._state = {
            "mode": self.mode,
            "input_dim": 0,
            "target": 0.0,
            "visible": False,
            "pending": None,
            "buffer": [],
            "steps": 0,
            "overrides": 0,
            "aligned": 0,
            "last_pred": 0.0,
            "last_bias": 0.0,
            "last_error": 0.0,
            "last_base_action": None,
            "last_final_action": None,
            "last_override": False,
            "last_aligned": False,
        }

    def attach(self, core, practice) -> None:
        """Bind to the subject's core and Practice after both exist."""
        self.core = core
        self.practice = practice
        dim = int(core.total_units) + int(core.concept_dim)
        self.w = np.zeros(dim, dtype=np.float64)
        self.b = 0.0
        self._state["input_dim"] = dim

    # ----------------------------------------------------------------- event
    def on_event(self, event) -> None:
        super().on_event(event)
        if not self.enabled:
            return
        if event.event_type == EventType.OBSERVATION.value:
            derived = (event.observation or {}).get("derived", {}) or {}
            scent = derived.get("scent", (0.0, 0.0)) or (0.0, 0.0)
            self._state["target"] = float(scent[0])
            self._state["visible"] = bool(
                derived.get("cue") == "ball" or abs(float(scent[0])) > 1e-9
            )
        elif event.event_type == EventType.OUTCOME.value:
            self._learn()

    # --------------------------------------------------------------- vectors
    def _current_vector(self) -> np.ndarray | None:
        if self.core is None:
            return None
        return np.concatenate(
            [
                np.asarray(self.core.h, dtype=np.float64),
                np.asarray(self.core.c, dtype=np.float64),
            ]
        )

    def _readout_vector(self, step: int) -> np.ndarray | None:
        current = self._current_vector()
        if current is None:
            return None
        if self.mode == "masked":
            return np.zeros_like(current)
        if self.mode == "shuffled":
            if self.shuffle_kind == "permute":
                # Deterministic per-step permutation; does not consume the
                # subject RNG. Training and action use the same shuffled input.
                seed = (
                    int(step) * 6364136223846793005 + 1442695040888963407
                ) & 0xFFFFFFFFFFFFFFFF
                rng = np.random.default_rng(seed)
                return current[rng.permutation(current.size)]
            buffer = self._state["buffer"]
            if len(buffer) >= self.shuffle_min_lag + 2:
                candidates = buffer[: len(buffer) - self.shuffle_min_lag]
                # Deterministic index: reproducible without consuming the
                # subject RNG, and independent of action-selection randomness.
                index = (int(step) * 2654435761) % len(candidates)
                return np.asarray(candidates[index], dtype=np.float64)
            return None
        return current

    def _remember(self, current: np.ndarray) -> None:
        if self.mode != "shuffled" or self.shuffle_kind != "time":
            return
        buffer = self._state["buffer"]
        buffer.append(current.tolist())
        if len(buffer) > self.shuffle_buffer:
            del buffer[: len(buffer) - self.shuffle_buffer]

    # ---------------------------------------------------------------- action
    def action(self, subject) -> tuple[str, str] | None:
        """Called after ``core.step``; returns None to keep Practice's action."""
        if not self.enabled or self.w is None or self.practice is None:
            return None
        current = self._current_vector()
        if current is None:
            return None
        # Keep the current state for the shuffled control's future draws before
        # sampling; the sampled candidates exclude the most recent min_lag.
        self._remember(current)
        x = self._readout_vector(int(subject.clock.step))
        if x is None:
            return None

        base_scores = getattr(self.practice, "last_scores", None)
        if not base_scores:
            return None
        base_action = self.practice._state.get("last_action")
        if base_action is None:
            return None

        pred = 0.0 if self.mode == "masked" else float(x @ self.w + self.b)
        target = float(self._state["target"])
        bias = float(self.gain * pred)
        self._state["steps"] += 1
        self._state["last_pred"] = pred
        self._state["last_bias"] = bias
        self._state["last_base_action"] = base_action
        self._state["last_final_action"] = base_action
        self._state["last_override"] = False
        self._state["last_aligned"] = False
        self._state["last_error"] = abs(pred - target)
        self._state["pending"] = {
            "x": x.tolist(),
            "pred": pred,
            "target": target,
            "base_action": base_action,
            "step": int(subject.clock.step),
        }
        self._publish(subject, pred=pred, bias=bias, final=base_action, override=False,
                      aligned=False, target=target)

        if self.mode == "masked" or abs(pred) <= self.deadzone:
            return None
        scores = [float(v) for v in base_scores]
        scores[EAST] += bias
        scores[WEST] -= bias
        base_best = max(float(v) for v in base_scores)
        if max(scores) <= base_best + 1e-12:
            return None
        final = ACTIONS[int(np.argmax(scores))]
        if final not in ("move_east", "move_west") or final == base_action:
            return None

        aligned = (final == "move_east" and target > 0) or (
            final == "move_west" and target < 0
        )
        self._state["overrides"] += 1
        if aligned:
            self._state["aligned"] += 1
        self._state["last_final_action"] = final
        self._state["last_override"] = True
        self._state["last_aligned"] = bool(aligned)
        # Let the current step's feature vector and logs see the final action.
        self.practice._state["last_action"] = final
        self.ctx.set_shared("practice.action", final, source=self.name)
        self._publish(subject, pred=pred, bias=bias, final=final, override=True,
                      aligned=bool(aligned), target=target)
        return final, f"core_readout.{self.mode}"

    def _publish(self, subject, *, pred: float, bias: float, final: str | None,
                 override: bool, aligned: bool, target: float) -> None:
        subject.shared.set("readout.pred", float(pred), source=self.name,
                           step=self.ctx.clock.step, confidence=1.0)
        subject.shared.set("readout.bias", float(bias), source=self.name,
                           step=self.ctx.clock.step, confidence=1.0)
        subject.shared.set("readout.base_action", self._state["last_base_action"],
                           source=self.name, step=self.ctx.clock.step, confidence=1.0)
        subject.shared.set("readout.final_action", final, source=self.name,
                           step=self.ctx.clock.step, confidence=1.0)
        subject.shared.set("readout.override", bool(override), source=self.name,
                           step=self.ctx.clock.step, confidence=1.0)
        subject.shared.set("readout.aligned", bool(aligned), source=self.name,
                           step=self.ctx.clock.step, confidence=1.0)
        subject.shared.set("readout.error", float(abs(pred - target)), source=self.name,
                           step=self.ctx.clock.step, confidence=1.0)

    # ----------------------------------------------------------------- learn
    # ------------------------------------------------------- coordination API
    def predict_state(self, h=None, c=None) -> float:
        """Current linear state->scent readout, reusable by the coordinator."""
        if self.w is None:
            return 0.0
        hh = np.asarray(self.core.h if h is None else h, dtype=np.float64)
        cc = np.asarray(self.core.c if c is None else c, dtype=np.float64)
        return float(np.concatenate([hh, cc]) @ self.w + self.b)

    def learn_state(self, target: float, h=None, c=None) -> float:
        """Normalised LMS update of the same readout, without changing its action path."""
        if self.w is None:
            return 0.0
        hh = np.asarray(self.core.h if h is None else h, dtype=np.float64)
        cc = np.asarray(self.core.c if c is None else c, dtype=np.float64)
        x = np.concatenate([hh, cc])
        pred = float(x @ self.w + self.b)
        error = float(target) - pred
        denom = float(x @ x) + 1e-6
        self.w += (self.lr * error / denom) * x
        self.b += 0.1 * self.lr * error
        self._state["last_pred"] = pred
        self._state["last_error"] = error
        return error

    def _learn(self) -> None:
        pending = self._state.get("pending")
        if not pending or self.w is None or self.mode == "masked":
            return
        x = np.asarray(pending["x"], dtype=np.float64)
        target = float(pending["target"])
        error = target - float(pending["pred"])
        denom = float(x @ x) + 1e-6
        self.w += (self.lr * error / denom) * x
        self.b += 0.1 * self.lr * error
        self._state["last_error"] = error
        self._state["pending"] = None

    # ------------------------------------------------------------ diagnostics
    def metrics(self) -> dict:
        base = super().metrics()
        steps = int(self._state["steps"])
        overrides = int(self._state["overrides"])
        base.update(
            {
                "mode": self.mode,
                "enabled": bool(self.enabled),
                "input_dim": int(self._state["input_dim"]),
                "readout_steps": steps,
                "overrides": overrides,
                "override_rate": round(overrides / steps, 6) if steps else 0.0,
                "aligned": int(self._state["aligned"]),
                "alignment": round(self._state["aligned"] / overrides, 6)
                if overrides
                else None,
                "last_pred": round(float(self._state["last_pred"]), 6),
                "last_bias": round(float(self._state["last_bias"]), 6),
                "last_error": round(float(self._state["last_error"]), 6),
                "weight_norm": round(float(np.linalg.norm(self.w)), 6)
                if self.w is not None
                else 0.0,
                "bias": round(float(self.b), 6),
            }
        )
        return base

    # ------------------------------------------------------------ checkpoint
    def snapshot(self) -> dict:
        payload = super().snapshot()
        # BaseModule.snapshot deep-copies _state. Keep JSON-serialisable values
        # there; the numpy weight vector is exported separately as a list.
        payload["state"]["w"] = None if self.w is None else self.w.tolist()
        payload["state"]["b"] = float(self.b)
        return payload

    def restore(self, payload: dict) -> None:
        super().restore(payload)
        state = payload.get("state", {})
        w = state.get("w")
        if w is not None:
            self.w = np.asarray(w, dtype=np.float64)
        self.b = float(state.get("b", 0.0))
        self.mode = str(state.get("mode", self.mode))
        self._state.setdefault("buffer", [])
        if "input_dim" in state:
            self._state["input_dim"] = int(state["input_dim"])
