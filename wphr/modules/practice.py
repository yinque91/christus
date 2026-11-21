"""M2 Practice: contextual action values, exploration and navigation primitive.

Two distinct mechanisms live here and are deliberately kept separate:

* **Cue-conditional value learning** — a contextual epsilon-greedy bandit over
  (faced-cell cue, action). This is the *learned* part: which cue is worth
  interacting with is discovered from realised outcomes and therefore transfers
  to layouts the subject has never seen, because the value is keyed by cue type
  rather than by position.
* **Navigation primitive** — a hand-coded scent-following bias. It is an action
  primitive (MASTER SPEC 4.3), not a learned skill, and is documented as such so
  no result is misattributed to learning.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from wphr.core.event import Event, EventType
from wphr.core.module import BaseModule

ACTIONS: tuple[str, ...] = (
    "move_north",
    "move_east",
    "move_south",
    "move_west",
    "interact",
    "wait",
)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
MOVE_VECTORS = {
    "move_north": (0.0, -1.0),
    "move_east": (1.0, 0.0),
    "move_south": (0.0, 1.0),
    "move_west": (-1.0, 0.0),
}
CUES: tuple[str, ...] = (
    "wall",
    "floor",
    "food",
    "hazard",
    "switch_blue",
    "switch_orange",
    "tool",
    "agent",
    "none",
)


class PracticeModule(BaseModule):
    name: ClassVar[str] = "practice"
    description: ClassVar[str] = "contextual action values + exploration + navigation primitive"
    group: ClassVar[str] = "fast"

    def reset(self) -> None:
        cfg = self.ctx.config.get("practice", {}) or {}
        self.navigation_gain = float(cfg.get("navigation_gain", 1.2))
        self.learn_cues = bool(cfg.get("learn_cues", True))
        self.epsilon_floor = float(cfg.get("epsilon_floor", 0.05))
        self.epsilon_ceiling = float(cfg.get("epsilon_ceiling", 0.90))
        self._state: dict[str, Any] = {
            "q": {cue: [0.0] * len(ACTIONS) for cue in CUES},
            "counts": {cue: [0] * len(ACTIONS) for cue in CUES},
            "last_action": None,
            "last_reward": 0.0,
            "total_reward": 0.0,
            "actions": 0,
            "interactions": 0,
            "cue_visits": {cue: 0 for cue in CUES},
            "learned_updates": 0,
            "epsilon": 0.25,
        }
        # Experiment control surface: None means "use the configured behaviour".
        self.epsilon_override: float | None = None
        self.learn_override: bool | None = None

    def set_mode(self, *, epsilon: float | None = None, learn: bool | None = None) -> None:
        """Force exploration/learning for evaluation phases (used by experiments)."""
        self.epsilon_override = epsilon
        self.learn_override = learn

    # ------------------------------------------------------------------ event
    def on_event(self, event: Event) -> None:
        super().on_event(event)
        if event.event_type != EventType.OUTCOME.value:
            return
        outcome = event.actual_outcome or {}
        reward = float(outcome.get("reward", 0.0))
        self._state["last_reward"] = reward
        self._state["total_reward"] += reward
        learn = self.learn_cues if self.learn_override is None else self.learn_override
        if not learn:
            return
        action = outcome.get("action") or self._state["last_action"]
        cue = str((outcome.get("info") or {}).get("cue", "none"))
        memory = getattr(self.ctx, "memory", None)
        if memory is not None:
            memory.record_outcome(
                cue=cue,
                action=str(action),
                reward=reward,
                step=int(event.step),
                enabled=True,
                importance_scale=float(self.ctx.get_shared("affect.memory_importance_gain", 1.0)),
            )
        if action in ACTION_INDEX and cue in self._state["q"]:
            index = ACTION_INDEX[action]
            counts = self._state["counts"][cue]
            q = self._state["q"][cue]
            counts[index] += 1
            lr = 1.0 / counts[index]
            q[index] += lr * (reward - q[index])
            self._state["learned_updates"] += 1
            if action == "interact":
                self._state["interactions"] += 1

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        cue = str(self.ctx.get_shared("world.cue", "none"))
        if cue not in self._state["q"]:
            cue = "none"
        self._state["cue_visits"][cue] += 1
        scent = self.ctx.get_shared("world.scent", [0.0, 0.0]) or [0.0, 0.0]
        if self.epsilon_override is not None:
            epsilon = float(np.clip(self.epsilon_override, 0.0, 1.0))
        else:
            epsilon = float(
                np.clip(
                    self.ctx.get_shared("affect.exploration_bias", 0.25),
                    self.epsilon_floor,
                    self.epsilon_ceiling,
                )
            )
            # Memory-driven uncertainty: a cue whose supporting evidence is old
            # or thin is explored more. This is the mechanism that counters
            # over-fixation after a rule change.
            memory = getattr(self.ctx, "memory", None)
            if memory is not None:
                epsilon = float(
                    np.clip(
                        epsilon + memory.exploration_bonus(cue, int(step)),
                        self.epsilon_floor,
                        self.epsilon_ceiling,
                    )
                )
        self._state["epsilon"] = epsilon

        scores = self._scores(cue, scent)
        if self.ctx.rng.random() < epsilon:
            action = ACTIONS[int(self.ctx.rng.integers(len(ACTIONS)))]
        else:
            best = max(scores)
            candidates = [i for i, value in enumerate(scores) if abs(value - best) < 1e-12]
            action = ACTIONS[candidates[int(self.ctx.rng.integers(len(candidates)))]]

        self._state["last_action"] = action
        self._state["actions"] += 1
        self.ctx.set_shared("practice.action", action, source=self.name)
        self.ctx.set_shared(
            "practice.value_mean",
            float(np.mean([v for cue_values in self._state["q"].values() for v in cue_values])),
            source=self.name,
        )
        self.ctx.set_shared("practice.epsilon", epsilon, source=self.name)
        self.ctx.set_shared("practice.cue", cue, source=self.name)

    def _scores(self, cue: str, scent: list[float]) -> list[float]:
        scores = list(self._state["q"][cue])
        goal = str(self.ctx.get_shared("volition.goal", "explore"))
        gain = self.navigation_gain
        if goal == "recharge":
            gain *= 1.6
            scores[ACTION_INDEX["interact"]] += 0.45
        elif goal == "rest":
            scores[ACTION_INDEX["wait"]] += 0.6
        elif goal == "consolidate":
            scores[ACTION_INDEX["wait"]] += 0.2
        # The navigation primitive is a *prior*: it only shapes the choice when
        # nothing positive has been learned for this cue. Otherwise a learned
        # reward of +1 could be out-weighted by the hand-coded bias and the
        # agent would walk away from food it has already learned to eat.
        if scores and max(scores) > 0.0 and goal not in ("recharge",):
            return scores
        sx, sy = float(scent[0]), float(scent[1])
        norm = float(np.hypot(sx, sy))
        if norm > 1e-9:
            ux, uy = sx / norm, sy / norm
            for index, action in enumerate(ACTIONS):
                vector = MOVE_VECTORS.get(action)
                if vector is None:
                    continue
                scores[index] += gain * (vector[0] * ux + vector[1] * uy)
        return scores

    # ------------------------------------------------------------- accessors
    def feature_state(self) -> dict[str, Any]:
        last_action = self._state["last_action"]
        return {
            "last_action": last_action,
            "action_index": ACTION_INDEX.get(last_action, 0),
            "last_reward": float(self._state["last_reward"]),
            "value_mean": float(
                np.mean([v for cue_values in self._state["q"].values() for v in cue_values])
            ),
            "epsilon": float(self.ctx.get_shared("affect.exploration_bias", 0.25)),
        }

    def q_table(self) -> dict[str, list[float]]:
        return {cue: [round(v, 6) for v in values] for cue, values in self._state["q"].items()}

    def greedy_actions(self) -> dict[str, str]:
        return {
            cue: ACTIONS[int(np.argmax(values))]
            for cue, values in self._state["q"].items()
        }

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "actions": int(self._state["actions"]),
                "interactions": int(self._state["interactions"]),
                "learned_updates": int(self._state["learned_updates"]),
                "total_reward": round(float(self._state["total_reward"]), 4),
                "last_action": self._state["last_action"],
            }
        )
        return base
