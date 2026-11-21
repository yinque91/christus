"""Bounded autonomous loop with pause/resume/stop and state persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class LoopState:
    step: int = 0
    running: bool = True
    paused: bool = False
    stopped: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    budget_left: int = 50


class AutonomousLoop:
    """Step → perceive → decide → act → check → record; hard bounds, no recursion."""

    def __init__(self, decide: Callable[[LoopState, dict], dict], *,
                 max_steps: int = 20, state_path: str | Path | None = None):
        self.decide = decide
        self.max_steps = max_steps
        self.state = LoopState(budget_left=max_steps)
        self.state_path = Path(state_path) if state_path else None

    def pause(self):
        self.state.paused = True

    def resume(self):
        self.state.paused = False

    def stop(self):
        self.state.stopped = True

    def save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "step": self.state.step, "budget_left": self.state.budget_left,
                "history": self.state.history[-100:],
            }, ensure_ascii=False), encoding="utf-8")

    def load(self):
        if self.state_path and self.state_path.exists():
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state.step = d.get("step", 0)
            self.state.budget_left = d.get("budget_left", self.max_steps)
            self.state.history = d.get("history", [])

    def run(self, perceive: Callable[[], dict], max_rounds: int | None = None) -> list[dict]:
        limit = max_rounds if max_rounds is not None else self.max_steps
        out = []
        while not self.state.stopped and self.state.step < limit and self.state.budget_left > 0:
            if self.state.paused:
                break
            obs = perceive()
            action = self.decide(self.state, obs)
            rec = {"step": self.state.step, "ts": time.time(),
                   "obs": obs, "action": action}
            self.state.history.append(rec)
            self.state.step += 1
            self.state.budget_left -= 1
            out.append(rec)
            if action.get("done"):
                break
        self.save()
        return out
