"""Symbolic planner over action primitives (MASTER SPEC 4.3 and 9.2).

The planner is deliberately *not* learned: it composes the existing action
primitives into a short plan for the currently selected goal, monitors the
outcome of every executed step, and replans or abandons on failure. That is what
makes the autonomy auditable — every action it issues can be attributed to a
plan, and every plan has a visible status.

Goal selection itself stays in M7 (Volition); the planner never chooses goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wphr.util.ids import new_id

MOVE_BY_AXIS = {
    (1, 0): "move_east",
    (-1, 0): "move_west",
    (0, 1): "move_south",
    (0, -1): "move_north",
}

#: when a step is blocked, try the perpendicular direction before giving up
DETOUR = {
    "move_north": ("move_east", "move_west"),
    "move_south": ("move_west", "move_east"),
    "move_east": ("move_south", "move_north"),
    "move_west": ("move_north", "move_south"),
}


@dataclass
class Plan:
    plan_id: str
    goal: str
    actions: list[str]
    created_step: int
    reason: str = ""
    index: int = 0
    executed: int = 0
    failures: int = 0
    status: str = "active"  # active | completed | failed | abandoned

    def next_action(self) -> str | None:
        if self.status != "active" or self.index >= len(self.actions):
            return None
        action = self.actions[self.index]
        self.index += 1
        return action

    @property
    def progress(self) -> float:
        return self.index / len(self.actions) if self.actions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "actions": list(self.actions),
            "index": self.index,
            "executed": self.executed,
            "failures": self.failures,
            "status": self.status,
            "reason": self.reason,
            "progress": round(self.progress, 4),
        }


class Planner:
    def __init__(self, config: Any) -> None:
        section = config.get("planning", {}) or {}
        self.enabled = bool(section.get("enabled", False))
        self.max_length = max(1, int(section.get("max_plan_length", 8)))
        self.max_failures = max(1, int(section.get("max_failures", 2)))
        self.goals = tuple(section.get("goals", ["recharge", "rest", "consolidate"]))
        self.plan: Plan | None = None
        self.history: list[dict[str, Any]] = []
        self.stats: dict[str, int] = {
            "plans": 0,
            "completed": 0,
            "failed": 0,
            "abandoned": 0,
            "steps_executed": 0,
            "detours": 0,
        }

    # ---------------------------------------------------------------- goals
    def handles(self, goal: str) -> bool:
        return self.enabled and goal in self.goals

    # -------------------------------------------------------------- planning
    def should_repropose(self, percept: dict[str, Any]) -> bool:
        """A plan is stale as soon as the situation it was built for changed."""
        if self.plan is None:
            return True
        cue = str(percept.get("cue", "none"))
        remaining = self.plan.actions[self.plan.index:self.plan.index + 1]
        if cue == "food" and remaining != ["interact"]:
            return True
        if cue == "hazard" and self.plan.actions[self.plan.index:self.plan.index + 1] == ["interact"]:
            return True
        return False

    def propose(self, goal: str, percept: dict[str, Any], step: int) -> Plan:
        actions = self._compose(goal, percept)
        reason = f"{goal}:{self._basis(goal, percept)}"
        plan = Plan(
            plan_id=new_id("plan_"),
            goal=goal,
            actions=actions,
            created_step=int(step),
            reason=reason,
        )
        self._retire("abandoned")
        self.plan = plan
        self.stats["plans"] += 1
        return plan

    def _basis(self, goal: str, percept: dict[str, Any]) -> str:
        if goal == "recharge":
            scent = percept.get("scent") or (0.0, 0.0)
            if abs(float(scent[0])) + abs(float(scent[1])) < 1e-6:
                return "no_scent_spin"
            return "scent"
        return "rest"

    def _compose(self, goal: str, percept: dict[str, Any]) -> list[str]:
        cue = str(percept.get("cue", "none"))
        if goal == "recharge" and cue == "food":
            # the resource is right in front of the body: the plan is one action
            return ["interact"]
        if goal == "recharge" and cue == "hazard":
            return ["move_north", "move_east"]
        if goal in ("rest", "consolidate"):
            return ["wait"] * min(3, self.max_length)
        scent = percept.get("scent") or (0.0, 0.0)
        sx, sy = float(scent[0]), float(scent[1])
        if abs(sx) + abs(sy) < 1e-6:
            # nothing to follow: sweep the four directions in a fixed order
            return (["move_north", "move_east", "move_south", "move_west"])[
                : self.max_length
            ]
        actions: list[str] = []
        horizontal = MOVE_BY_AXIS[(1, 0) if sx > 0 else (-1, 0)]
        vertical = MOVE_BY_AXIS[(0, 1) if sy > 0 else (0, -1)]
        half = max(1, self.max_length // 2)
        actions.extend([horizontal] * (half if abs(sx) >= abs(sy) else max(1, half // 2)))
        actions.extend([vertical] * (half if abs(sy) > abs(sx) else max(1, half // 2)))
        return actions[: self.max_length]

    # ------------------------------------------------------------- execution
    def next_action(self) -> str | None:
        if self.plan is None:
            return None
        action = self.plan.next_action()
        if action is None and self.plan.status == "active":
            # ran out of steps without a failure: treat as completed
            self.plan.status = "completed"
            self.stats["completed"] += 1
            self._retire(None)
        return action

    def observe(self, action: str, result: Any) -> str:
        """Feed one executed step back; returns ``continue`` / ``replan`` / ``idle``."""
        if self.plan is None:
            return "idle"
        self.plan.executed += 1
        self.stats["steps_executed"] += 1
        blocked = bool((getattr(result, "info", {}) or {}).get("blocked", False))
        if blocked:
            self.plan.failures += 1
            self.stats["detours"] = self.stats.get("detours", 0) + 1
            if self.plan.failures >= self.max_failures:
                self.plan.status = "failed"
                self.stats["failed"] += 1
                self.plan = None
                return "replan"
            # failure recovery: keep the goal, change the route
            options = DETOUR.get(action)
            if options:
                detour = options[(self.plan.failures - 1) % len(options)]
                remaining = max(1, len(self.plan.actions) - self.plan.index)
                self.plan.actions = (
                    self.plan.actions[: self.plan.index]
                    + [detour] * min(remaining, 2)
                )
            return "detour"
        if self.plan is not None and self.plan.index >= len(self.plan.actions):
            self.plan.status = "completed"
            self.stats["completed"] += 1
            self.plan = None
            return "idle"
        return "continue"

    def abandon(self) -> None:
        self._retire("abandoned")

    def _retire(self, status: str | None) -> None:
        if self.plan is None:
            return
        if status is not None:
            self.plan.status = status
            key = "abandoned" if status == "abandoned" else "failed"
            self.stats[key] += 1
        self.history.append(self.plan.to_dict())
        if len(self.history) > 64:
            del self.history[:-64]
        self.plan = None

    # ------------------------------------------------------------- reporting
    def state(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "goals": list(self.goals),
            "active": self.plan.to_dict() if self.plan else None,
            "stats": dict(self.stats),
            "recent": self.history[-5:],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "history": self.history[-32:],
            "stats": dict(self.stats),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        plan = payload.get("plan")
        if plan:
            self.plan = Plan(
                plan_id=str(plan["plan_id"]),
                goal=str(plan["goal"]),
                actions=list(plan["actions"]),
                created_step=int(plan.get("created_step", 0)),
                reason=str(plan.get("reason", "")),
                index=int(plan.get("index", 0)),
                executed=int(plan.get("executed", 0)),
                failures=int(plan.get("failures", 0)),
                status=str(plan.get("status", "active")),
            )
        else:
            self.plan = None
        self.history = list(payload.get("history", []))
        self.stats = {**self.stats, **dict(payload.get("stats", {}))}
