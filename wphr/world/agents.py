"""Simulated other agents for the social stage (MASTER SPEC 9.1).

An agent is a simple scripted policy, deliberately *not* another WPHR subject:
the point of V0.7 is whether the subject can model an observable other from its
behaviour alone, so the other must be cheap and stable.

Each agent has a hidden disposition (probability of cooperating when
interacted with). The subject never reads it; it can only accumulate evidence
from realised interactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wphr.world.cells import SOLID

MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "move_north": (0, -1),
    "move_east": (1, 0),
    "move_south": (0, 1),
    "move_west": (-1, 0),
}
ALL_MOVES: tuple[str, ...] = tuple(MOVE_VECTORS)


@dataclass
class ScriptedAgent:
    agent_id: str
    x: int
    y: int
    heading: str = "east"
    last_move: str = "wait"
    persistence: float = 0.80
    cooperation: float = 0.70
    wait_probability: float = 0.05
    seed: int = 0
    history: list[str] = field(default_factory=list)
    _rng: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed + 7919)

    # ------------------------------------------------------------------ policy
    def step(self, world: Any) -> str:
        if float(self._rng.random()) < self.wait_probability:
            self.last_move = "wait"
            return "wait"
        if self.last_move in MOVE_VECTORS and float(self._rng.random()) < self.persistence:
            candidates = [self.last_move]
        else:
            order = list(ALL_MOVES)
            self._rng.shuffle(order)
            candidates = order
        for move in candidates:
            dx, dy = MOVE_VECTORS[move]
            nx, ny = self.x + dx, self.y + dy
            if world.cell_at(nx, ny) in SOLID:
                continue
            self.x, self.y = nx, ny
            self.heading = move.split("_", 1)[1]
            self.last_move = move
            self.history.append(move)
            if len(self.history) > 64:
                del self.history[:-64]
            return move
        self.last_move = "wait"
        return "wait"

    # --------------------------------------------------------------- helpers
    def delta(self) -> tuple[int, int]:
        return MOVE_VECTORS.get(self.last_move, (0, 0))

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "last_move": self.last_move,
            "persistence": self.persistence,
            "cooperation": self.cooperation,
            "wait_probability": self.wait_probability,
            "seed": self.seed,
            "history": list(self.history),
            "rng_state": json.dumps(self._rng.bit_generator.state, sort_keys=True),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScriptedAgent":
        agent = cls(
            agent_id=str(data["agent_id"]),
            x=int(data["x"]),
            y=int(data["y"]),
            heading=str(data.get("heading", "east")),
            last_move=str(data.get("last_move", "wait")),
            persistence=float(data.get("persistence", 0.80)),
            cooperation=float(data.get("cooperation", 0.70)),
            wait_probability=float(data.get("wait_probability", 0.05)),
            seed=int(data.get("seed", 0)),
            history=list(data.get("history", [])),
        )
        state = data.get("rng_state")
        if state:
            agent._rng.bit_generator.state = json.loads(state)
        return agent


def spawn_agents(count: int, width: int, height: int, seed: int, cooperation: float = 0.7) -> list[ScriptedAgent]:
    """Place ``count`` agents on distinct interior cells."""
    rng = np.random.default_rng(seed + 104729)
    agents: list[ScriptedAgent] = []
    used: set[tuple[int, int]] = set()
    for index in range(max(0, count)):
        for _ in range(200):
            x = int(rng.integers(1, max(2, width - 1)))
            y = int(rng.integers(1, max(2, height - 1)))
            if (x, y) not in used:
                used.add((x, y))
                break
        agents.append(
            ScriptedAgent(
                agent_id=f"agent_{index}",
                x=x,
                y=y,
                seed=seed + 31 * index,
                cooperation=float(np.clip(cooperation, 0.0, 1.0)),
            )
        )
    return agents
