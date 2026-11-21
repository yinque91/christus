"""Demo world: a small continuous 2-D space with an energy budget.

This is the V0.1 carrier for the closed perception/action loop. It is
deliberately simple. V0.2 replaces it with a grid world containing rooms, doors,
tools, hazards and other agents.

The subject only sees :class:`WorldObservation`; the resource position is
ground truth and is only exposed by :meth:`DemoWorld.ground_truth`, which the
cognitive loop never calls.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from wphr.world.base import WorldObservation, WorldResult


@dataclass
class DemoWorldConfig:
    size: float = 5.0
    move_step: float = 0.5
    resource_radius: float = 0.9
    energy_start: float = 1.0
    energy_cost_per_step: float = 0.0025
    recharge: float = 0.35
    relocate_on_reach: bool = True
    observation_noise: float = 0.02
    world_id: str = "demo"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DemoWorldConfig":
        fields = {f for f in cls.__dataclass_fields__ if f != "world_id"}
        return cls(**{k: v for k, v in data.items() if k in fields})


MOVES: dict[str, tuple[float, float]] = {
    "move_north": (0.0, 1.0),
    "move_east": (1.0, 0.0),
    "move_south": (0.0, -1.0),
    "move_west": (-1.0, 0.0),
}


class DemoWorld:
    world_id = "demo"

    def __init__(self, config: DemoWorldConfig | None = None, seed: int = 0) -> None:
        self.config = config or DemoWorldConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.x = 0.0
        self.y = 0.0
        self.energy = float(self.config.energy_start)
        self.resource = np.zeros(2, dtype=np.float64)
        self.step = 0
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.x = 0.0
        self.y = 0.0
        self.energy = float(self.config.energy_start)
        self.step = 0
        self._place_resource()

    def _place_resource(self) -> None:
        limit = max(1e-9, self.config.size - 1.0)
        self.resource = self.rng.uniform(-limit, limit, size=2)

    def actions(self) -> tuple[str, ...]:
        return ("move_north", "move_east", "move_south", "move_west", "wait", "observe")

    # ------------------------------------------------------------ observation
    def observe(self) -> WorldObservation:
        dx = float(self.resource[0] - self.x)
        dy = float(self.resource[1] - self.y)
        noise = float(self.config.observation_noise)
        if noise:
            dx += float(self.rng.normal(0.0, noise))
            dy += float(self.rng.normal(0.0, noise))
        phase = 2.0 * math.pi * (self.step % 50) / 50.0
        size = max(1e-9, self.config.size)
        values = [
            float(self.x) / size,
            float(self.y) / size,
            float(self.energy),
            dx / size,
            dy / size,
            math.sin(phase),
        ]
        return WorldObservation(
            modality="multimodal",
            values=values,
            energy=float(self.energy),
            step=self.step,
            source_device="simulator",
            confidence=1.0,
            meta={"world": self.world_id, "sensor_set": ["proprioception", "vision.distance"]},
        )

    # ---------------------------------------------------------------- actions
    def execute(self, action: str) -> WorldResult:
        if action not in self.actions():
            raise ValueError(f"invalid action for demo world: {action!r}")
        if action in MOVES:
            ux, uy = MOVES[action]
            self.x = float(np.clip(self.x + ux * self.config.move_step, -self.config.size, self.config.size))
            self.y = float(np.clip(self.y + uy * self.config.move_step, -self.config.size, self.config.size))
        self.step += 1
        self.energy = float(max(0.0, self.energy - self.config.energy_cost_per_step))

        distance = float(math.hypot(self.resource[0] - self.x, self.resource[1] - self.y))
        reached = distance <= self.config.resource_radius
        reward = -0.01
        if reached:
            reward = 1.0
            self.energy = float(min(1.0, self.energy + self.config.recharge))
            if self.config.relocate_on_reach:
                self._place_resource()
        return WorldResult(
            action=action,
            success=True,
            reward=reward,
            energy_after=float(self.energy),
            reached=reached,
            info={"distance": round(distance, 6), "step": self.step},
        )

    # ------------------------------------------------------------- hidden truth
    def ground_truth(self) -> dict[str, Any]:
        """Simulator internals for evaluation only; the cognitive loop must not call this."""
        return {
            "resource": self.resource.tolist(),
            "body": [self.x, self.y],
            "energy": self.energy,
            "step": self.step,
            "world": self.world_id,
        }

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "config": self.config.__dict__,
            "seed": self.seed,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "resource": self.resource.tolist(),
            "step": self.step,
            "rng_state": json.dumps(self.rng.bit_generator.state, sort_keys=True),
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.config = DemoWorldConfig.from_dict(data.get("config", {}))
        self.seed = int(data.get("seed", 0))
        self.x = float(data.get("x", 0.0))
        self.y = float(data.get("y", 0.0))
        self.energy = float(data.get("energy", self.config.energy_start))
        self.resource = np.asarray(data.get("resource", [0.0, 0.0]), dtype=np.float64)
        self.step = int(data.get("step", 0))
        state = data.get("rng_state")
        self.rng = np.random.default_rng(self.seed)
        if state:
            self.rng.bit_generator.state = json.loads(state)
