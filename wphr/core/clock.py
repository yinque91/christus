"""Simulation clock: step counter, simulation time and wall-clock accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Clock:
    dt: float
    step: int = 0
    sim_time: float = 0.0
    wall_started: float = 0.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("clock dt must be positive")
        if not self.wall_started:
            self.wall_started = time.time()

    def tick(self) -> int:
        self.step += 1
        self.sim_time += self.dt
        return self.step

    @property
    def wall_elapsed(self) -> float:
        return time.time() - self.wall_started

    def reset(self, step: int = 0) -> None:
        self.step = step
        self.sim_time = step * self.dt
        self.wall_started = time.time()

    def snapshot(self) -> dict[str, float | int]:
        return {"step": self.step, "sim_time": self.sim_time, "dt": self.dt}
