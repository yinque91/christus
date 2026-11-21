"""Virtual body (MASTER SPEC 4.1).

The first-generation body is deliberately simple: a planar pose, a heading, an
energy budget, a load and a set of contact flags. The interface is written so
that a 3-D rigid body or a URDF-backed robot can replace it without touching the
cognitive core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

HEADINGS: tuple[str, ...] = ("north", "east", "south", "west")
HEADING_DELTA: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}


@dataclass
class VirtualBody:
    x: int = 0
    y: int = 0
    heading: str = "east"
    energy: float = 1.0
    load: float = 0.0
    radius: float = 0.4
    contact: tuple[str, ...] = ()
    damage_risk: float = 0.0
    last_delta: tuple[int, int] = (0, 0)
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ pose
    def turn_to(self, heading: str) -> None:
        if heading not in HEADINGS:
            raise ValueError(f"unknown heading: {heading!r}")
        self.heading = heading

    def delta(self) -> tuple[int, int]:
        return HEADING_DELTA[self.heading]

    def move_to(self, x: int, y: int) -> None:
        self.last_delta = (x - self.x, y - self.y)
        self.x = int(x)
        self.y = int(y)

    # ---------------------------------------------------------------- energy
    def spend(self, amount: float) -> None:
        self.energy = float(max(0.0, self.energy - max(0.0, amount)))

    def recharge(self, amount: float) -> None:
        self.energy = float(min(1.0, self.energy + max(0.0, amount)))

    def take_hit(self, damage: float) -> None:
        """A safe numeric penalty, not real pain."""
        self.damage_risk = float(min(1.0, self.damage_risk + damage))
        self.energy = float(max(0.0, self.energy - damage))

    # --------------------------------------------------------------- helpers
    def heading_angle(self) -> float:
        return HEADINGS.index(self.heading) * (math.pi / 2.0)

    def energy_ratio(self) -> float:
        return float(max(0.0, min(1.0, self.energy)))

    def snapshot(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "energy": self.energy,
            "load": self.load,
            "radius": self.radius,
            "contact": list(self.contact),
            "damage_risk": self.damage_risk,
            "last_delta": list(self.last_delta),
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.x = int(data.get("x", 0))
        self.y = int(data.get("y", 0))
        heading = str(data.get("heading", "east"))
        self.heading = heading if heading in HEADINGS else "east"
        self.energy = float(data.get("energy", 1.0))
        self.load = float(data.get("load", 0.0))
        self.radius = float(data.get("radius", 0.4))
        self.contact = tuple(data.get("contact", ()))
        self.damage_risk = float(data.get("damage_risk", 0.0))
        delta = data.get("last_delta", (0, 0))
        self.last_delta = (int(delta[0]), int(delta[1]))
