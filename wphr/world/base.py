"""World interface.

Two rules are enforced at this level and re-checked in tests:

1. ``observe()`` returns *measurements*, never the simulator's ground truth. The
   subject only ever sees :class:`WorldObservation`.
2. ``ground_truth()`` is a separate, explicitly named method that the cognitive
   loop must not call; it exists for evaluation and replay only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class WorldObservation:
    modality: str
    values: list[float]
    energy: float
    step: int
    source_device: str = "simulator"
    confidence: float = 1.0
    valid: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "energy": self.energy,
            "meta": dict(self.meta),
            "valid": self.valid,
        }


@dataclass
class WorldResult:
    action: str
    success: bool
    reward: float
    energy_after: float
    reached: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "success": self.success,
            "reward": self.reward,
            "energy_after": self.energy_after,
            "reached": self.reached,
            "info": dict(self.info),
        }


class World(Protocol):
    world_id: str

    def reset(self, seed: int | None = None) -> None: ...

    def observe(self) -> WorldObservation: ...

    def execute(self, action: str) -> WorldResult: ...

    def actions(self) -> tuple[str, ...]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, data: dict[str, Any]) -> None: ...
