"""Inter-module shared blackboard with explicit provenance.

MASTER SPEC 5.1 requires "explicit module-to-module messages". The recurrent
core carries numerical messages between module state slices; this blackboard
carries symbolic ones (hypotheses, goals, affect reads) and records who wrote
each value, at which step, and with what confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedValue:
    value: Any
    source: str
    step: int
    confidence: float = 1.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "step": self.step,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedValue":
        return cls(
            value=data.get("value"),
            source=str(data.get("source", "")),
            step=int(data.get("step", 0)),
            confidence=float(data.get("confidence", 1.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass
class SharedState:
    """A key/value blackboard with per-entry provenance."""

    values: dict[str, SharedValue] = field(default_factory=dict)

    def set(
        self,
        key: str,
        value: Any,
        *,
        source: str,
        step: int,
        confidence: float = 1.0,
        now: float = 0.0,
    ) -> None:
        self.values[key] = SharedValue(
            value=value,
            source=source,
            step=step,
            confidence=float(confidence),
            updated_at=now,
        )

    def get(self, key: str, default: Any = None) -> Any:
        entry = self.values.get(key)
        return default if entry is None else entry.value

    def entry(self, key: str) -> SharedValue | None:
        return self.values.get(key)

    def provenance(self) -> dict[str, dict[str, Any]]:
        return {key: entry.to_dict() for key, entry in self.values.items()}

    def clear(self) -> None:
        self.values.clear()

    def snapshot(self) -> dict[str, Any]:
        return {key: entry.to_dict() for key, entry in self.values.items()}

    def restore(self, payload: dict[str, Any]) -> None:
        self.values = {key: SharedValue.from_dict(val) for key, val in payload.items()}
