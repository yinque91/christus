"""The unified event structure (MASTER SPEC 2.2).

Every piece of information the subject receives or produces is an Event with an
explicit source, time, confidence and provenance. Distinguishing
``observation`` from ``language_claim`` from ``ground_truth`` is a hard rule of
the specification: the cognitive core never receives simulator ground truth
through this channel unless the event is explicitly marked as such and the
consumer checks for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wphr.util.ids import new_id, utc_now
from wphr.version import SCHEMA_VERSION


class Source(str, Enum):
    """Where an event came from."""

    SIMULATED = "SIMULATED"
    LOCAL_DEVICE = "LOCAL_DEVICE"
    USER_PROVIDED = "USER_PROVIDED"


class PrivacyClass(str, Enum):
    """Privacy handling class; mirrors MASTER SPEC section 12."""

    PUBLIC_SIM = "PUBLIC_SIM"
    PROJECT_LOCAL = "PROJECT_LOCAL"
    USER_PRIVATE = "USER_PRIVATE"


class EventType(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    OUTCOME = "outcome"
    PREDICTION = "prediction"
    INTERNAL = "internal"
    LANGUAGE_CLAIM = "language_claim"
    SOCIAL = "social"
    GOAL = "goal"
    SYSTEM = "system"


_JSON_SCALARS = (str, int, float, bool, type(None))


def _json_safe(value: Any, *, where: str) -> Any:
    """Convert numpy scalars/arrays and nested containers into JSON-safe data."""
    if isinstance(value, _JSON_SCALARS):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, where=where) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, where=where) for v in value]
    if hasattr(value, "tolist"):  # numpy array / scalar
        return _json_safe(value.tolist(), where=where)
    raise TypeError(f"event field {where} contains a non-serialisable value: {type(value)!r}")


def _unit(name: str, value: Any) -> float:
    number = float(value)
    if not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {number}")
    return number


@dataclass(frozen=True)
class Event:
    """An immutable record of one observation, action, outcome or claim."""

    event_type: str
    step: int
    episode_id: str
    actor_id: str
    modality: str
    source: str = Source.SIMULATED.value
    source_device: str = "simulator"
    source_reliability: float = 1.0
    observation: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    expected_outcome: dict[str, Any] | None = None
    actual_outcome: dict[str, Any] | None = None
    prediction_error: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    affect_snapshot: dict[str, Any] = field(default_factory=dict)
    memory_refs: tuple[str, ...] = ()
    concept_refs: tuple[str, ...] = ()
    confidence: float = 1.0
    privacy_class: str = PrivacyClass.PUBLIC_SIM.value
    provenance: str = ""
    timestamp: float = field(default_factory=utc_now)
    event_id: str = field(default_factory=lambda: new_id("evt_"))
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ checks
    def __post_init__(self) -> None:
        if self.event_type not in {e.value for e in EventType}:
            raise ValueError(f"unknown event_type: {self.event_type!r}")
        if self.source not in {s.value for s in Source}:
            raise ValueError(f"unknown source: {self.source!r}")
        if self.privacy_class not in {p.value for p in PrivacyClass}:
            raise ValueError(f"unknown privacy_class: {self.privacy_class!r}")
        if self.step < 0:
            raise ValueError("step must be >= 0")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "source_reliability", _unit("source_reliability", self.source_reliability))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        if self.prediction_error is not None:
            pe = float(self.prediction_error)
            if not math.isfinite(pe):
                raise ValueError("prediction_error must be finite")
            object.__setattr__(self, "prediction_error", pe)
        for name in (
            "observation",
            "action",
            "expected_outcome",
            "actual_outcome",
            "context",
            "affect_snapshot",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _json_safe(value, where=name))
        object.__setattr__(self, "memory_refs", tuple(str(x) for x in self.memory_refs))
        object.__setattr__(self, "concept_refs", tuple(str(x) for x in self.concept_refs))
        if not self.event_id:
            raise ValueError("event_id must not be empty")

    # ------------------------------------------------------------ construction
    @classmethod
    def create(
        cls,
        event_type: str,
        *,
        step: int,
        episode_id: str,
        actor_id: str,
        modality: str,
        **kwargs: Any,
    ) -> "Event":
        return cls(
            event_type=event_type,
            step=step,
            episode_id=episode_id,
            actor_id=actor_id,
            modality=modality,
            **kwargs,
        )

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "step": self.step,
            "episode_id": self.episode_id,
            "actor_id": self.actor_id,
            "modality": self.modality,
            "source": self.source,
            "source_device": self.source_device,
            "source_reliability": self.source_reliability,
            "observation": self.observation,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "prediction_error": self.prediction_error,
            "context": self.context,
            "affect_snapshot": self.affect_snapshot,
            "memory_refs": list(self.memory_refs),
            "concept_refs": list(self.concept_refs),
            "confidence": self.confidence,
            "privacy_class": self.privacy_class,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        payload = dict(data)
        payload["memory_refs"] = tuple(payload.get("memory_refs") or ())
        payload["concept_refs"] = tuple(payload.get("concept_refs") or ())
        return cls(**payload)

    # ---------------------------------------------------------------- helpers
    @property
    def is_language(self) -> bool:
        return self.event_type == EventType.LANGUAGE_CLAIM.value

    @property
    def is_ground_truth(self) -> bool:
        return self.context.get("ground_truth") is True

    def assert_persistable(self) -> None:
        if self.privacy_class == PrivacyClass.USER_PRIVATE.value:
            raise PermissionError(
                "refusing to persist an event classified USER_PRIVATE without explicit consent"
            )

    def summary(self) -> str:
        bits = [f"#{self.step}", self.event_type, self.modality, self.source]
        if self.action:
            bits.append(f"action={self.action.get('name', '?')}")
        if self.prediction_error is not None:
            bits.append(f"err={self.prediction_error:.4f}")
        return " ".join(bits)
