"""Unified sensor protocol (MASTER SPEC 3.1).

Every channel, simulated or real, returns a :class:`SensorReading` with an
explicit unit, coordinate frame, confidence, noise level, latency, validity and
provenance. Missing data is represented as ``is_valid=False`` plus a reason; it
is never silently replaced by a zero.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Source(str, Enum):
    SIMULATED = "SIMULATED"
    LOCAL_DEVICE = "LOCAL_DEVICE"
    USER_PROVIDED = "USER_PROVIDED"


class SensorStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class SensorReading:
    sensor_id: str
    modality: str
    timestamp: float
    values: tuple[float, ...]
    units: str = ""
    coordinate_frame: str = ""
    confidence: float = 1.0
    noise_level: float = 0.0
    latency_ms: float = 0.0
    is_valid: bool = True
    missing_reason: str | None = None
    source: str = Source.SIMULATED.value
    provenance: str = ""
    permission_scope: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "values": list(self.values),
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "confidence": self.confidence,
            "noise_level": self.noise_level,
            "latency_ms": self.latency_ms,
            "is_valid": self.is_valid,
            "missing_reason": self.missing_reason,
            "source": self.source,
            "provenance": self.provenance,
            "permission_scope": self.permission_scope,
        }


@dataclass
class SensorHealth:
    sensor_id: str
    status: str
    detail: str = ""
    samples: int = 0
    invalid: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "status": self.status,
            "detail": self.detail,
            "samples": self.samples,
            "invalid": self.invalid,
        }


@dataclass
class SensorConfig:
    sensor_id: str
    modality: str
    enabled: bool = True
    noise: float = 0.0
    latency_steps: int = 0
    dropout: float = 0.0
    drift: float = 0.0
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, sensor_id: str, modality: str, data: dict[str, Any] | None) -> "SensorConfig":
        data = dict(data or {})
        return cls(
            sensor_id=sensor_id,
            modality=modality,
            enabled=bool(data.get("enabled", True)),
            noise=float(data.get("noise", 0.0)),
            latency_steps=int(data.get("latency_steps", 0)),
            dropout=float(data.get("dropout", 0.0)),
            drift=float(data.get("drift", 0.0)),
            confidence=float(data.get("confidence", 1.0)),
            extra={
                key: value
                for key, value in data.items()
                if key not in {"enabled", "noise", "latency_steps", "dropout", "drift", "confidence"}
            },
        )


class Sensor(ABC):
    """Base class implementing the fault/noise/latency pipeline."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self.samples = 0
        self.invalid = 0
        self.started = False
        self._latency_buffer: list[tuple[float, ...]] = []
        self._drift = 0.0

    # ------------------------------------------------------------- interface
    @property
    def sensor_id(self) -> str:
        return self.config.sensor_id

    @property
    def modality(self) -> str:
        return self.config.modality

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def configure(self, config: SensorConfig) -> None:
        self.config = config
        self._latency_buffer.clear()

    def calibrate(self) -> None:
        self._drift = 0.0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def health_check(self) -> SensorHealth:
        if not self.enabled:
            return SensorHealth(self.sensor_id, SensorStatus.DISABLED.value)
        if not self.started:
            return SensorHealth(self.sensor_id, SensorStatus.FAILED.value, "not started")
        if self.invalid and self.samples and self.invalid / self.samples > 0.5:
            return SensorHealth(
                self.sensor_id, SensorStatus.DEGRADED.value,
                f"{self.invalid}/{self.samples} invalid readings", self.samples, self.invalid,
            )
        return SensorHealth(self.sensor_id, SensorStatus.OK.value, "", self.samples, self.invalid)

    # --------------------------------------------------------------- sampling
    @abstractmethod
    def _raw(self, world: Any, now: float) -> tuple[float, ...]: ...

    def sample(self, world: Any, now: float, rng) -> SensorReading:
        if not self.enabled or not self.started:
            return self._invalid(now, "disabled")
        self.samples += 1
        values = tuple(float(v) for v in self._raw(world, now))

        # latency: return the value observed `latency_steps` samples ago
        latency_steps = max(0, int(self.config.latency_steps))
        if latency_steps:
            self._latency_buffer.append(values)
            if len(self._latency_buffer) > latency_steps:
                values = self._latency_buffer.pop(0)
            else:
                values = tuple(0.0 for _ in values)

        # drift
        if self.config.drift:
            self._drift += float(rng.normal(0.0, self.config.drift))
            values = tuple(v + self._drift for v in values)

        # noise
        if self.config.noise:
            values = tuple(v + float(rng.normal(0.0, self.config.noise)) for v in values)

        # dropout
        if self.config.dropout and float(rng.random()) < self.config.dropout:
            return self._invalid(now, "dropout")

        return SensorReading(
            sensor_id=self.sensor_id,
            modality=self.modality,
            timestamp=float(now),
            values=values,
            units=str(self.config.extra.get("units", "")),
            coordinate_frame=str(self.config.extra.get("frame", "")),
            confidence=float(self.config.confidence),
            noise_level=float(self.config.noise),
            latency_ms=float(latency_steps) * 1000.0,
            is_valid=True,
            source=Source.SIMULATED.value,
            provenance=str(self.config.extra.get("provenance", f"sim.{self.sensor_id}")),
        )

    def _invalid(self, now: float, reason: str) -> SensorReading:
        self.invalid += 1
        return SensorReading(
            sensor_id=self.sensor_id,
            modality=self.modality,
            timestamp=float(now),
            values=(),
            confidence=0.0,
            is_valid=False,
            missing_reason=reason,
            source=Source.SIMULATED.value,
            provenance=f"sim.{self.sensor_id}",
        )
