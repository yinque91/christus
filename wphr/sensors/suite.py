"""Sensor suite: construction, fault injection, health and coverage reporting."""

from __future__ import annotations

from typing import Any

import numpy as np

from wphr.sensors.base import Sensor, SensorConfig, SensorHealth, SensorReading, SensorStatus
from wphr.sensors.simulated import SENSOR_CLASSES

FAULT_KINDS = ("dropout", "noise", "stale", "drift", "fail")


class SensorSuite:
    """Owns one instance of every configured sensor and applies faults on top."""

    def __init__(self, config_section: dict[str, Any], rng: np.random.Generator) -> None:
        self.rng = rng
        self.enabled = bool(config_section.get("enabled", True))
        self.sensors: dict[str, Sensor] = {}
        self.faults: dict[str, dict[str, Any]] = {}
        self.reading_count = 0
        self.invalid_count = 0
        for name, (cls, _modality) in SENSOR_CLASSES.items():
            entry = config_section.get(name, {}) or {}
            config = SensorConfig.from_dict(name, entry.get("modality", _modality), entry)
            self.sensors[name] = cls(config)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        for sensor in self.sensors.values():
            sensor.start()

    def stop(self) -> None:
        for sensor in self.sensors.values():
            sensor.stop()

    def calibrate(self) -> None:
        for sensor in self.sensors.values():
            sensor.calibrate()

    # --------------------------------------------------------------- sampling
    def sample_all(self, world: Any, now: float) -> dict[str, SensorReading]:
        readings: dict[str, SensorReading] = {}
        if not self.enabled:
            return readings
        for sensor_id, sensor in self.sensors.items():
            reading = sensor.sample(world, now, self.rng)
            fault = self.faults.get(sensor_id)
            if fault:
                reading = self._apply_fault(reading, fault)
            readings[sensor_id] = reading
        self.reading_count += len(readings)
        self.invalid_count += sum(1 for r in readings.values() if not r.is_valid)
        readings["uncertainty"] = self._uncertainty_reading(readings, now)
        return readings

    def _uncertainty_reading(
        self, readings: dict[str, SensorReading], now: float
    ) -> SensorReading:
        total = len(readings) or 1
        missing = sum(1 for r in readings.values() if not r.is_valid)
        missing_rate = missing / total
        noise = float(np.mean([r.noise_level for r in readings.values()])) if readings else 0.0
        confidence = 1.0 - missing_rate
        return SensorReading(
            sensor_id="uncertainty",
            modality="uncertainty",
            timestamp=float(now),
            values=(missing_rate, noise, confidence),
            units="ratio",
            coordinate_frame="suite",
            confidence=confidence,
            noise_level=noise,
            latency_ms=0.0,
            is_valid=True,
            source="SIMULATED",
            provenance="suite.uncertainty_probe",
        )

    def _apply_fault(self, reading: SensorReading, fault: dict[str, Any]) -> SensorReading:
        kind = fault["kind"]
        magnitude = float(fault.get("magnitude", 0.0))
        if kind in ("dropout", "fail"):
            return SensorReading(
                sensor_id=reading.sensor_id,
                modality=reading.modality,
                timestamp=reading.timestamp,
                values=(),
                confidence=0.0,
                is_valid=False,
                missing_reason=kind,
                source=reading.source,
                provenance=f"fault.{kind}",
            )
        if kind == "stale":
            return SensorReading(
                sensor_id=reading.sensor_id,
                modality=reading.modality,
                timestamp=reading.timestamp,
                values=tuple(0.0 for _ in reading.values),
                confidence=reading.confidence * 0.5,
                is_valid=reading.is_valid,
                missing_reason="stale",
                source=reading.source,
                provenance="fault.stale",
            )
        if kind == "noise":
            return SensorReading(
                sensor_id=reading.sensor_id,
                modality=reading.modality,
                timestamp=reading.timestamp,
                values=tuple(
                    v + float(self.rng.normal(0.0, magnitude)) for v in reading.values
                ),
                units=reading.units,
                coordinate_frame=reading.coordinate_frame,
                confidence=reading.confidence * 0.7,
                noise_level=reading.noise_level + magnitude,
                is_valid=reading.is_valid,
                source=reading.source,
                provenance="fault.noise",
            )
        if kind == "drift":
            return SensorReading(
                sensor_id=reading.sensor_id,
                modality=reading.modality,
                timestamp=reading.timestamp,
                values=tuple(v + magnitude for v in reading.values),
                units=reading.units,
                coordinate_frame=reading.coordinate_frame,
                confidence=reading.confidence * 0.8,
                is_valid=reading.is_valid,
                source=reading.source,
                provenance="fault.drift",
            )
        raise ValueError(f"unknown fault kind: {kind!r}")

    # ----------------------------------------------------------------- faults
    def inject_fault(self, sensor_id: str, kind: str, magnitude: float = 0.0) -> None:
        if sensor_id not in self.sensors:
            raise KeyError(f"unknown sensor: {sensor_id}")
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {kind!r}; use one of {FAULT_KINDS}")
        self.faults[sensor_id] = {"kind": kind, "magnitude": float(magnitude)}

    def clear_faults(self) -> None:
        self.faults.clear()

    # ----------------------------------------------------------------- health
    def health(self) -> list[SensorHealth]:
        reports = [sensor.health_check() for sensor in self.sensors.values()]
        for sensor_id, fault in self.faults.items():
            for report in reports:
                if report.sensor_id == sensor_id:
                    report.status = SensorStatus.FAILED.value
                    report.detail = f"injected fault: {fault['kind']}"
        return reports

    def overall_status(self) -> str:
        statuses = [report.status for report in self.health()]
        if SensorStatus.FAILED.value in statuses:
            return SensorStatus.FAILED.value
        if SensorStatus.DEGRADED.value in statuses:
            return SensorStatus.DEGRADED.value
        return SensorStatus.OK.value

    def coverage(self) -> dict[str, list[str]]:
        """modality -> sensor ids, for the V1.0 coverage checklist."""
        out: dict[str, list[str]] = {}
        for sensor in self.sensors.values():
            out.setdefault(sensor.modality, []).append(sensor.sensor_id)
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "faults": dict(self.faults),
            "samples": {s.sensor_id: s.samples for s in self.sensors.values()},
            "invalid": {s.sensor_id: s.invalid for s in self.sensors.values()},
            "readings": self.reading_count,
            "invalid_readings": self.invalid_count,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.faults = dict(payload.get("faults", {}))
        samples = payload.get("samples", {})
        invalid = payload.get("invalid", {})
        for sensor in self.sensors.values():
            sensor.samples = int(samples.get(sensor.sensor_id, 0))
            sensor.invalid = int(invalid.get(sensor.sensor_id, 0))
        self.reading_count = int(payload.get("readings", 0))
        self.invalid_count = int(payload.get("invalid_readings", 0))
