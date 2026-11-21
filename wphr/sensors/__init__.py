"""Unified sensor subsystem."""

from wphr.sensors.base import (
    Sensor,
    SensorConfig,
    SensorHealth,
    SensorReading,
    SensorStatus,
    Source,
)
from wphr.sensors.simulated import SENSOR_CLASSES
from wphr.sensors.suite import FAULT_KINDS, SensorSuite

__all__ = [
    "Sensor",
    "SensorConfig",
    "SensorHealth",
    "SensorReading",
    "SensorStatus",
    "Source",
    "SensorSuite",
    "SENSOR_CLASSES",
    "FAULT_KINDS",
]
