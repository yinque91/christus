"""Simulated sensor channels.

Each class is a thin adapter over the world's *public* accessors. None of them
touch ``ground_truth()``; a test replaces that method with one that raises and
runs the full loop to prove it.
"""

from __future__ import annotations

import math
from typing import Any

from wphr.sensors.base import Sensor
from wphr.world.cells import CELL_NAMES, N_CELL_TYPES


class VisionSensor(Sensor):
    """3x3 one-hot patch; the centre-forward cell is the attended cue."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        radius = int(self.config.extra.get("radius", 1))
        patch = world.visible_cells(radius)
        out: list[float] = []
        for row in patch:
            for code in row:
                onehot = [0.0] * N_CELL_TYPES
                if 0 <= int(code) < N_CELL_TYPES:
                    onehot[int(code)] = 1.0
                out.extend(onehot)
        return tuple(out)


class DepthSensor(Sensor):
    """Distance (in cells) to the first solid cell straight ahead."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        distance = float(world.depth_ahead())
        return (distance, 1.0 / (1.0 + distance))


class AudioSensor(Sensor):
    """Loudness and bearing of the nearest hazard sound source."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        loudness, bearing, distance = world.sound_profile()
        return (float(loudness), float(math.sin(bearing)), float(math.cos(bearing)), float(distance))


class TouchSensor(Sensor):
    """Contact flag and normal pressure against the faced cell."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        solid, pressure = world.contact_at_face()
        return (1.0 if solid else 0.0, float(pressure))


class ProprioceptionSensor(Sensor):
    """Body pose and energy in normalised body coordinates."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        body = world.body
        width = max(1, int(world.width))
        height = max(1, int(world.height))
        angle = body.heading_angle()
        return (
            body.x / width,
            body.y / height,
            math.sin(angle),
            math.cos(angle),
            body.energy_ratio(),
            float(getattr(world, "step", 0)) / 1000.0,
        )


class VestibularSensor(Sensor):
    """Last displacement and accumulated damage risk (safe numeric proxy)."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        body = world.body
        dx, dy = body.last_delta
        return (float(dx), float(dy), float(math.hypot(dx, dy)), float(body.damage_risk))


class InteroceptionSensor(Sensor):
    """Resource channels: energy, deficit, load, integrity risk."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        body = world.body
        energy = body.energy_ratio()
        return (energy, 1.0 - energy, float(body.load), float(body.damage_risk))


class OlfactionSensor(Sensor):
    """Simulated scent field: direction and concentration of the nearest food."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        dx, dy, strength = world.scent_profile()
        return (float(dx), float(dy), float(strength))


class GustationSensor(Sensor):
    """Taste proxy: only meaningful when standing on a food cell."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        sweet, sour = world.taste_profile()
        return (float(sweet), float(sour))


class SpatialSensor(Sensor):
    """Normalised position and heading in world coordinates."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        body = world.body
        width = max(1, int(world.width))
        height = max(1, int(world.height))
        return (body.x / width, body.y / height, float(body.x), float(body.y))


class TemporalSensor(Sensor):
    """Phase and elapsed simulated time."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        phase = 2.0 * math.pi * (float(now) % 50.0) / 50.0
        return (math.sin(phase), math.cos(phase), float(now) % 1.0)


class SocialSensor(Sensor):
    """Nearest other agent: presence, distance, bearing and last observed move."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        agents = getattr(world, "agents", None) or ()
        visible = world.visible_agent(radius=3) if hasattr(world, "visible_agent") else None
        if not visible:
            return (float(len(agents)), 0.0, 0.0, 0.0, 0.0, 0.0)
        move_vector = {
            "move_north": (0.0, -1.0),
            "move_east": (1.0, 0.0),
            "move_south": (0.0, 1.0),
            "move_west": (-1.0, 0.0),
        }.get(str(visible.get("last_move", "wait")), (0.0, 0.0))
        return (
            float(len(agents)),
            float(visible["distance"]) / 4.0,
            float(math.sin(float(visible["bearing"]))),
            float(math.cos(float(visible["bearing"]))),
            float(move_vector[0]),
            float(move_vector[1]),
        )


class LanguageChannel(Sensor):
    """Text channel placeholder: no text is injected in V0.2."""

    def _raw(self, world: Any, now: float) -> tuple[float, ...]:
        return (0.0, 0.0)


SENSOR_CLASSES = {
    "vision": (VisionSensor, "vision"),
    "depth": (DepthSensor, "vision.depth"),
    "audio": (AudioSensor, "audio"),
    "touch": (TouchSensor, "touch"),
    "proprioception": (ProprioceptionSensor, "proprioception"),
    "vestibular": (VestibularSensor, "vestibular"),
    "interoception": (InteroceptionSensor, "interoception"),
    "olfaction": (OlfactionSensor, "olfaction"),
    "gustation": (GustationSensor, "gustation"),
    "spatial": (SpatialSensor, "spatial"),
    "temporal": (TemporalSensor, "temporal"),
    "social": (SocialSensor, "social"),
    "language": (LanguageChannel, "language"),
}

__all__ = ["SENSOR_CLASSES", "CELL_NAMES"] + [
    name for name, _ in SENSOR_CLASSES.values()
]
