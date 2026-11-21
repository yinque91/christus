"""Feature assembly: percept + internal state -> per-module core input.

The percept is built from *sensor readings* (or, for the legacy demo world, from
``World.observe``). Every module declares the feature names it consumes; the
config declares how many inputs it has. A module may declare fewer features than
its ``input_dim`` (the remainder is zero-padded), but never more — an over-long
list is a hard error rather than a silent truncation, because silent truncation
would quietly change what the model sees.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from wphr.cognition.smctrn import SparseModularCTRN
from wphr.world.cells import CELL_NAMES, N_CELL_TYPES

#: index inside a 3x3 axis-aligned patch of the cell the body faces
FACED_PATCH_INDEX = {0: 1, 1: 5, 2: 7, 3: 3}  # north, east, south, west

FEATURE_NAMES: dict[str, list[str]] = {
    "world": [
        "x", "y", "energy", "heading_sin", "heading_cos",
        "depth", "patch_food", "patch_hazard", "patch_wall", "bias",
    ],
    "practice": [
        "a0", "a1", "a2", "a3", "a4", "a5",
        "cue_food", "cue_blue", "cue_orange", "cue_hazard", "cue_wall",
        "energy", "last_reward", "epsilon", "value_mean", "bias",
    ],
    "affect": [
        "frustration", "curiosity", "fatigue", "stress", "relief", "energy", "deficit", "bias",
    ],
    "temporal": [
        "step_phase", "err_ema", "events_per_step", "episode_phase", "fatigue", "bias",
    ],
    "concept_language": [
        "x", "y", "energy", "cue_food", "cue_blue", "cue_orange", "cue_hazard",
        "patch_wall", "concept_norm", "heading_sin", "heading_cos", "err_ema", "bias",
    ],
    "social": [
        "agent_count", "trust_mean", "agent_near", "agent_dx", "agent_dy",
        "predicted_dx", "social_motivation", "bias",
    ],
    "volition": [
        "goal_value", "info_gain", "cost", "risk", "energy", "curiosity", "bias", "fatigue",
    ],
}


def heading_index(sin_value: float, cos_value: float) -> int:
    angle = math.atan2(sin_value, cos_value)
    index = int(round(angle / (math.pi / 2.0))) % 4
    return index


def build_percept(
    *,
    observation_values: list[float],
    energy: float,
    readings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compress raw sensor readings into named derived features."""
    percept: dict[str, Any] = {
        "x": float(observation_values[0]) if len(observation_values) > 0 else 0.0,
        "y": float(observation_values[1]) if len(observation_values) > 1 else 0.0,
        "energy": float(energy),
        "heading_sin": 0.0,
        "heading_cos": 1.0,
        "heading": 0,
        "cue": "none",
        "patch": [0.0] * N_CELL_TYPES,
        "depth": 0.0,
        "scent": (0.0, 0.0),
        "loudness": 0.0,
        "valid_ratio": 1.0,
        "missing": [],
        "agent": None,
    }
    if not readings:
        # legacy/demo path: residual direction doubles as a scent-like cue
        if len(observation_values) >= 5:
            percept["scent"] = (float(observation_values[3]), float(observation_values[4]))
        return percept

    valid = [r for r in readings.values() if r.is_valid]
    percept["valid_ratio"] = len(valid) / max(1, len(readings))
    percept["missing"] = [r.sensor_id for r in readings.values() if not r.is_valid]

    vision = readings.get("vision")
    if vision is not None and vision.is_valid and vision.values:
        values = np.asarray(vision.values, dtype=np.float64)
        if values.size == 9 * N_CELL_TYPES:
            grid = values.reshape(9, N_CELL_TYPES)
            percept["patch"] = grid.sum(axis=0).tolist()
            proprio = readings.get("proprioception")
            heading = 1
            if proprio is not None and proprio.is_valid and len(proprio.values) >= 4:
                heading = heading_index(proprio.values[2], proprio.values[3])
            percept["heading"] = heading
            faced = int(np.argmax(grid[FACED_PATCH_INDEX[heading]]))
            percept["cue"] = CELL_NAMES[faced] if 0 <= faced < len(CELL_NAMES) else "none"

    proprio = readings.get("proprioception")
    if proprio is not None and proprio.is_valid and len(proprio.values) >= 4:
        percept["heading_sin"] = float(proprio.values[2])
        percept["heading_cos"] = float(proprio.values[3])
    depth = readings.get("depth")
    if depth is not None and depth.is_valid and depth.values:
        percept["depth"] = float(depth.values[0])
    olfaction = readings.get("olfaction")
    if olfaction is not None and olfaction.is_valid and len(olfaction.values) >= 2:
        percept["scent"] = (float(olfaction.values[0]), float(olfaction.values[1]))
    audio = readings.get("audio")
    if audio is not None and audio.is_valid and audio.values:
        percept["loudness"] = float(audio.values[0])
    social = readings.get("social")
    if social is not None and social.is_valid and len(social.values) >= 6:
        count = float(social.values[0])
        distance = float(social.values[1]) * 4.0
        if count > 0 and distance > 0:
            percept["agent"] = {
                "count": count,
                "distance": distance,
                "dx": float(social.values[2]) * distance,
                "dy": float(social.values[3]) * distance,
                "last_move": (float(social.values[4]), float(social.values[5])),
            }
    return percept


def percept_to_features(
    percept: dict[str, Any],
    *,
    affect: dict[str, float],
    practice: dict[str, Any],
    temporal: dict[str, Any],
    concept: dict[str, Any],
    volition: dict[str, Any],
    error_ema: float,
    step: int,
    episode_index: int = 0,
) -> dict[str, float]:
    patch = percept.get("patch") or [0.0] * N_CELL_TYPES
    cue = str(percept.get("cue", "none"))
    phase = 2.0 * math.pi * (step % 50) / 50.0
    onehot = [0.0] * 6
    action_index = practice.get("action_index", 0)
    if isinstance(action_index, int) and 0 <= action_index < 6:
        onehot[action_index] = 1.0

    def patch_of(name: str) -> float:
        try:
            return float(patch[CELL_NAMES.index(name)])
        except ValueError:
            return 0.0

    energy = float(percept.get("energy", 0.0))
    scent = percept.get("scent", (0.0, 0.0)) or (0.0, 0.0)
    features = {
        "x": float(percept.get("x", 0.0)),
        "y": float(percept.get("y", 0.0)),
        "energy": energy,
        "deficit": 1.0 - energy,
        # The existing WORLD02 adapter publishes the ball offset as ``scent``.
        # These names are intentionally not added to FEATURE_NAMES: only a config
        # with core.extra_features asks for them, so the default core input and
        # old checkpoints are unchanged.
        "scent_x": float(scent[0]),
        "scent_y": float(scent[1]),
        "heading_sin": float(percept.get("heading_sin", 0.0)),
        "heading_cos": float(percept.get("heading_cos", 1.0)),
        "depth": float(percept.get("depth", 0.0)) / 8.0,
        "sin_phase": float(math.sin(phase)),
        "cos_phase": float(math.cos(phase)),
        "step_phase": float(math.sin(phase)),
        "step": float(step),
        "bias": 1.0,
        "a0": onehot[0], "a1": onehot[1], "a2": onehot[2],
        "a3": onehot[3], "a4": onehot[4], "a5": onehot[5],
        "last_reward": float(practice.get("last_reward", 0.0)),
        "epsilon": float(practice.get("epsilon", 0.25)),
        "value_mean": float(practice.get("value_mean", 0.0)),
        "err_ema": float(error_ema),
        "frustration": float(affect.get("frustration", 0.0)),
        "curiosity": float(affect.get("curiosity", 0.0)),
        "fatigue": float(affect.get("fatigue", 0.0)),
        "stress": float(affect.get("stress_load", 0.0)),
        "relief": float(affect.get("relief", 0.0)),
        "risk_sensitivity": float(affect.get("risk_sensitivity", 0.0)),
        "social_motivation": float(affect.get("social_motivation", 0.0)),
        "events_per_step": float(temporal.get("events_per_step", 0.0)),
        "episode_phase": float(episode_index % 10) / 10.0,
        "concept_norm": float(concept.get("norm", 0.0)),
        "agent_count": float(concept.get("agent_count", 0)),
        "trust_mean": float(practice.get("trust_mean", 0.5)),
        "goal_value": float(volition.get("value", 0.0)),
        "info_gain": float(volition.get("info", 0.0)),
        "cost": float(volition.get("cost", 0.0)),
        "risk": float(volition.get("risk", 0.0)),
        "loudness": float(percept.get("loudness", 0.0)),
        "valid_ratio": float(percept.get("valid_ratio", 1.0)),
    }
    agent = percept.get("agent") or {}
    features["agent_near"] = 1.0 if agent else 0.0
    features["agent_dx"] = float(agent.get("dx", 0.0))
    features["agent_dy"] = float(agent.get("dy", 0.0))
    features["predicted_dx"] = float(practice.get("predicted_dx", 0.0))
    features["agent_count"] = float(agent.get("count", 0.0))
    for name in CELL_NAMES:
        features[f"patch_{name}"] = patch_of(name)
        features[f"cue_{name}"] = 1.0 if cue == name else 0.0
    # short aliases used in FEATURE_NAMES
    features["cue_blue"] = features.get("cue_switch_blue", 0.0)
    features["cue_orange"] = features.get("cue_switch_orange", 0.0)
    return features


def assemble_inputs(
    core: SparseModularCTRN, features: dict[str, float], *, clip: float = 5.0
) -> np.ndarray:
    flat = np.zeros(core.total_inputs, dtype=np.float64)
    extra = core.cfg.get("extra_features", {}) or {}
    for module_name in core.module_names:
        names = list(FEATURE_NAMES.get(module_name, [])) + list(extra.get(module_name, []))
        start, end = core.input_offsets[module_name]
        dim = end - start
        if len(names) > dim:
            raise ValueError(
                f"module {module_name} declares {len(names)} features but input_dim is {dim}"
            )
        vec = [float(features.get(name, 0.0)) for name in names]
        vec.extend([0.0] * (dim - len(vec)))
        flat[start:end] = np.clip(np.asarray(vec, dtype=np.float64), -clip, clip)
    return flat
