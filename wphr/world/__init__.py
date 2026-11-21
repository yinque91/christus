"""World subsystem: simulated environments with hidden ground truth."""

from __future__ import annotations

from typing import Any

import numpy as np

from wphr.world.base import World, WorldObservation, WorldResult
from wphr.world.demo import DemoWorld, DemoWorldConfig
from wphr.world.grid import GridMap, GridWorld, generate_map, generate_map_set

WORLD_REGISTRY: dict[str, type] = {"demo": DemoWorld, "grid": GridWorld}


def create_world(
    name: str,
    *,
    seed: int,
    config: dict[str, Any] | None = None,
    grid_map: GridMap | None = None,
):
    """Build a world by id from configuration. Unknown ids are a hard error.

    The hidden rule (``safe_color``) is *not* taken from the subject config: it
    is either supplied by the experiment or drawn from the world seed here, so
    the subject's configuration cannot leak it.
    """
    section = (config or {}) or {}
    if name == "demo":
        return DemoWorld(DemoWorldConfig.from_dict(section), seed=seed)
    if name == "grid":
        if grid_map is not None:
            return GridWorld(grid_map, seed=seed, **_grid_kwargs(section))
        rng = np.random.default_rng(int(seed) + 104729)
        safe_color = "blue" if float(rng.random()) < 0.5 else "orange"
        grid_map = generate_map(
            rng=np.random.default_rng(int(seed)),
            width=int(section.get("width", 9)),
            height=int(section.get("height", 9)),
            wall_density=float(section.get("wall_density", 0.18)),
            food=int(section.get("food", 4)),
            hazards=int(section.get("hazards", 3)),
            switches_blue=int(section.get("switches_blue", 2)),
            switches_orange=int(section.get("switches_orange", 2)),
            tools=int(section.get("tools", 1)),
            safe_color=safe_color,
            map_id=f"grid_{seed}",
        )
        return GridWorld(grid_map, seed=seed, **_grid_kwargs(section))
    raise KeyError(f"unknown world: {name!r} (known: {sorted(WORLD_REGISTRY)})")


def _grid_kwargs(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "energy_start": float(section.get("energy_start", 1.0)),
        "energy_cost_per_step": float(section.get("energy_cost_per_step", 0.0015)),
        "move_cost": float(section.get("move_cost", 0.001)),
        "food_energy": float(section.get("food_energy", 0.35)),
        "hazard_penalty": float(section.get("hazard_penalty", 0.30)),
        "tool_reward": float(section.get("tool_reward", 0.5)),
        "max_steps": int(section.get("max_steps", 400)),
        "agents": int(section.get("agents", 0)),
        "agent_cooperation": float(section.get("agent_cooperation", 0.7)),
    }


__all__ = [
    "World",
    "WorldObservation",
    "WorldResult",
    "DemoWorld",
    "DemoWorldConfig",
    "GridWorld",
    "GridMap",
    "generate_map",
    "generate_map_set",
    "WORLD_REGISTRY",
    "create_world",
]
