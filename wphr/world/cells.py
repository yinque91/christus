"""Cell / object vocabulary shared by the grid world and the sensors."""

from __future__ import annotations

from enum import IntEnum


class Cell(IntEnum):
    WALL = 0
    FLOOR = 1
    FOOD = 2
    HAZARD = 3
    SWITCH_BLUE = 4
    SWITCH_ORANGE = 5
    TOOL = 6


CELL_NAMES: tuple[str, ...] = tuple(cell.name.lower() for cell in Cell)
N_CELL_TYPES = len(CELL_NAMES)

#: cells the agent cannot walk into
SOLID: frozenset[int] = frozenset({int(Cell.WALL)})

#: cells that can be interacted with
INTERACTABLE: frozenset[int] = frozenset(
    {
        int(Cell.FOOD),
        int(Cell.HAZARD),
        int(Cell.SWITCH_BLUE),
        int(Cell.SWITCH_ORANGE),
        int(Cell.TOOL),
    }
)

#: the hidden rule encodes which switch colour is currently active
SWITCH_COLORS: dict[int, str] = {
    int(Cell.SWITCH_BLUE): "blue",
    int(Cell.SWITCH_ORANGE): "orange",
}
COLOR_TO_CELL: dict[str, int] = {color: cell for cell, color in SWITCH_COLORS.items()}
