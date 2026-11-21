"""M4 Temporal/History: working memory, event segmentation, rate estimates."""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np

from wphr.core.event import Event, EventType
from wphr.core.module import BaseModule


class TemporalModule(BaseModule):
    name: ClassVar[str] = "temporal"
    description: ClassVar[str] = "working memory buffer, event boundaries, temporal statistics"
    group: ClassVar[str] = "fast"

    def reset(self) -> None:
        capacity = int(self.ctx.config.get("memory.working_capacity", 32))
        self._state: dict[str, Any] = {"buffer": [], "capacity": capacity, "boundaries": 0}
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    # ------------------------------------------------------------------ event
    def on_event(self, event: Event) -> None:
        super().on_event(event)
        entry = {
            "step": int(event.step),
            "type": event.event_type,
            "modality": event.modality,
            "error": float(event.prediction_error or 0.0),
        }
        self._buffer.append(entry)
        self._state["buffer"] = list(self._buffer)

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        buffer = list(self._buffer)
        if len(buffer) >= 2:
            span = max(1, buffer[-1]["step"] - buffer[0]["step"])
            events_per_step = len(buffer) / span
            errors = [entry["error"] for entry in buffer if entry["error"]]
            error_rate = float(np.mean(errors)) if errors else 0.0
        else:
            events_per_step = 0.0
            error_rate = 0.0
        self._state["events_per_step"] = events_per_step
        self._state["error_rate"] = error_rate
        self.ctx.set_shared("temporal.events_per_step", events_per_step, source=self.name)
        self.ctx.set_shared("temporal.error_rate", error_rate, source=self.name)

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(self._buffer)[-limit:]

    def restore(self, payload: dict[str, Any]) -> None:
        super().restore(payload)
        capacity = int(self._state.get("capacity", self._buffer.maxlen or 32))
        self._buffer = deque(self._state.get("buffer", []), maxlen=capacity)

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "buffer_len": len(self._buffer),
                "capacity": int(self._state["capacity"]),
                "events_per_step": round(float(self._state.get("events_per_step", 0.0)), 4),
                "error_rate": round(float(self._state.get("error_rate", 0.0)), 6),
            }
        )
        return base
