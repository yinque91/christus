"""Module base class and health reporting.

The seven logical modules of MASTER SPEC 2.1 share this interface: they receive
events, expose a snapshot for checkpointing, and report health. A module failure
degrades the system but does not crash the main loop; the failure is recorded
and surfaces in ``wphr status``.
"""

from __future__ import annotations

import copy
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from wphr.config import Config
from wphr.core.bus import EventBus
from wphr.core.clock import Clock
from wphr.core.event import Event
from wphr.core.state import SharedState
from wphr.logging_setup import AuditLog, get_logger


class ModuleStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass
class ModuleHealth:
    module: str
    status: str
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "status": self.status,
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass
class ModuleContext:
    """Everything a module may use; injected rather than imported globally."""

    config: Config
    clock: Clock
    bus: EventBus
    shared: SharedState
    rng: np.random.Generator
    audit: AuditLog | None = None
    memory: Any | None = None

    def emit(self, event: Event) -> None:
        self.bus.publish(event)

    def set_shared(self, key: str, value: Any, *, source: str, confidence: float = 1.0) -> None:
        self.shared.set(
            key, value, source=source, step=self.clock.step, confidence=confidence,
            now=self.clock.sim_time,
        )

    def get_shared(self, key: str, default: Any = None) -> Any:
        return self.shared.get(key, default)


class BaseModule(ABC):
    """Minimal interface every logical module implements."""

    name: ClassVar[str] = "module"
    description: ClassVar[str] = "base module"
    group: ClassVar[str] = "fast"

    def __init__(self, ctx: ModuleContext) -> None:
        self.ctx = ctx
        self.log = get_logger(f"modules.{self.name}")
        self.enabled = True
        self.steps = 0
        self.events_seen = 0
        self.last_error: str | None = None
        self._state: dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        self._state = {}

    def on_event(self, event: Event) -> None:
        self.events_seen += 1

    def update(self, step: int) -> None:
        self.steps += 1

    # ------------------------------------------------------------ checkpoint
    def snapshot(self) -> dict[str, Any]:
        return {
            "state": copy.deepcopy(self._state),
            "steps": self.steps,
            "events_seen": self.events_seen,
            "enabled": self.enabled,
            "last_error": self.last_error,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self._state = copy.deepcopy(payload.get("state", {}))
        self.steps = int(payload.get("steps", 0))
        self.events_seen = int(payload.get("events_seen", 0))
        self.enabled = bool(payload.get("enabled", True))
        self.last_error = payload.get("last_error")

    # ---------------------------------------------------------------- health
    def metrics(self) -> dict[str, Any]:
        return {"steps": self.steps, "events_seen": self.events_seen}

    def health(self) -> ModuleHealth:
        if not self.enabled:
            return ModuleHealth(self.name, ModuleStatus.DISABLED.value, "disabled by config")
        if self.last_error:
            return ModuleHealth(
                self.name, ModuleStatus.DEGRADED.value, self.last_error, self.metrics()
            )
        return ModuleHealth(self.name, ModuleStatus.OK.value, "", self.metrics())

    # -------------------------------------------------------------- helpers
    def flag_error(self, exc: BaseException) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.log.error("module %s error: %s", self.name, self.last_error)
