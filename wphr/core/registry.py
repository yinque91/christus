"""Module registry: canonical ordering, aggregate health, bulk snapshot."""

from __future__ import annotations

from typing import Any, Iterator

from wphr.core.module import BaseModule, ModuleHealth, ModuleStatus

MODULE_ORDER: tuple[str, ...] = (
    "world",
    "practice",
    "affect",
    "temporal",
    "concept_language",
    "social",
    "volition",
)


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, BaseModule] = {}

    def register(self, module: BaseModule) -> None:
        if module.name in self._modules:
            raise ValueError(f"module already registered: {module.name}")
        self._modules[module.name] = module

    def get(self, name: str) -> BaseModule:
        if name not in self._modules:
            raise KeyError(f"unknown module: {name}")
        return self._modules[name]

    def __contains__(self, name: object) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def names(self) -> list[str]:
        rank = {name: i for i, name in enumerate(MODULE_ORDER)}
        return sorted(self._modules, key=lambda n: (rank.get(n, len(rank)), n))

    def __iter__(self) -> Iterator[BaseModule]:
        for name in self.names():
            yield self._modules[name]

    def missing_required(self) -> list[str]:
        return [name for name in MODULE_ORDER if name not in self._modules]

    def health_report(self) -> list[ModuleHealth]:
        return [module.health() for module in self]

    def overall_status(self) -> str:
        statuses = [health.status for health in self.health_report()]
        if ModuleStatus.FAILED.value in statuses:
            return ModuleStatus.FAILED.value
        if ModuleStatus.DEGRADED.value in statuses:
            return ModuleStatus.DEGRADED.value
        if statuses and all(s == ModuleStatus.DISABLED.value for s in statuses):
            return ModuleStatus.DISABLED.value
        return ModuleStatus.OK.value

    def snapshot(self) -> dict[str, Any]:
        return {module.name: module.snapshot() for module in self}

    def restore(self, payload: dict[str, Any]) -> None:
        for name, data in payload.items():
            module = self._modules.get(name)
            if module is not None:
                module.restore(data)

    def reset(self) -> None:
        for module in self:
            module.reset()

    def dispatch(self, event) -> None:
        for module in self:
            if not module.enabled:
                continue
            try:
                module.on_event(event)
            except Exception as exc:  # isolate module failures
                module.flag_error(exc)

    def update_group(self, group: str, step: int) -> None:
        for module in self:
            if module.group != group or not module.enabled:
                continue
            try:
                module.update(step)
            except Exception as exc:
                module.flag_error(exc)
