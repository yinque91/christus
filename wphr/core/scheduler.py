"""Multi-timescale scheduler (MASTER SPEC 5.3).

Groups update at fixed step periods taken from configuration. Actual execution
time per group is recorded so that the report can state real cost instead of
claiming "fast".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GroupStats:
    runs: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.runs += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.runs if self.runs else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "runs": self.runs,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 4),
            "max_ms": round(self.max_ms, 3),
        }


class Scheduler:
    ORDER = ("fast", "affect", "concept", "slow")

    def __init__(self, periods: dict[str, int]) -> None:
        if not periods:
            raise ValueError("scheduler needs at least one group")
        for group, period in periods.items():
            if int(period) < 1:
                raise ValueError(f"scheduler period for {group} must be >= 1")
        self.periods: dict[str, int] = {k: int(v) for k, v in periods.items()}
        self.stats: dict[str, GroupStats] = {g: GroupStats() for g in self.periods}
        self._timers: dict[str, float] = {}

    def due(self, step: int) -> list[str]:
        """Groups whose period divides ``step``, in canonical order."""
        if step <= 0:
            return []
        groups = [g for g, period in self.periods.items() if step % period == 0]
        rank = {name: i for i, name in enumerate(self.ORDER)}
        return sorted(groups, key=lambda g: rank.get(g, len(rank)))

    def begin(self, group: str) -> None:
        self._timers[group] = time.perf_counter()

    def end(self, group: str) -> float:
        started = self._timers.pop(group, None)
        if started is None:
            return 0.0
        ms = (time.perf_counter() - started) * 1000.0
        self.stats.setdefault(group, GroupStats()).record(ms)
        return ms

    def time_group(self, group: str, fn, *args, **kwargs):
        self.begin(group)
        try:
            return fn(*args, **kwargs)
        finally:
            self.end(group)

    def report(self) -> dict[str, dict[str, float | int]]:
        return {group: stat.to_dict() for group, stat in self.stats.items()}

    def reset(self) -> None:
        for stat in self.stats.values():
            stat.runs = 0
            stat.total_ms = 0.0
            stat.max_ms = 0.0
