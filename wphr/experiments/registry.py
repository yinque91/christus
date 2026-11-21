"""Pre-registered experiment specifications.

An experiment is frozen before it runs: hypothesis, primary metric, conditions,
seeds, sample size and stopping rule. The runner may not silently change any of
them; the frozen JSON is written next to the raw results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

REGISTRY: dict[str, "ExperimentSpec"] = {}


@dataclass
class ExperimentSpec:
    experiment_id: str
    version: str
    title: str
    hypothesis: str
    primary_metric: str
    secondary_metrics: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    episodes_per_seed: int = 1
    steps_per_episode: int = 1000
    exclusion_rules: list[str] = field(default_factory=list)
    stopping_rule: str = ""
    notes: str = ""
    runner: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("runner", None)
        return payload


def register(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.experiment_id in REGISTRY:
        raise ValueError(f"experiment already registered: {spec.experiment_id}")
    REGISTRY[spec.experiment_id] = spec
    return spec


def list_experiments() -> list[ExperimentSpec]:
    return [REGISTRY[key] for key in sorted(REGISTRY)]


def get_experiment(experiment_id: str) -> ExperimentSpec:
    if experiment_id not in REGISTRY:
        raise KeyError(f"unknown experiment: {experiment_id}")
    return REGISTRY[experiment_id]
