"""Experiment registry, runner entry points and the V0.2 specifications."""

from wphr.experiments.registry import (
    ExperimentSpec,
    list_experiments,
    get_experiment,
    register,
    REGISTRY,
)
from wphr.experiments.suite import run_experiment, summarize
from wphr.experiments import specs  # noqa: F401  (registers E001/E002 on import)

__all__ = [
    "ExperimentSpec",
    "list_experiments",
    "get_experiment",
    "register",
    "REGISTRY",
    "run_experiment",
    "summarize",
]
