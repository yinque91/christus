"""Concept graph (MASTER SPEC 8.1).

A concept is not a label. Each node carries:

* observed attributes (mean outcome, interaction count, energy effect),
* evidence and counter-evidence counts,
* a confidence derived from them,
* the contexts in which it was observed,
* optionally several *modes* after a split.

The distinction that matters for the project's hypothesis is causal vs
correlational evidence:

* ``observe`` records that a cue was seen (co-occurrence),
* ``intervene`` records the outcome of an action the subject itself took.

Only interventions can raise causal confidence. A node whose intervention
outcomes are bimodal (clearly positive *and* clearly negative) is marked
``contextual`` and split into modes — that is how "this colour is safe" turns
into "this colour is safe in context A, harmful in context B" after a rule
reversal, without deleting the earlier evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from wphr.memory.layers import MemoryRecord
from wphr.util.ids import new_id


@dataclass
class ConceptMode:
    mode_id: str
    mean_reward: float
    count: int
    sign: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "mean_reward": round(self.mean_reward, 6),
            "count": self.count,
            "sign": self.sign,
        }


@dataclass
class ConceptNode:
    concept_id: str
    cue: str
    kind: str = "cue_outcome"
    attributes: dict[str, float] = field(default_factory=dict)
    observed: int = 0
    interventions: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    reward_sum: float = 0.0
    first_step: int = 0
    last_step: int = 0
    contexts: dict[str, int] = field(default_factory=dict)
    modes: list[ConceptMode] = field(default_factory=list)
    contextual: bool = False
    splits: int = 0

    # -------------------------------------------------------------- evidence
    @property
    def total_interventions(self) -> int:
        return self.positive + self.negative + self.neutral

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.total_interventions if self.total_interventions else 0.0

    @property
    def confidence(self) -> float:
        """Evidence-weighted confidence in the node's mean outcome."""
        total = self.total_interventions
        if total == 0:
            return 0.0
        sample = 1.0 - math.exp(-total / 4.0)
        agreement = max(self.positive, self.negative) / total
        return float(round(sample * agreement, 6))

    def observe(self, *, step: int, context: str) -> None:
        self.observed += 1
        self.last_step = step
        self.contexts[context] = self.contexts.get(context, 0) + 1

    def intervene(self, reward: float, *, step: int, context: str) -> None:
        self.interventions += 1
        self.reward_sum += reward
        if reward > 0.05:
            self.positive += 1
        elif reward < -0.05:
            self.negative += 1
        else:
            self.neutral += 1
        self.last_step = step
        self.contexts[context] = self.contexts.get(context, 0) + 1

    # ----------------------------------------------------------------- split
    def maybe_split(self, *, min_evidence: int = 3) -> bool:
        """Split into modes when interventions are clearly bimodal.

        A node splits at most once per evidence regime: without the guard the
        counter would increment on every further counter-example after the
        threshold, which would make "number of splits" meaningless.
        """
        if self.contextual and self.modes:
            return False
        if self.positive < min_evidence or self.negative < min_evidence:
            self.contextual = False
            return False
        total = self.total_interventions
        positive_ratio = self.positive / total
        if 0.2 <= positive_ratio <= 0.8:
            # estimate the two modes from the counts, not from a hidden label
            positive_mean = 1.0
            negative_mean = -1.0
            self.modes = [
                ConceptMode(new_id("mode_"), positive_mean, self.positive, 1),
                ConceptMode(new_id("mode_"), negative_mean, self.negative, -1),
            ]
            self.contextual = True
            self.splits += 1
            return True
        self.contextual = False
        return False

    def belief(self, mode: str = "mean") -> float:
        """What the concept currently predicts for this cue.

        ``mean``  — plain average (correlational estimate)
        ``major`` — sign of the dominant mode (what a policy would act on)
        """
        if mode == "major" and self.contextual and self.modes:
            best = max(self.modes, key=lambda m: m.count)
            return float(best.mean_reward)
        return self.mean_reward

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "cue": self.cue,
            "kind": self.kind,
            "attributes": {k: round(v, 6) for k, v in self.attributes.items()},
            "observed": self.observed,
            "interventions": self.interventions,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "reward_sum": round(self.reward_sum, 6),
            "mean_reward": round(self.mean_reward, 6),
            "confidence": self.confidence,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "contexts": dict(self.contexts),
            "contextual": self.contextual,
            "splits": self.splits,
            "modes": [m.to_dict() for m in self.modes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConceptNode":
        node = cls(
            concept_id=str(data["concept_id"]),
            cue=str(data["cue"]),
            kind=str(data.get("kind", "cue_outcome")),
        )
        node.attributes = {k: float(v) for k, v in data.get("attributes", {}).items()}
        node.observed = int(data.get("observed", 0))
        node.interventions = int(data.get("interventions", 0))
        node.positive = int(data.get("positive", 0))
        node.negative = int(data.get("negative", 0))
        node.neutral = int(data.get("neutral", 0))
        node.reward_sum = float(data.get("reward_sum", 0.0))
        node.first_step = int(data.get("first_step", 0))
        node.last_step = int(data.get("last_step", 0))
        node.contexts = {str(k): int(v) for k, v in data.get("contexts", {}).items()}
        node.contextual = bool(data.get("contextual", False))
        node.splits = int(data.get("splits", 0))
        node.modes = [
            ConceptMode(
                mode_id=str(m["mode_id"]),
                mean_reward=float(m["mean_reward"]),
                count=int(m["count"]),
                sign=int(m["sign"]),
            )
            for m in data.get("modes", [])
        ]
        return node


class ConceptGraph:
    """A small, auditable concept graph over cue/outcome experience."""

    def __init__(self, *, mode: str = "full", split_min_evidence: int = 3) -> None:
        self.mode = mode
        self.split_min_evidence = int(split_min_evidence)
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: list[dict[str, Any]] = []
        self.revisions = 0
        self.splits = 0
        self.observations = 0
        self.interventions = 0
        self.preset: dict[str, float] = {}
        # language-derived priors: a claim can tilt a belief but never overwrite
        # verified world evidence (MASTER SPEC 8.3).
        self.priors: dict[str, float] = {}
        self.prior_weights: dict[str, float] = {}

    # ------------------------------------------------------------- learning
    def node(self, cue: str, step: int = 0) -> ConceptNode:
        if cue not in self.nodes:
            self.nodes[cue] = ConceptNode(
                concept_id=f"cpt_{cue}", cue=cue, first_step=step, last_step=step
            )
        return self.nodes[cue]

    def observe(self, cue: str, *, step: int, context: str = "seen") -> None:
        if self.mode == "preset" or cue in ("", "none"):
            return
        node = self.node(cue, step)
        node.observe(step=step, context=context)
        self.observations += 1

    def intervene(
        self, cue: str, reward: float, *, step: int, context: str = "interact"
    ) -> None:
        if self.mode == "preset" or cue in ("", "none"):
            return
        if self.mode == "cooccurrence":
            # correlation-only condition: the cue is registered but the outcome
            # of the subject's own action is deliberately NOT attributed to it.
            self.observe(cue, step=step, context="cooccurrence")
            return
        node = self.node(cue, step)
        previous = node.mean_reward
        node.intervene(reward, step=step, context=context)
        self.interventions += 1
        if abs(previous - node.mean_reward) > 0.35:
            self.revisions += 1
        if self.mode == "full" and node.maybe_split(min_evidence=self.split_min_evidence):
            self.splits += 1

    # --------------------------------------------------------------- queries
    def set_prior(self, cue: str, value: float, weight: float, *, step: int = 0) -> None:
        """A language claim enters as a prior with a bounded weight."""
        if cue in ("", "none"):
            return
        previous = self.priors.get(cue, 0.0)
        self.priors[cue] = float(value)
        self.prior_weights[cue] = float(max(0.0, weight))
        self.node(cue, step)
        if abs(previous - value) > 0.35:
            self.revisions += 1

    def belief(self, cue: str, mode: str = "mean") -> float:
        if self.mode == "preset":
            return float(self.preset.get(cue, 0.0))
        node = self.nodes.get(cue)
        evidence_mean = 0.0 if node is None else node.belief(mode)
        evidence_n = 0 if node is None else node.total_interventions
        prior = self.priors.get(cue)
        if prior is None:
            return evidence_mean
        weight = self.prior_weights.get(cue, 0.0)
        total = evidence_n + weight
        if total <= 0:
            return float(prior)
        return float((evidence_mean * evidence_n + prior * weight) / total)

    def rank(self, cues: list[str]) -> list[tuple[str, float]]:
        return sorted(((cue, self.belief(cue)) for cue in cues), key=lambda kv: -kv[1])

    def cues(self) -> list[str]:
        """Every cue the graph has any information about (nodes, priors, presets)."""
        return sorted(set(self.nodes) | set(self.priors) | set(self.preset))

    def confidence(self, cue: str) -> float:
        node = self.nodes.get(cue)
        return 0.0 if node is None else node.confidence

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nodes": len(self.nodes),
            "observations": self.observations,
            "interventions": self.interventions,
            "revisions": self.revisions,
            "splits": self.splits,
            "contextual_nodes": sum(1 for n in self.nodes.values() if n.contextual),
        }

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nodes": {cue: node.to_dict() for cue, node in self.nodes.items()},
            "edges": list(self.edges),
            "revisions": self.revisions,
            "splits": self.splits,
            "observations": self.observations,
            "interventions": self.interventions,
            "preset": dict(self.preset),
            "priors": dict(self.priors),
            "prior_weights": dict(self.prior_weights),
            "split_min_evidence": self.split_min_evidence,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.nodes = {
            cue: ConceptNode.from_dict(data) for cue, data in payload.get("nodes", {}).items()
        }
        self.edges = list(payload.get("edges", []))
        self.revisions = int(payload.get("revisions", 0))
        self.splits = int(payload.get("splits", 0))
        self.observations = int(payload.get("observations", 0))
        self.interventions = int(payload.get("interventions", 0))
        self.preset = {str(k): float(v) for k, v in payload.get("preset", {}).items()}
        self.priors = {str(k): float(v) for k, v in payload.get("priors", {}).items()}
        self.prior_weights = {
            str(k): float(v) for k, v in payload.get("prior_weights", {}).items()
        }
        self.mode = payload.get("mode", self.mode)
        self.split_min_evidence = int(payload.get("split_min_evidence", self.split_min_evidence))

    def to_records(self) -> list[MemoryRecord]:
        """Semantic memory records for every node with at least one intervention."""
        records: list[MemoryRecord] = []
        for cue, node in self.nodes.items():
            if node.total_interventions == 0:
                continue
            records.append(
                MemoryRecord(
                    memory_id=f"concept_{cue}",
                    layer="semantic",
                    kind="concept_node",
                    key=cue,
                    content=node.to_dict(),
                    source="intervention" if node.interventions else "observation",
                    confidence=node.confidence,
                    importance=min(1.0, abs(node.mean_reward)),
                    support_count=node.positive,
                    counter_count=node.negative,
                    scope="grid_world/cue",
                    created_step=node.first_step,
                    updated_step=node.last_step,
                )
            )
        return records
