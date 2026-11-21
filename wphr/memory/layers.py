"""Layered memory (MASTER SPEC 6.1).

Five layers, each with an explicit lifecycle:

============  ==================================================  ==================
layer         what it holds                                       update rule
============  ==================================================  ==================
working       current buffer, held by M4                          capacity-bounded
episodic      raw events (append-only, in ``events``)             never rewritten
semantic      cue -> action rules with support/counter counts     evidence-driven
procedural    action policy statistics (Q table + success rates)  outcome-driven
autobiographical  periodic summaries of the subject's own history  periodic
============  ==================================================  ==================

Every non-episodic record carries source, confidence, support count, counter
count, applicability scope, created/updated step and an accessibility value that
decays with time and grows with use. This is what makes "over-fixation" testable:
a rule whose evidence is stale becomes *less* certain, and the policy raises its
exploration for that cue instead of freezing on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from wphr.util.ids import new_id, utc_iso, utc_now

LAYERS = ("working", "episodic", "semantic", "procedural", "autobiographical")


@dataclass
class MemoryRecord:
    memory_id: str
    layer: str
    kind: str
    key: str
    content: dict[str, Any]
    source: str
    confidence: float
    importance: float
    support_count: int = 0
    counter_count: int = 0
    scope: str = ""
    created_step: int = 0
    updated_step: int = 0
    created_at: float = field(default_factory=utc_now)
    updated_at: float = field(default_factory=utc_now)
    last_access: float | None = None
    access_count: int = 0
    active: bool = True

    def to_row(self) -> tuple:
        return (
            self.memory_id,
            self.layer,
            self.kind,
            self.key,
            json.dumps(self.content, ensure_ascii=False),
            self.source,
            self.confidence,
            self.importance,
            self.support_count,
            self.counter_count,
            self.scope,
            self.created_step,
            self.updated_step,
            self.created_at,
            self.updated_at,
            self.last_access,
            self.access_count,
            1 if self.active else 0,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=row["memory_id"],
            layer=row["layer"],
            kind=row["kind"],
            key=row["key"] or "",
            content=json.loads(row["content"]),
            source=row["source"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            support_count=int(row["support_count"]),
            counter_count=int(row["counter_count"]),
            scope=row["scope"] or "",
            created_step=int(row["created_step"]),
            updated_step=int(row["updated_step"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_access=row["last_access"],
            access_count=int(row["access_count"]),
            active=bool(row["active"]),
        )

    def accessibility(self, now: float, half_life_s: float = 3600.0) -> float:
        """How retrievable this memory is right now.

        Decays exponentially with time since last access and is boosted
        logarithmically by use. Never reaches zero so nothing is silently lost.
        """
        stamp = self.last_access or self.updated_at
        age = max(0.0, now - stamp)
        decay = math.exp(-math.log(2.0) * age / max(1.0, half_life_s))
        boost = 1.0 + math.log1p(self.access_count)
        importance = 0.5 + 0.5 * min(1.0, self.importance)
        return float(min(2.0, decay * boost * importance))


class SemanticMemory:
    """Cue/action rules with support, counter-evidence and staleness."""

    def __init__(
        self,
        *,
        staleness_tau_steps: float = 600.0,
        uncertainty_gain: float = 0.5,
        min_confidence_for_certainty: float = 0.6,
    ) -> None:
        self.staleness_tau_steps = float(staleness_tau_steps)
        self.uncertainty_gain = float(uncertainty_gain)
        self.min_confidence = float(min_confidence_for_certainty)
        self.rules: dict[str, dict[str, dict[str, Any]]] = {}
        self.updates = 0
        self.revisions = 0

    # ---------------------------------------------------------------- evidence
    def record_evidence(
        self,
        cue: str,
        action: str,
        reward: float,
        *,
        step: int,
        source: str = "action_feedback",
        importance_scale: float = 1.0,
    ) -> dict[str, Any]:
        cue_rules = self.rules.setdefault(cue, {})
        rule = cue_rules.setdefault(
            action,
            {
                "support": 0,
                "counter": 0,
                "reward_sum": 0.0,
                "confidence": 0.0,
                "last_step": step,
                "source": source,
                "importance_scale": 1.0,
            },
        )
        rule["importance_scale"] = float(max(rule.get("importance_scale", 1.0), importance_scale))
        previous_confidence = float(rule["confidence"])
        if reward > 0.05:
            rule["support"] += 1
        elif reward < -0.05:
            rule["counter"] += 1
        else:
            rule["support"] += 1  # neutral evidence still counts as confirmed inaction cost
        rule["reward_sum"] += reward
        total = rule["support"] + rule["counter"]
        rule["confidence"] = rule["support"] / total if total else 0.0
        rule["last_step"] = step
        self.updates += 1
        # A "revision" is a material drop in confidence after counter-evidence,
        # not any arithmetic change. 0.15 is a reporting threshold, not a claim
        # about how belief should work.
        if previous_confidence - rule["confidence"] >= 0.15:
            self.revisions += 1
        return rule

    # ------------------------------------------------------------- staleness
    def effective_confidence(self, cue: str, step: int) -> float:
        """Confidence discounted by how long ago the evidence was collected."""
        cue_rules = self.rules.get(cue)
        if not cue_rules:
            return 0.0
        best = 0.0
        for rule in cue_rules.values():
            age = max(0, step - int(rule["last_step"]))
            freshness = math.exp(-age / max(1.0, self.staleness_tau_steps))
            total = rule["support"] + rule["counter"]
            weight = min(1.0, total / 3.0)
            best = max(best, float(rule["confidence"]) * freshness * weight)
        return float(best)

    def uncertainty(self, cue: str, step: int) -> float:
        """0 = well-established rule, 1 = no trustworthy knowledge for this cue."""
        return float(1.0 - min(1.0, self.effective_confidence(cue, step)))

    def exploration_bonus(self, cue: str, step: int) -> float:
        return self.uncertainty_gain * self.uncertainty(cue, step)

    # --------------------------------------------------------------- queries
    def best_action(self, cue: str) -> str | None:
        cue_rules = self.rules.get(cue)
        if not cue_rules:
            return None
        return max(cue_rules, key=lambda action: cue_rules[action]["reward_sum"])

    def snapshot(self) -> dict[str, Any]:
        return {
            "rules": json.loads(json.dumps(self.rules)),
            "updates": self.updates,
            "revisions": self.revisions,
            "staleness_tau_steps": self.staleness_tau_steps,
            "uncertainty_gain": self.uncertainty_gain,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.rules = json.loads(json.dumps(payload.get("rules", {})))
        self.updates = int(payload.get("updates", 0))
        self.revisions = int(payload.get("revisions", 0))
        self.staleness_tau_steps = float(payload.get("staleness_tau_steps", self.staleness_tau_steps))
        self.uncertainty_gain = float(payload.get("uncertainty_gain", self.uncertainty_gain))

    def rule_rows(self) -> list[MemoryRecord]:
        out: list[MemoryRecord] = []
        for cue, actions in self.rules.items():
            for action, rule in actions.items():
                total = rule["support"] + rule["counter"]
                out.append(
                    MemoryRecord(
                        memory_id=f"sem_{cue}_{action}",
                        layer="semantic",
                        kind="cue_action_rule",
                        key=f"{cue}:{action}",
                        content={
                            "cue": cue,
                            "action": action,
                            "mean_reward": (rule["reward_sum"] / total) if total else 0.0,
                            "support": rule["support"],
                            "counter": rule["counter"],
                        },
                        source=str(rule.get("source", "action_feedback")),
                        confidence=float(rule["confidence"]),
                        importance=min(
                            1.0,
                            abs(rule["reward_sum"])
                            / max(1, total)
                            * float(rule.get("importance_scale", 1.0)),
                        ),
                        support_count=int(rule["support"]),
                        counter_count=int(rule["counter"]),
                        scope="grid_world/cue_conditional",
                        created_step=int(rule["last_step"]),
                        updated_step=int(rule["last_step"]),
                    )
                )
        return out


class AutobiographicalMemory:
    """Periodic summaries of the subject's own history, raw events kept intact."""

    def __init__(self) -> None:
        self.summaries: list[dict[str, Any]] = []

    def add(
        self,
        *,
        start_step: int,
        end_step: int,
        text: str,
        source: str = "consolidation",
    ) -> dict[str, Any]:
        entry = {
            "summary_id": new_id("sum_"),
            "layer": "autobiographical",
            "start_step": int(start_step),
            "end_step": int(end_step),
            "text": text,
            "source": source,
            "created_at": utc_now(),
            "created_iso": utc_iso(),
        }
        self.summaries.append(entry)
        return entry

    def latest(self) -> dict[str, Any] | None:
        return self.summaries[-1] if self.summaries else None

    def snapshot(self) -> dict[str, Any]:
        return {"summaries": self.summaries[-64:]}

    def restore(self, payload: dict[str, Any]) -> None:
        self.summaries = list(payload.get("summaries", []))


class MemoryStore:
    """Facade over the layers plus optional SQLite persistence."""

    def __init__(self, store: Any | None, config: Any) -> None:
        section = config.get("memory.semantic", {}) if config else {}
        self.store = store
        self.enabled = bool(config.get("memory.enabled", True)) if config else True
        self.semantic = SemanticMemory(
            staleness_tau_steps=float(section.get("staleness_tau_steps", 600.0)),
            uncertainty_gain=float(section.get("uncertainty_gain", 0.5)),
        )
        self.autobiographical = AutobiographicalMemory()
        self.consolidate_every = int(config.get("memory.consolidate_every", 100)) if config else 100
        self.working: list[dict[str, Any]] = []
        self.working_capacity = int(config.get("memory.working_capacity", 32)) if config else 32
        self.last_consolidate_step = 0
        self.consolidations = 0
        self.records_written = 0

    # --------------------------------------------------------------- working
    def remember_working(self, item: dict[str, Any]) -> None:
        self.working.append(item)
        if len(self.working) > self.working_capacity:
            del self.working[: len(self.working) - self.working_capacity]

    # ---------------------------------------------------------------- write
    def record_outcome(
        self,
        *,
        cue: str,
        action: str,
        reward: float,
        step: int,
        enabled: bool = True,
        importance_scale: float = 1.0,
    ) -> None:
        if not self.enabled or not enabled or cue in ("", "none"):
            return
        self.semantic.record_evidence(
            cue, action, reward, step=step, importance_scale=importance_scale
        )

    def exploration_bonus(self, cue: str, step: int, enabled: bool = True) -> float:
        if not self.enabled or not enabled:
            return 0.0
        return self.semantic.exploration_bonus(cue, step)

    # ----------------------------------------------------------- consolidate
    def maybe_consolidate(
        self, *, step: int, subject_id: str, module_metrics: dict[str, Any]
    ) -> list[MemoryRecord]:
        if not self.enabled or step - self.last_consolidate_step < self.consolidate_every:
            return []
        self.last_consolidate_step = step
        self.consolidations += 1
        records = self._persist_semantic(step)
        self._write_summary(step, subject_id, module_metrics, records)
        return records

    def _persist_semantic(self, step: int) -> list[MemoryRecord]:
        rows = self.semantic.rule_rows()
        if self.store is None:
            self.records_written += len(rows)
            return rows
        for record in rows:
            self.store.upsert_memory(record)
        self.store.log_consolidation(step=step, layer="semantic", written=len(rows))
        self.records_written += len(rows)
        return rows

    def _write_summary(
        self,
        step: int,
        subject_id: str,
        module_metrics: dict[str, Any],
        records: Iterable[MemoryRecord],
    ) -> None:
        known = len(self.semantic.rules)
        revisions = self.semantic.revisions
        text = (
            f"subject {subject_id} at step {step}: {known} cue contexts tracked, "
            f"{len(list(records))} semantic rules consolidated, {revisions} confidence revisions. "
            f"practice metrics: {module_metrics}"
        )
        entry = self.autobiographical.add(
            start_step=self.last_consolidate_step, end_step=step, text=text
        )
        if self.store is not None:
            self.store.add_summary(entry)

    # --------------------------------------------------------------- queries
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "semantic_contexts": len(self.semantic.rules),
            "semantic_rules": sum(len(v) for v in self.semantic.rules.values()),
            "revisions": self.semantic.revisions,
            "consolidations": self.consolidations,
            "summaries": len(self.autobiographical.summaries),
            "working_items": len(self.working),
            "records_written": self.records_written,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic.snapshot(),
            "autobiographical": self.autobiographical.snapshot(),
            "working": list(self.working),
            "last_consolidate_step": self.last_consolidate_step,
            "consolidations": self.consolidations,
            "records_written": self.records_written,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.semantic.restore(payload.get("semantic", {}))
        self.autobiographical.restore(payload.get("autobiographical", {}))
        self.working = list(payload.get("working", []))
        self.last_consolidate_step = int(payload.get("last_consolidate_step", 0))
        self.consolidations = int(payload.get("consolidations", 0))
        self.records_written = int(payload.get("records_written", 0))
