"""M5 Concept/Language: concept graph, lexicon, language games.

Two halves, one module:

* **Concept formation** (V0.5) — a :class:`ConceptGraph` built from observation
  and from the outcomes of the subject's own actions. Interventions are the only
  thing that can make a cue-outcome relation causal; co-occurrence alone cannot.
* **Language** (V0.6) — a lexicon bound by cross-situational co-occurrence, plus
  the two-way rule from MASTER SPEC 8.3: a language claim becomes a bounded
  prior that never overwrites verified world evidence, and a concept with
  enough evidence can be named (produced) by the subject.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from wphr.concepts.graph import ConceptGraph
from wphr.core.event import Event, EventType
from wphr.core.module import BaseModule
from wphr.language.lexicon import Lexicon


class ConceptLanguageModule(BaseModule):
    name: ClassVar[str] = "concept_language"
    description: ClassVar[str] = "concept graph + lexicon + language games"
    group: ClassVar[str] = "concept"

    def reset(self) -> None:
        concepts_cfg = self.ctx.config.get("concepts", {}) or {}
        language_cfg = self.ctx.config.get("language", {}) or {}
        self.mode = str(concepts_cfg.get("mode", "full"))
        self.graph = ConceptGraph(
            mode=self.mode, split_min_evidence=int(concepts_cfg.get("split_min_evidence", 3))
        )
        # Existing default is the grid action "interact"; CHRISTUS-03 can list
        # its real PUSH_* actions without changing the default behavior.
        self.interaction_actions = {
            str(a) for a in (concepts_cfg.get("interaction_actions") or ["interact"])
        }
        self.lexicon = Lexicon()
        self.language_enabled = bool(language_cfg.get("enabled", True))
        self.prior_weight = float(language_cfg.get("prior_weight", 1.5))
        self.persist_every = int(concepts_cfg.get("persist_every", 100))
        self._last_cue = "none"
        self._state: dict[str, Any] = {
            "claims": 0,
            "productions": 0,
            "persist_step": 0,
            "beliefs": {},
        }

    # ------------------------------------------------------------------ event
    def on_event(self, event: Event) -> None:
        super().on_event(event)
        if event.event_type == EventType.OBSERVATION.value:
            cue = str((event.observation or {}).get("derived", {}).get("cue", "none"))
            self._last_cue = cue
            self.graph.observe(cue, step=int(event.step), context="seen")
            return

        if event.event_type == EventType.OUTCOME.value:
            outcome = event.actual_outcome or {}
            action = outcome.get("action")
            info = outcome.get("info") or {}
            cue = str(info.get("cue", "none"))
            reward = float(outcome.get("reward", 0.0))
            if action in self.interaction_actions and cue not in ("", "none"):
                self.graph.intervene(cue, reward, step=int(event.step))
                self._maybe_produce(cue)
            return

        if event.event_type == EventType.LANGUAGE_CLAIM.value and self.language_enabled:
            self._handle_claim(event)

    def _handle_claim(self, event: Event) -> None:
        payload = event.observation or {}
        if payload.get("speaker") == "self":
            return  # self-produced names are outputs, not new evidence
        words = [str(w) for w in payload.get("words", [])]
        if not words:
            return
        cue = self._last_cue
        for word in words:
            self.lexicon.observe(word, cue, step=int(event.step))
        polarity = payload.get("polarity")
        if polarity is None or cue in ("", "none"):
            return
        reliability = float(event.source_reliability)
        # A claim is a prior with bounded weight; it can never overwrite world
        # evidence because belief() weights it against the evidence count.
        self.graph.set_prior(
            cue,
            float(np.sign(float(polarity))) * reliability,
            self.prior_weight,
            step=int(event.step),
        )
        self._state["claims"] += 1

    def _maybe_produce(self, cue: str) -> None:
        if not self.language_enabled:
            return
        if self.graph.confidence(cue) < 0.5:
            return
        word, confidence = self.lexicon.word_for(cue)
        if not word:
            return
        self.ctx.set_shared(
            "language.production",
            {"word": word, "cue": cue, "confidence": round(confidence, 4)},
            source=self.name,
            confidence=round(confidence, 4),
        )
        self._state["productions"] += 1

    # ----------------------------------------------------------------- update
    def update(self, step: int) -> None:
        super().update(step)
        beliefs = {
            cue: round(self.graph.belief(cue), 6) for cue in self.graph.cues()
        }
        self._state["beliefs"] = beliefs
        switch_beliefs = {
            cue: value for cue, value in beliefs.items() if cue.startswith("switch_")
        }
        self.ctx.set_shared("concept.beliefs", beliefs, source=self.name)
        self.ctx.set_shared(
            "concept.confidence",
            {cue: round(self.graph.confidence(cue), 6) for cue in self.graph.cues()},
            source=self.name,
        )
        self.ctx.set_shared(
            "concept.safe_cue",
            max(switch_beliefs, key=switch_beliefs.get) if switch_beliefs else "",
            source=self.name,
        )
        self.ctx.set_shared(
            "concept.norm",
            float(np.mean([abs(v) for v in beliefs.values()])) if beliefs else 0.0,
            source=self.name,
        )
        self.ctx.set_shared("concept.count", len(self.graph.nodes), source=self.name)
        self.ctx.set_shared("concept.stats", self.graph.stats(), source=self.name)
        if step - int(self._state.get("persist_step", 0)) >= self.persist_every:
            self._state["persist_step"] = step
            memory = getattr(self.ctx, "memory", None)
            store = getattr(memory, "store", None) if memory is not None else None
            if store is not None:
                for record in self.graph.to_records():
                    store.upsert_memory(record)

    # ------------------------------------------------------------- accessors
    def belief(self, cue: str) -> float:
        return self.graph.belief(cue)

    def beliefs(self) -> dict[str, float]:
        return {cue: round(self.graph.belief(cue), 6) for cue in self.graph.cues()}

    def naming_accuracy(self) -> float:
        """Is each word bound to the cue it actually names?"""
        correct = 0
        total = 0
        for cue, word in (("switch_blue", "blue"), ("switch_orange", "orange")):
            bound, _confidence = self.lexicon.cue_for(word)
            total += 1
            if bound == cue:
                correct += 1
        return correct / total if total else 0.0

    def metrics(self) -> dict[str, Any]:
        base = super().metrics()
        base.update(
            {
                "concept_nodes": len(self.graph.nodes),
                "interventions": self.graph.interventions,
                "splits": self.graph.splits,
                "revisions": self.graph.revisions,
                "claims": int(self._state["claims"]),
                "productions": int(self._state["productions"]),
                "words": len(self.lexicon.bindings),
            }
        )
        return base

    # ------------------------------------------------------------ checkpoint
    def snapshot(self) -> dict[str, Any]:
        payload = super().snapshot()
        payload["graph"] = self.graph.snapshot()
        payload["lexicon"] = self.lexicon.snapshot()
        payload["last_cue"] = self._last_cue
        return payload

    def restore(self, payload: dict[str, Any]) -> None:
        super().restore(payload)
        if payload.get("graph"):
            self.graph.restore(payload["graph"])
        if payload.get("lexicon"):
            self.lexicon.restore(payload["lexicon"])
        self._last_cue = str(payload.get("last_cue", "none"))
