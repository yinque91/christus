"""CHRISTUS-COORDINATION-01: one decision loop over existing modules.

It is a module, not a second subject. The only action exit is
``Subject.action_hook``. Candidate actions are scored by:

* world model: predicted next ball state / contact opportunity;
* cognitive readout: CORE-READOUT-01's h/c -> scent direction;
* affect: exploration/risk/fatigue terms (never a reward substitute);
* memory: gated semantic cue->action evidence with confidence/staleness;
* volition: current goal priority;
* concept/language: gated belief valence.

The coordinator logs every contribution and resolves conflicts explicitly.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from wphr.core.event import EventType
from wphr.core.module import BaseModule
from christus03.policy import ActionSlotAdapter
from christus03.predictor import transition_id
from christus03.world import ACTIONS, ACTION_INDEX
from christus import observed_drift


class CoordinationModule(BaseModule):
    name = "coordination"
    group = "fast"

    def reset(self) -> None:
        cfg = self.ctx.config.get("coordination", {}) or {}
        self.cfg = {
            "use_world": bool(cfg.get("use_world", True)),
            "use_cognitive": bool(cfg.get("use_cognitive", True)),
            "use_affect": bool(cfg.get("use_affect", True)),
            "use_memory": bool(cfg.get("use_memory", True)),
            "use_volition": bool(cfg.get("use_volition", True)),
            "use_concept": bool(cfg.get("use_concept", True)),
            "world_weight": float(cfg.get("world_weight", 1.0)),
            "uncertainty_weight": float(cfg.get("uncertainty_weight", 40.0)),
            "cognitive_weight": float(cfg.get("cognitive_weight", 0.8)),
            "affect_explore_weight": float(cfg.get("affect_explore_weight", 0.15)),
            "affect_risk_weight": float(cfg.get("affect_risk_weight", 0.25)),
            "affect_fatigue_weight": float(cfg.get("affect_fatigue_weight", 0.08)),
            "memory_weight": float(cfg.get("memory_weight", 0.45)),
            "memory_conf_threshold": float(cfg.get("memory_conf_threshold", 0.20)),
            "memory_external_reliability": float(cfg.get("memory_external_reliability", 0.5)),
            # Current-reality gate. Disabled by default for compatibility with
            # archived COORD-01/02 runs; the new runner enables it explicitly.
            "memory_reality_gate": bool(cfg.get("memory_reality_gate", False)),
            # Ablation switch: keep retrieval but freeze new evidence. This is
            # an experimental control, never used to write a chosen rule.
            "memory_write_enabled": bool(cfg.get("memory_write_enabled", True)),
            "memory_reality_margin": float(cfg.get("memory_reality_margin", 0.05)),
            "memory_reality_fraction": float(cfg.get("memory_reality_fraction", 0.25)),
            "memory_conflict_discount": float(cfg.get("memory_conflict_discount", 0.5)),
            "memory_allowed_sources": list(cfg.get(
                "memory_allowed_sources",
                ["coordination.observed", "coordination.observed_wait"])),
            "volition_weight": float(cfg.get("volition_weight", 0.20)),
            "concept_weight": float(cfg.get("concept_weight", 0.30)),
            "concept_conf_threshold": float(cfg.get("concept_conf_threshold", 0.20)),
            "affect_override_margin": float(cfg.get("affect_override_margin", 1e-9)),
        }
        self.subject = None
        self.predictor = None
        self.readout = None
        self.current_meta: dict = {}
        self.current_values: list[float] = [0.0] * 5
        self.pending: dict | None = None
        self.awaiting_affect = None
        self.processed: set[str] = set()
        self.last: dict[str, Any] = {}
        self._gate_meta: dict[str, Any] = {}
        self.stats = {
            "steps": 0, "memory_records": 0, "memory_gate_open": 0,
            "memory_gate_closed": 0, "conflicts": 0, "affect_overrides_blocked": 0,
            "readout_updates": 0, "duplicates": 0, "chosen": "WAIT",
            "memory_gate_rejected": 0,
        }

    def attach(self, subject, predictor, readout) -> None:
        self.subject = subject
        self.predictor = predictor
        self.readout = readout
        self.predictor.attach_subject(subject)

    # ---------------------------------------------------------------- events
    def on_event(self, event) -> None:
        super().on_event(event)
        if event.event_type == EventType.OBSERVATION.value:
            payload = event.observation or {}
            self.current_values = list(payload.get("values") or [0.0] * 5)
            self.current_meta = dict(payload.get("meta") or {})
            if self.pending and not self.current_meta.get("visible"):
                self._trace("unobserved", {"tid": self.pending["tid"], "reason": "occluded"})
            self._finish_readout()
            return
        if event.event_type == EventType.OUTCOME.value:
            outcome = event.actual_outcome or {}
            if (self.pending and event.step == self.pending["step"]
                    and outcome.get("action") == self.pending["action"]):
                self.pending["outcome"] = dict(outcome)
        elif event.event_type == EventType.PREDICTION.value:
            self._learn(event)

    def _finish_readout(self) -> None:
        pending = self.pending
        if pending is None or self.readout is None:
            return
        if self.current_meta.get("visible") and "scent" in self.current_meta:
            target = float((self.current_meta.get("scent") or [0.0, 0.0])[0])
            error = self.readout.learn_state(
                target,
                h=np.asarray(pending["h"], dtype=np.float64),
                c=np.asarray(pending["c"], dtype=np.float64),
            )
            self.stats["readout_updates"] += 1
            self.last["readout_target"] = target
            self.last["readout_error"] = float(error)
        self.pending = None

    def _cue(self, meta: dict | None = None) -> str:
        meta = meta or self.current_meta
        if not meta.get("visible", False):
            return "ball_occluded"
        if float(meta.get("contact_estimate", 0.0)) > 0.0:
            return "ball_near"
        return "ball_far"

    def _scope(self):
        if self._cue() != "ball_near":
            return None
        # Only a contacting push can change the ball. Bin visible position
        # relative to the anchor and velocity at detector resolution.
        x, y, _, vx, vy = self.current_values
        bins = [math.floor((x * 64 - 32) / 5), math.floor((y * 64 - 32) / 5),
                round(vx * 4), round(vy * 4)]
        prefix = "coord04/contact/" if self.cfg.get("observed_wait") else "coord02/contact/"
        return prefix + "/".join(map(str, bins))

    def _trace(self, kind, data):
        # Persist diagnostics alongside bus events without feeding them back
        # as sensory events (Affect normalises impulses by event count).
        event = self.subject._event(
            EventType.INTERNAL.value, kind, observation=data,
            provenance="coordination." + kind)
        event.assert_persistable()
        self.subject._pending.append(event)

    def _learn(self, event):
        p = self.pending
        tid = (event.observation or {}).get("transition_id")
        if tid in self.processed:
            self.stats["duplicates"] += 1
            return
        if not p or tid != p["tid"]:
            return
        if "prediction" not in p or "outcome" not in p:
            return
        actual = self.predictor.current_obs[:4]
        before = np.linalg.norm(np.asarray(p["observation"][:2]) - 0.5)
        after = np.linalg.norm(actual[:2] - 0.5)
        progress = float(before - after)
        record = {k: p[k] for k in ("tid", "step", "scope", "cue", "contact",
                                      "observation", "action", "prediction", "outcome")}
        record.update(actual=actual.tolist(), residual=(actual-p["prediction"]).tolist(),
                      prediction_error=event.prediction_error, progress=progress)
        self.awaiting_affect = record

    def finish_experience(self, subject):
        record = self.awaiting_affect
        if not record or record["tid"] in self.processed:
            return
        memory = subject.memory
        gain = float(subject.shared.get("affect.memory_importance_gain", 1.0))
        rule = None
        write_memory = bool(self.cfg.get("memory_write_enabled", True))
        if memory.enabled and write_memory and self.cfg.get("observed_wait"):
            record["drift"] = observed_drift.observe(
                self._state.setdefault("drift", {}), record, memory.semantic, subject.clock.step, gain)
        elif memory.enabled and write_memory and record["scope"] and abs(record["progress"]) > 1e-12:
            rule = memory.semantic.record_evidence(
                record["scope"], record["action"], float(np.sign(record["progress"])),
                step=subject.clock.step, source="coordination.observed", importance_scale=gain)
        record.update(
            rule=dict(rule) if rule else None,
            affect=subject.affect_values(),
            importance_scale=gain,
            # Minimum auditable experience structure. action and outcome are
            # separate fields; provenance says where the evidence came from.
            context_before=list(record["observation"]),
            context_after=list(record["actual"]),
            provenance=("coordination.observed_wait"
                        if self.cfg.get("observed_wait") else "coordination.observed"),
            prediction_source="christus03.ensemble",
            applicability={"scope": record["scope"], "cue": record.get("cue")},
            confidence=float(rule["confidence"]) if rule else 0.0,
            age=0,
        )
        self.processed.add(record["tid"])
        self.awaiting_affect = None
        memory.remember_working(record)
        self.stats["memory_records"] += 1
        self._trace("experience", record)

    def _experience_scores(self, subject, scope, preds):
        scores = np.zeros(len(ACTIONS))
        confidence = np.zeros(len(ACTIONS))
        memory = subject.memory
        self._gate_meta = {"entries": {}, "rejected_sources": [],
                           "reality_gate": bool(self.cfg.get("memory_reality_gate"))}
        if not scope or not memory.enabled or not self.cfg["use_memory"]:
            return scores, confidence
        distances = np.linalg.norm(preds[:, :, :2] - 0.5, axis=2)  # (actions, members)
        mean_dist = distances.mean(axis=1)
        world_value = 1.0 - mean_dist
        world_best = int(np.argmax(world_value))
        second_best = int(np.argsort(world_value)[-2]) if len(world_value) > 1 else world_best
        allowed_sources = set(self.cfg.get("memory_allowed_sources") or [])
        margin = float(self.cfg.get("memory_reality_margin", 0.05))
        fraction = float(self.cfg.get("memory_reality_fraction", 0.25))
        discount = float(self.cfg.get("memory_conflict_discount", 0.5))
        for action, rule in memory.semantic.rules.get(scope, {}).items():
            index = ACTION_INDEX.get(action)
            if index is None:
                continue
            source = str(rule.get("source", ""))
            if source not in allowed_sources:
                # The old action_feedback evidence was an action log, not an
                # outcome. It must never re-enter the current decision.
                self._gate_meta["rejected_sources"].append(
                    {"action": action, "source": source})
                self.stats["memory_gate_rejected"] = self.stats.get("memory_gate_rejected", 0) + 1
                continue
            n = rule["support"] + rule["counter"]
            if n <= 0:
                continue
            age = max(0, subject.clock.step - rule["last_step"])
            freshness = math.exp(-age / max(1, memory.semantic.staleness_tau_steps))
            balance = (rule["support"] - rule["counter"]) / n
            confidence[index] = abs(balance) * min(1, n / 3) * freshness
            if confidence[index] < self.cfg["memory_conf_threshold"]:
                continue
            sign = 1 if balance > 0 else -1
            # Current-reality compatibility: if memory wants to lift a non-best
            # action, or suppress the current best action, measure the current
            # world-evidence gap. A clear gap caps and discounts memory; an
            # ambiguous gap (<= margin) lets memory modulate.
            if sign > 0 and index != world_best:
                gap = float(world_value[world_best] - world_value[index])
            elif sign < 0 and index == world_best:
                gap = float(world_value[world_best] - world_value[second_best])
            else:
                gap = 0.0
            conflict = self.cfg.get("memory_reality_gate", False) and gap > margin
            limit = self.cfg["world_weight"] * float(distances[index].std())
            if conflict:
                limit = min(limit, max(0.0, gap) * fraction)
            score = sign * confidence[index] * min(limit, self.cfg["memory_weight"])
            if conflict:
                score *= discount
            scores[index] = score
            self._gate_meta["entries"][action] = {
                "source": source, "confidence": round(float(confidence[index]), 6),
                "balance": round(float(balance), 6), "world_gap": round(gap, 6),
                "conflict": bool(conflict),
                "cap": round(float(limit), 6), "score": round(float(score), 6),
            }
        return scores, confidence

    @staticmethod
    def resolve_scores(base, affect_score, memory_term, volition_score, concept_score):
        without_affect = base + memory_term + volition_score + concept_score
        scores = without_affect + affect_score
        best = int(np.argmax(scores))
        blocked = best not in (int(np.argmax(without_affect)), int(np.argmax(base)))
        if blocked:
            return without_affect, int(np.argmax(without_affect)), True
        return scores, best, False

    # ---------------------------------------------------------------- action
    def action(self, subject) -> tuple[str, str] | None:
        if self.predictor is None or self.predictor.predictor is None:
            return None
        base_obs = self.predictor.current_obs
        if base_obs is None:
            values = list(subject.world.last_values)
            meta = dict(subject.world.last_meta)
            base_obs = np.array([
                float(values[0]), float(values[1]), float(values[3]), float(values[4]),
                1.0 if meta.get("visible", False) else 0.0,
                min(float(meta.get("age_since_seen", 0.0)), 20.0) / 10.0,
                float(meta.get("contact_estimate", 0.0)), 1.0,
            ], dtype=np.float64)
        h = np.asarray(subject.core.h, dtype=np.float64)
        c = np.asarray(subject.core.c, dtype=np.float64)
        preds, zs = self.predictor.predictor.predict_all(base_obs, h, c)
        means = preds.mean(axis=1)                       # (5,4)
        disagreement = self.predictor.predictor.disagreement(preds)  # (5,)
        pred_ball = means[:, :2] * 64.0
        anchor = np.array([32.0, 32.0])
        pred_dist = np.linalg.norm(pred_ball - anchor, axis=1) / 64.0
        world_value = 1.0 - pred_dist
        base_score = (self.cfg["world_weight"] * world_value +
                      self.cfg["uncertainty_weight"] * disagreement)
        if not self.cfg["use_world"]:
            base_score = np.zeros(len(ACTIONS), dtype=np.float64)

        # CORE-READOUT-01: h/c -> predicted ball offset, only left/right.
        cognitive = np.zeros(len(ACTIONS), dtype=np.float64)
        readout_pred = float(self.readout.predict_state(h, c)) if (
            self.readout is not None and self.cfg["use_cognitive"]) else 0.0
        if readout_pred > 0:
            cognitive[ACTION_INDEX["PUSH_RIGHT"]] += self.cfg["cognitive_weight"] * min(1.0, abs(readout_pred))
        elif readout_pred < 0:
            cognitive[ACTION_INDEX["PUSH_LEFT"]] += self.cfg["cognitive_weight"] * min(1.0, abs(readout_pred))

        # Affect: exploration / risk / fatigue, not reward.
        affect_values = dict(self.ctx.get_shared("affect.values", {}) or {})
        exploration = float(self.ctx.get_shared("affect.exploration_bias", 0.25))
        risk = float(affect_values.get("risk_sensitivity", 0.5))
        fatigue = float(affect_values.get("fatigue", 0.0))
        affect_score = np.zeros(len(ACTIONS), dtype=np.float64)
        if self.cfg["use_affect"]:
            affect_score += self.cfg["affect_explore_weight"] * exploration * disagreement
            affect_score -= self.cfg["affect_risk_weight"] * risk * pred_dist
            affect_score[ACTION_INDEX["WAIT"]] += self.cfg["affect_fatigue_weight"] * fatigue

        cue = self._cue()
        scope = self._scope()
        memory_score, confidence = self._experience_scores(subject, scope, preds)
        memory_gate = float(confidence.max())
        memory_best = None
        if np.ptp(memory_score) and (memory_score == memory_score.max()).sum() == 1:
            memory_best = ACTIONS[int(np.argmax(memory_score))]
        self.stats["memory_gate_open" if np.any(memory_score) else "memory_gate_closed"] += 1

        # Volition: current goal priority.
        goal = str(self.ctx.get_shared("volition.goal", "explore"))
        goal_score = float(self.ctx.get_shared("volition.score", 0.0))
        volition_score = np.zeros(len(ACTIONS), dtype=np.float64)
        if self.cfg["use_volition"]:
            if goal == "explore":
                volition_score += self.cfg["volition_weight"] * np.maximum(
                    0.0, disagreement - disagreement.mean())
            else:
                volition_score[ACTION_INDEX["WAIT"]] += self.cfg["volition_weight"] * max(0.0, goal_score)

        # Concept/language: gated belief valence, only if real evidence exists.
        beliefs = dict(self.ctx.get_shared("concept.beliefs", {}) or {})
        confidences = dict(self.ctx.get_shared("concept.confidence", {}) or {})
        belief = float(beliefs.get("ball", 0.0))
        concept_conf = float(confidences.get("ball", 0.0))
        # A real language prior has belief but no intervention confidence yet.
        # Gate on the stronger of intervention confidence and belief magnitude;
        # both are logged separately, so the source remains auditable.
        concept_gate = max(concept_conf, min(1.0, abs(belief)))
        concept_score = np.zeros(len(ACTIONS), dtype=np.float64)
        if (self.cfg["use_concept"] and
                concept_gate >= self.cfg["concept_conf_threshold"] and
                abs(belief) > 1e-9):
            bias = self.cfg["concept_weight"] * belief * concept_gate
            for name in ("PUSH_LEFT", "PUSH_RIGHT", "PUSH_UP", "PUSH_DOWN"):
                concept_score[ACTION_INDEX[name]] += bias
            concept_score[ACTION_INDEX["WAIT"]] -= bias

        base = base_score + cognitive
        base_best = int(np.argmax(base))

        def resolve(memory_term):
            return self.resolve_scores(base, affect_score, memory_term, volition_score, concept_score)

        total, total_best, affect_blocked = resolve(memory_score)
        no_memory_scores, no_memory_best, _ = resolve(0.0)
        if affect_blocked:
            self.stats["affect_overrides_blocked"] += 1
        def select(best, scores):
            if not (self.cfg.get("observed_wait") and self.cfg.get("active_probe", True)
                    and subject.memory.enabled):
                return ACTIONS[best], "ranked"
            return observed_drift.choose_probe(
                self._state.get("drift", {}), scope, ACTIONS[best],
                sorted(ACTIONS[1:], key=lambda a: -scores[ACTION_INDEX[a]]), subject.clock.step,
                subject.memory.semantic.staleness_tau_steps)

        ranked = ACTIONS[total_best]
        chosen, reason = select(total_best, total)
        without_memory, _ = select(no_memory_best, no_memory_scores)
        module_bests = {
            "world_cognitive": ACTIONS[base_best],
            "affect": ACTIONS[int(np.argmax(affect_score))],
            "memory": memory_best or "—",
            "volition": ACTIONS[int(np.argmax(volition_score))],
            "concept": ACTIONS[int(np.argmax(concept_score))],
        }
        for name, vector in (("world_cognitive", base), ("affect", affect_score),
                             ("volition", volition_score), ("concept", concept_score)):
            if np.count_nonzero(vector == np.max(vector)) != 1:
                module_bests[name] = "—"
        conflict = len({v for v in module_bests.values() if v != "—"}) > 1
        if conflict:
            self.stats["conflicts"] += 1
        self.last = {
            "scores": {name: round(float(total[i]), 6) for i, name in enumerate(ACTIONS)},
            "base_scores": {name: round(float(base[i]), 6) for i, name in enumerate(ACTIONS)},
            "contributions": {
                "world": [round(float(v), 6) for v in base_score],
                "cognitive": [round(float(v), 6) for v in cognitive],
                "affect": [round(float(v), 6) for v in affect_score],
                "memory": [round(float(v), 6) for v in memory_score],
                "volition": [round(float(v), 6) for v in volition_score],
                "concept": [round(float(v), 6) for v in concept_score],
            },
            "base_best": ACTIONS[base_best],
            "chosen": chosen,
            "ranked": ranked,
            "selection_reason": reason,
            "chosen_without_memory": without_memory,
            "conflict": conflict,
            "module_bests": module_bests,
            "affect_override_blocked": affect_blocked,
            "memory_gate": round(memory_gate, 6),
            "memory_best": memory_best,
            "memory_gate_meta": json.loads(json.dumps(self._gate_meta)),
            "experience_scope": scope,
            "confidence_by_action": confidence.tolist(),
            "cue": cue,
            "readout_pred": round(readout_pred, 6),
            "readout_weight_norm": round(float(np.linalg.norm(self.readout.w)), 6)
            if self.readout is not None else 0.0,
            "goal": goal,
            "concept_belief": belief,
            "concept_confidence": concept_conf,
            "concept_gate": round(concept_gate, 6),
            "disagreement": [round(float(v), 6) for v in disagreement],
            "predicted_distance": [round(float(v), 6) for v in pred_dist],
        }
        idx = ACTION_INDEX[chosen]
        tid = transition_id(subject, int(subject.clock.step), chosen)
        contact = float(self.current_meta.get("contact_estimate", 0.0))
        self.pending = {
            "tid": tid,
            "action": chosen,
            "step": int(subject.clock.step),
            "cue": self._cue(),
            "contact": contact,
            "scope": scope,
            "observation": list(self.current_values),
            "prediction": means[idx].tolist(),
            "h": h.tolist(),
            "c": c.tolist(),
        }
        self.stats["steps"] += 1
        self.stats["chosen"] = chosen
        self.predictor.prepare_action(subject, chosen, idx, preds=preds, zs=zs)
        self._trace("decision", {**self.last, "tid": tid, "predictions": means.tolist(),
                                 "affect": affect_values, "observation": list(self.current_values)})
        ActionSlotAdapter.write(subject.registry.get("practice"), chosen)
        subject.shared.set("practice.action", chosen, source=self.name,
                           step=subject.clock.step, confidence=1.0)
        subject.shared.set("coordination.last", json.loads(json.dumps(self.last)),
                           source=self.name, step=subject.clock.step, confidence=1.0)
        return chosen, "coordination.weighted" if reason == "ranked" else "coordination." + reason

    # ------------------------------------------------------------ diagnostics
    def metrics(self) -> dict:
        base = super().metrics()
        base.update(dict(self.stats))
        base["last"] = json.loads(json.dumps(self.last))
        return base

    def snapshot(self) -> dict:
        payload = super().snapshot()
        payload["pending"] = self.pending
        payload["processed"] = sorted(self.processed)
        payload["last"] = self.last
        payload["awaiting_affect"] = self.awaiting_affect
        payload["stats"] = self.stats.copy()
        payload["config"] = self.cfg
        return payload

    def restore(self, payload: dict) -> None:
        super().restore(payload)
        self.awaiting_affect = payload.get("awaiting_affect")
        self.stats.update(payload.get("stats", {}))
        self.pending = payload.get("pending")
        self.processed = set(payload.get("processed") or [])
        self.last = payload.get("last") or {}
        if payload.get("config"):
            self.cfg.update(payload["config"])
