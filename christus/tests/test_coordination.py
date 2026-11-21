"""Targeted causal/interfacing tests for CHRISTUS-COORDINATION-01.

No multi-seed formal experiment. Each test changes one internal variable and
checks the same decision chain responds (or the ablation removes the path).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from christus.coordination_runner import make_coordinated_subject, run, summarize
from christus03.predictor import PredictorModule
from christus03.world import ACTIONS, ACTION_INDEX
from wphr.core.event import EventType, Source


def _subject(tmp_path: Path, seed: int = 71000):
    subject, predictor, readout, coordination, world = make_coordinated_subject(
        tmp_path, seed, steps=20)
    return subject, predictor, readout, coordination, world


def _set_fake_predictions(predictor, means: np.ndarray, disagreement: np.ndarray):
    """means (5,2), disagreement (5,); build (5,5,4) and zero zs."""
    def fake_predict_all(base, h, c):
        preds = np.zeros((len(ACTIONS), 5, 4), dtype=np.float64)
        for a in range(len(ACTIONS)):
            preds[a, :, :2] = means[a] / 64.0
            preds[a, :, 2:] = 0.0
        # Inject disagreement by spreading member predictions on x.
        for a in range(len(ACTIONS)):
            preds[a, 0, 0] += disagreement[a]
            preds[a, 1, 0] -= disagreement[a]
        return preds, np.zeros((len(ACTIONS), 277), dtype=np.float64)

    predictor.predictor.predict_all = fake_predict_all


def _neutralize(coord, *, keep=()):
    weight_owner = {
        "affect_explore_weight": "affect",
        "affect_risk_weight": "affect",
        "affect_fatigue_weight": "affect",
        "memory_weight": "memory",
        "volition_weight": "volition",
        "concept_weight": "concept",
        "uncertainty_weight": "world",
    }
    for key, owner in weight_owner.items():
        if owner not in keep:
            coord.cfg[key] = 0.0
    for key in ("use_affect", "use_memory", "use_volition", "use_concept",
                "use_cognitive", "use_world"):
        coord.cfg[key] = key.replace("use_", "") in keep or key == "use_world"


def test_cognitive_state_changes_candidate_scores(tmp_path: Path):
    subject, predictor, readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("cognitive", "world"))
        coord.cfg["uncertainty_weight"] = 0.0
        _set_fake_predictions(predictor, np.full((5, 2), 32.0), np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        readout.w = np.ones(264) * 0.01
        readout.b = 0.0
        subject.core.h = np.ones(200)
        subject.core.c = np.ones(64)
        coord.action(subject)
        right = coord.last["contributions"]["cognitive"][ACTION_INDEX["PUSH_RIGHT"]]
        subject.core.h = -np.ones(200)
        subject.core.c = -np.ones(64)
        coord.action(subject)
        left = coord.last["contributions"]["cognitive"][ACTION_INDEX["PUSH_LEFT"]]
        assert right > 0
        assert left > 0
    finally:
        subject.close()


def test_world_prediction_changes_decision(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("world",))
        coordinator = np.full((5, 2), 58.0)
        coordinator[2] = [32.0, 32.0]   # PUSH_RIGHT near anchor
        _set_fake_predictions(predictor, coordinator, np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        coord.action(subject)
        assert coord.last["chosen"] == "PUSH_RIGHT"
        coordinator = np.full((5, 2), 58.0)
        coordinator[0] = [32.0, 32.0]
        _set_fake_predictions(predictor, coordinator, np.zeros(5))
        coord.action(subject)
        assert coord.last["chosen"] == "WAIT"
    finally:
        subject.close()


def test_affect_changes_scores_and_cannot_override_cognition(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("affect", "world"))
        coord.cfg["affect_explore_weight"] = 200.0
        coord.cfg["uncertainty_weight"] = 0.0
        subject.shared.set("affect.exploration_bias", 1.0, source="test", step=0)
        means = np.full((5, 2), 32.0)
        means[0] = [32.0, 32.0]       # WAIT is best by world value
        means[2] = [58.0, 58.0]       # right is far
        disagreement = np.zeros(5)
        disagreement[2] = 1.0         # right has high uncertainty (test scale)
        _set_fake_predictions(predictor, means, disagreement)
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        coord.action(subject)
        assert coord.last["contributions"]["affect"][2] > 0
        # The large affect term would flip to PUSH_RIGHT; the coordinator must
        # block that override and keep the cognitive/world choice.
        assert coord.last["chosen"] == "WAIT"
        assert coord.last["affect_override_blocked"] is True
    finally:
        subject.close()


def test_memory_gate_applicability(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("memory", "world"))
        coord.cfg["memory_conf_threshold"] = 0.2
        _set_fake_predictions(predictor, np.full((5, 2), 32.0), np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        for step in range(1, 5):
            subject.memory.record_outcome(cue="ball_near", action="PUSH_RIGHT",
                                          reward=1.0, step=step)
        coord.current_meta = {"visible": True, "contact_estimate": 1.0}
        coord.action(subject)
        assert coord.last["memory_gate"] > 0
        assert coord.last["contributions"]["memory"][ACTION_INDEX["PUSH_RIGHT"]] > 0
        coord.current_meta = {"visible": True, "contact_estimate": 0.0}
        coord.action(subject)
        assert coord.last["memory_gate"] == 0.0
        assert coord.last["memory_best"] is None
    finally:
        subject.close()


def test_memory_low_confidence_is_closed(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("memory", "world"))
        coord.cfg["memory_conf_threshold"] = 0.8
        _set_fake_predictions(predictor, np.full((5, 2), 32.0), np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        subject.memory.record_outcome(cue="ball_far", action="PUSH_LEFT",
                                      reward=1.0, step=1)
        coord.current_meta = {"visible": True, "contact_estimate": 0.0}
        coord.action(subject)
        assert coord.last["memory_gate"] < 0.8
        assert all(v == 0.0 for v in coord.last["contributions"]["memory"])
    finally:
        subject.close()


def test_volition_goal_changes_priority(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("volition",))
        coord.cfg["use_world"] = False
        _set_fake_predictions(predictor, np.full((5, 2), 32.0), np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        subject.shared.set("volition.goal", "rest", source="test", step=0)
        subject.shared.set("volition.score", 1.0, source="test", step=0)
        coord.action(subject)
        assert coord.last["contributions"]["volition"][ACTION_INDEX["WAIT"]] > 0
        assert coord.last["chosen"] == "WAIT"
        subject.shared.set("volition.goal", "explore", source="test", step=0)
        coord.action(subject)
        assert coord.last["contributions"]["volition"][ACTION_INDEX["WAIT"]] == 0.0
    finally:
        subject.close()


def test_concept_belief_changes_scores(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord, keep=("concept", "world"))
        _set_fake_predictions(predictor, np.full((5, 2), 32.0), np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        subject.shared.set("concept.beliefs", {"ball": 0.8}, source="test", step=0)
        subject.shared.set("concept.confidence", {"ball": 1.0}, source="test", step=0)
        coord.action(subject)
        assert coord.last["contributions"]["concept"][ACTION_INDEX["PUSH_RIGHT"]] > 0
        assert coord.last["contributions"]["concept"][ACTION_INDEX["WAIT"]] < 0
    finally:
        subject.close()


def test_real_language_claim_updates_concept_belief(tmp_path: Path):
    subject, _predictor, _readout, _coord, _world = _subject(tmp_path)
    try:
        for _ in range(2):
            subject.step()
        concept = subject.registry.get("concept_language")
        before = concept.belief("ball")
        subject.publish(subject._event(
            EventType.LANGUAGE_CLAIM.value, "language",
            source=Source.USER_PROVIDED.value,
            source_reliability=0.9,
            observation={"text": "ball good", "words": ["ball", "good"],
                         "polarity": 1, "speaker": "test_user"},
            provenance="test.real_user_claim",
        ))
        concept.update(subject.clock.step)
        after = concept.belief("ball")
        assert after != before
        _neutralize(_coord, keep=("concept", "world"))
        _set_fake_predictions(_predictor, np.full((5, 2), 32.0), np.zeros(5))
        _predictor.current_obs = np.zeros(8)
        _predictor.current_visible = True
        _coord.action(subject)
        assert _coord.last["contributions"]["concept"][ACTION_INDEX["PUSH_RIGHT"]] > 0
    finally:
        subject.close()


def test_prediction_error_reaches_affect(tmp_path: Path):
    subject, _predictor, _readout, _coord, _world = _subject(tmp_path)
    try:
        affect = subject.registry.get("affect")
        before = float(affect._state["impulses"]["frustration"])
        affect.on_event(SimpleNamespace(event_type="prediction", prediction_error=1.0))
        after = float(affect._state["impulses"]["frustration"])
        assert after > before
    finally:
        subject.close()


def test_prediction_event_is_persisted(tmp_path: Path):
    import sqlite3
    subject, _predictor, _readout, _coord, _world = _subject(tmp_path)
    try:
        for _ in range(4):
            subject.step()
        subject.flush_events()
        db = sqlite3.connect(str(subject.config.resolve_path("paths.database")))
        count = db.execute("select count(*) from events where event_type='prediction'").fetchone()[0]
        db.close()
        assert count >= 1
    finally:
        subject.close()


def test_same_feedback_is_processed_once(tmp_path: Path):
    subject, _predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        coord.pending = {"tid": "tid-1", "action": "PUSH_RIGHT",
                         "cue": "ball_near", "contact": 1.0,
                         "step": 1, "h": [0.0] * 200, "c": [0.0] * 64}
        before = subject.memory.semantic.updates
        coord._record_memory(SimpleNamespace(step=1))
        after_first = subject.memory.semantic.updates
        coord._record_memory(SimpleNamespace(step=1))
        after_second = subject.memory.semantic.updates
        assert after_first == before + 1
        assert after_second == after_first
        assert coord.stats["duplicates"] == 1
    finally:
        subject.close()


def test_module_disable_removes_path(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _set_fake_predictions(predictor, np.full((5, 2), 32.0), np.zeros(5))
        predictor.current_obs = np.zeros(8)
        predictor.current_visible = True
        coord.cfg.update({
            "use_affect": False, "use_memory": False,
            "use_volition": False, "use_concept": False,
        })
        coord.action(subject)
        contributions = coord.last["contributions"]
        assert all(v == 0.0 for v in contributions["affect"])
        assert all(v == 0.0 for v in contributions["memory"])
        assert all(v == 0.0 for v in contributions["volition"])
        assert all(v == 0.0 for v in contributions["concept"])
    finally:
        subject.close()


def test_all_modules_share_subject_state(tmp_path: Path):
    subject, predictor, readout, coord, _world = _subject(tmp_path)
    try:
        assert predictor.core is subject.core
        assert readout.core is subject.core
        assert coord.subject is subject
        assert predictor.ctx is readout.ctx is coord.ctx
    finally:
        subject.close()


def test_deterministic_replay_same_seed(tmp_path: Path):
    runs = []
    for label in ("a", "b"):
        run_dir = tmp_path / label
        subject, _predictor, _readout, _coord, _world = make_coordinated_subject(
            run_dir, 71000, steps=5)
        rows = []
        try:
            for _ in range(5):
                result = subject.step()
                rows.append((result.action, round(result.reward, 8),
                             round(result.sim_time, 8)))
        finally:
            subject.close()
        runs.append(rows)
    assert runs[0] == runs[1]


def test_checkpoint_restore_coordination_state(tmp_path: Path):
    subject, predictor, readout, coord, _world = _subject(tmp_path)
    run_dir = tmp_path
    try:
        for _ in range(5):
            subject.step()
        predictor_snapshot = predictor.predictor.snapshot()
        readout_w = readout.w.copy()
        readout_b = float(readout.b)
        memory_rules = json.loads(json.dumps(subject.memory.semantic.rules))
        affect_values = dict(subject.affect_values())
        concept = dict(subject.shared.get("concept.beliefs", {}) or {})
        subject.save_checkpoint("coordination restore test")
        subject.close()
        subject = None
        subject2, predictor2, readout2, coord2, _w2 = make_coordinated_subject(
            run_dir, 71000, restore=True, steps=5)
        try:
            assert predictor2.predictor.snapshot()["updates"] == predictor_snapshot["updates"]
            assert np.allclose(readout2.w, readout_w)
            assert abs(float(readout2.b) - readout_b) < 1e-12
            assert json.loads(json.dumps(subject2.memory.semantic.rules)) == memory_rules
            assert subject2.affect_values() == affect_values
            assert dict(subject2.shared.get("concept.beliefs", {}) or {}) == concept
            assert coord2.last == coord.last
        finally:
            subject2.close()
    finally:
        if subject is not None:
            subject.close()
