"""CHRISTUS-COORDINATION-02A directed tests: current reality > past memory.

These are deterministic gate tests, not a multi-seed experiment. They use the
existing coordination subject and memory APIs; no old seed is rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from christus.coordination_runner import (make_coordinated_subject,
                                          save_coordinated_checkpoint)
from christus03.world import ACTIONS, ACTION_INDEX

SCOPE = "coord02/contact/0/0/0/0"
SOURCE = "coordination.observed"


def _subject(tmp_path: Path, seed: int = 73000):
    subject, predictor, readout, coord, world = make_coordinated_subject(
        tmp_path, seed, reality_gate=True)
    return subject, predictor, readout, coord, world


def _preds(means: np.ndarray):
    preds = np.zeros((len(ACTIONS), 5, 4), dtype=np.float64)
    preds[:, :, :2] = means[:, None, :]
    # Member spread is needed for the existing uncertainty-band cap; the mean
    # remains the requested candidate outcome.
    preds[:, 0, 0] += 0.02
    preds[:, 1, 0] -= 0.02
    return preds, np.zeros((len(ACTIONS), 277), dtype=np.float64)


def _install_fake(predictor, means):
    preds, zs = _preds(means)
    predictor.predictor.predict_all = lambda base, h, c: (preds, zs)
    predictor.current_obs = np.zeros(8)
    predictor.current_visible = True
    return preds, zs


def _positive_rule(subject, action="PUSH_RIGHT", n=3, source=SOURCE):
    for step in range(1, n + 1):
        subject.memory.semantic.record_evidence(
            SCOPE, action, 1.0, step=step, source=source)


def _neutralize(coord):
    for key in ("affect_explore_weight", "affect_risk_weight", "affect_fatigue_weight",
                "volition_weight", "concept_weight", "uncertainty_weight"):
        coord.cfg[key] = 0.0
    coord.cfg["use_affect"] = False
    coord.cfg["use_volition"] = False
    coord.cfg["use_concept"] = False
    coord.cfg["use_cognitive"] = False


def test_a_applicable_experience_changes_score(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        _positive_rule(subject)
        preds, _ = _preds(np.full((5, 2), 0.5))
        scores, confidence = coord._experience_scores(subject, SCOPE, preds)
        assert scores[ACTION_INDEX["PUSH_RIGHT"]] > 0
        assert confidence[ACTION_INDEX["PUSH_RIGHT"]] > 0
        assert coord.last == {}  # score query is side-effect free
    finally:
        subject.close()


def test_b_bad_outcome_does_not_reinforce_action(tmp_path: Path):
    subject, _predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        for step in range(1, 4):
            subject.memory.semantic.record_evidence(
                SCOPE, "PUSH_RIGHT", -1.0, step=step, source=SOURCE)
        preds, _ = _preds(np.full((5, 2), 0.5))
        scores, _ = coord._experience_scores(subject, SCOPE, preds)
        assert scores[ACTION_INDEX["PUSH_RIGHT"]] < 0
        assert not np.any(scores > 0)
    finally:
        subject.close()


def test_c_environment_change_revises_old_rule(tmp_path: Path):
    subject, _predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        _positive_rule(subject, n=3)
        preds, _ = _preds(np.full((5, 2), 0.5))
        before, _ = coord._experience_scores(subject, SCOPE, preds)
        for step in range(10, 13):
            subject.memory.semantic.record_evidence(
                SCOPE, "PUSH_RIGHT", -1.0, step=step, source=SOURCE)
        after, conf = coord._experience_scores(subject, SCOPE, preds)
        assert before[ACTION_INDEX["PUSH_RIGHT"]] > 0
        assert after[ACTION_INDEX["PUSH_RIGHT"]] <= 0
        rule = subject.memory.semantic.rules[SCOPE]["PUSH_RIGHT"]
        assert rule["support"] == 3 and rule["counter"] == 3
    finally:
        subject.close()


def test_d_other_context_is_rejected(tmp_path: Path):
    subject, _predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        _positive_rule(subject)
        preds, _ = _preds(np.full((5, 2), 0.5))
        scores, _ = coord._experience_scores(subject, SCOPE + "/other", preds)
        assert not np.any(scores)
    finally:
        subject.close()


def test_gate_off_removes_all_memory_effect(tmp_path: Path):
    subject, _predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        _positive_rule(subject)
        preds, _ = _preds(np.full((5, 2), 0.5))
        coord.cfg["use_memory"] = True
        on, _ = coord._experience_scores(subject, SCOPE, preds)
        coord.cfg["use_memory"] = False
        off, _ = coord._experience_scores(subject, SCOPE, preds)
        assert np.any(on != 0)
        assert not np.any(off)
    finally:
        subject.close()


def test_action_feedback_is_not_evidence(tmp_path: Path):
    subject, _predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        # Legacy action_feedback, including a zero-reward "support" row, must be
        # rejected before it can bias the current action.
        subject.memory.semantic.record_evidence(
            SCOPE, "PUSH_RIGHT", 0.0, step=1, source="action_feedback")
        preds, _ = _preds(np.full((5, 2), 0.5))
        scores, confidence = coord._experience_scores(subject, SCOPE, preds)
        assert not np.any(scores)
        assert coord._gate_meta["rejected_sources"]
        assert "action_feedback" in json.dumps(coord._gate_meta)
    finally:
        subject.close()


def test_current_world_beats_positive_memory(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        means = np.full((5, 2), 0.95)
        means[ACTION_INDEX["WAIT"]] = [0.5, 0.5]  # current reality: WAIT is best
        _install_fake(predictor, means)
        coord.current_values = [0.5, 0.5, 1.0, 0.0, 0.0]
        coord.current_meta = {"visible": True, "contact_estimate": 1.0}
        _positive_rule(subject)
        preds, zs = _preds(means)
        coord.action(subject)
        meta = coord.last["memory_gate_meta"]["entries"]["PUSH_RIGHT"]
        assert meta["conflict"] is True
        assert coord.last["chosen"] == "WAIT"
    finally:
        subject.close()


def test_current_world_beats_negative_memory(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        means = np.full((5, 2), 0.95)
        means[ACTION_INDEX["PUSH_RIGHT"]] = [0.5, 0.5]  # current reality: right best
        _install_fake(predictor, means)
        coord.current_values = [0.5, 0.5, 1.0, 0.0, 0.0]
        coord.current_meta = {"visible": True, "contact_estimate": 1.0}
        for step in range(1, 4):
            subject.memory.semantic.record_evidence(
                SCOPE, "PUSH_RIGHT", -1.0, step=step, source=SOURCE)
        coord.action(subject)
        meta = coord.last["memory_gate_meta"]["entries"]["PUSH_RIGHT"]
        assert meta["conflict"] is True
        assert coord.last["chosen"] == "PUSH_RIGHT"
    finally:
        subject.close()


def test_history_changes_action_when_reality_ambiguous(tmp_path: Path):
    subject, predictor, _readout, coord, _world = _subject(tmp_path)
    try:
        _neutralize(coord)
        means = np.full((5, 2), 0.5)  # ambiguous current world
        _install_fake(predictor, means)
        coord.current_values = [0.5, 0.5, 1.0, 0.0, 0.0]
        coord.current_meta = {"visible": True, "contact_estimate": 1.0}
        _positive_rule(subject, action="PUSH_DOWN", n=3)
        coord.cfg["use_memory"] = True
        coord.action(subject)
        assert coord.last["chosen"] == "PUSH_DOWN"
        coord.cfg["use_memory"] = False
        subject.clock.step += 1
        coord.action(subject)
        assert coord.last["chosen"] == "WAIT"
    finally:
        subject.close()


def test_checkpoint_restores_gate_state(tmp_path: Path):
    subject, predictor, _readout, coord, world = _subject(tmp_path)
    try:
        _neutralize(coord)
        _positive_rule(subject)
        preds, _ = _preds(np.full((5, 2), 0.5))
        before, _ = coord._experience_scores(subject, SCOPE, preds)
        save_coordinated_checkpoint(subject, world, "reality gate checkpoint")
        rules = json.loads(json.dumps(subject.memory.semantic.rules))
        subject.close()
        subject = None
        subject2, _p2, _r2, coord2, _w2 = make_coordinated_subject(
            tmp_path, 73000, restore=True, reality_gate=True)
        try:
            assert json.loads(json.dumps(subject2.memory.semantic.rules)) == rules
            after, _ = coord2._experience_scores(subject2, SCOPE, preds)
            assert np.allclose(before, after)
            assert coord2.cfg["memory_reality_gate"] is True
        finally:
            subject2.close()
    finally:
        if subject is not None:
            subject.close()


def test_body_error_is_not_fabricated(tmp_path: Path):
    _subject_obj, _predictor, _readout, _coord, world = _subject(tmp_path)
    try:
        assert not hasattr(world, "body")
        assert not hasattr(world, "pred_body")
        truth = world.ground_truth()
        assert "body" not in truth and "body_error" not in truth
    finally:
        _subject_obj.close()
