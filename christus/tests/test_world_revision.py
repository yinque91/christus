"""COORDINATION-02D directed tests.

No test writes memory or actions. Every branch restores the same 02C SOURCE
checkpoint and only toggles predictor/memory learning switches.
"""
from __future__ import annotations

from coordination02d.experiment import (positive_control, repeated_change,
                                        run_branch)


def test_prediction_error_changes_predictor_state():
    a = run_branch(reverse=True, predictor_on=True, memory_on=True)
    b = run_branch(reverse=True, predictor_on=False, memory_on=True)
    assert any(r["prediction_error"] > 0 for r in a["rows"])
    assert all(r["predictor_changed"] for r in a["rows"])
    assert all(r["predictor_updates_delta"] > 0 for r in a["rows"])
    assert not any(r["predictor_changed"] for r in b["rows"])
    assert all(r["predictor_updates_delta"] == 0 for r in b["rows"])


def test_behavior_recovery_does_not_require_predictor_update():
    a = run_branch(reverse=True, predictor_on=True, memory_on=True)
    b = run_branch(reverse=True, predictor_on=False, memory_on=True)
    assert [r["action"] for r in a["rows"]] == [r["action"] for r in b["rows"]]
    assert a["rows"][0]["good"] is False
    assert a["rows"][1]["good"] is True
    # This is the important negative causal result: B has no predictor update
    # but still recovers through other current state (memory/core) changes.
    assert b["rows"][1]["good"] is True


def test_frozen_memory_also_recovers():
    a = run_branch(reverse=True, predictor_on=True, memory_on=True,
                   memory_write=False)
    b = run_branch(reverse=True, predictor_on=False, memory_on=True,
                   memory_write=False)
    assert a["rows"][0]["action"] == "PUSH_LEFT"
    assert a["rows"][1]["action"] == "PUSH_RIGHT"
    assert b["rows"][1]["action"] == "PUSH_RIGHT"
    assert not any(r["memory_changed"] for r in a["rows"])
    assert not any(r["memory_changed"] for r in b["rows"])


def test_positive_control_does_not_decay_old_confidence():
    pos = positive_control()
    confs = [r["memory_gate_meta"]["entries"]["PUSH_LEFT"]["confidence"]
             for r in pos["rows"]
             if "PUSH_LEFT" in r["memory_gate_meta"]["entries"]]
    assert confs and min(confs) > 0.9
    assert pos["rows"][0]["memory_rule"]["support"] >= 6


def test_repeated_A_B_A_does_not_relearn_A():
    rev = repeated_change()
    a2 = [r for r in rev["rows"] if r["phase"] == "A2"]
    assert a2
    assert not any(r["good"] for r in a2)
    assert all(r["action"] == "PUSH_RIGHT" for r in a2)
