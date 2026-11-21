"""CHRISTUS-AFFECT-01 directed tests.

These tests use the existing affect/memory modules only. They do not write a
semantic rule or choose an action; all conditions restore the same 02C SOURCE
checkpoint and toggle the existing causal paths.
"""
from __future__ import annotations

from affect01.experiment import condition, formation


def test_affect_state_changes_with_outcome_when_enabled():
    off = condition(affect_on=False, memory_on=True, direct_affect=False)
    on = condition(affect_on=True, memory_on=True, direct_affect=True)
    assert off["hashes"]["affect_before"] == off["hashes"]["affect_after"]
    assert on["hashes"]["affect_before"] != on["hashes"]["affect_after"]
    assert on["affect_after"]["frustration"] != off["affect_after"]["frustration"]


def test_affect_does_not_change_retrieval_or_action_in_2x2():
    a0 = condition(affect_on=False, memory_on=False, direct_affect=False)
    a1 = condition(affect_on=False, memory_on=True, direct_affect=False)
    a2 = condition(affect_on=True, memory_on=False, direct_affect=True)
    a3 = condition(affect_on=True, memory_on=True, direct_affect=True)
    assert a0["action"] == a2["action"] == "PUSH_RIGHT"
    assert a1["action"] == a3["action"] == "PUSH_LEFT"
    assert a3["contributions"]["affect"] != [0.0] * 5
    assert a1["contributions"]["affect"] == [0.0] * 5
    # Memory is the causal difference; affect contribution does not change the
    # memory contribution or the action in this target.
    assert a3["contributions"]["memory"][1] > 0
    assert abs(a3["contributions"]["memory"][1] - a1["contributions"]["memory"][1]) < 1e-3


def test_formation_importance_is_metadata_not_retrieval():
    on = formation(label="positive", state=(36.0, 30.0, 0.0, 0.0),
                   reverse=False, affect_on=True)
    off = formation(label="positive", state=(36.0, 30.0, 0.0, 0.0),
                    reverse=False, affect_on=False)
    assert on["rule"]["support"] == off["rule"]["support"]
    assert on["rule"]["confidence"] == off["rule"]["confidence"]
    assert on["rule"]["importance_scale"] == off["rule"]["importance_scale"]
    # The stored experience importance can differ, but it does not change rule
    # confidence/support or retrieval.
    assert on["experience"] is not None and off["experience"] is not None


def test_affect_checkpoint_restore_keeps_state():
    a = condition(affect_on=True, memory_on=True, direct_affect=True)
    b = condition(affect_on=True, memory_on=True, direct_affect=True)
    assert a["hashes"]["affect_before"] == b["hashes"]["affect_before"]
    assert a["hashes"]["memory_before"] == b["hashes"]["memory_before"]
    assert a["action"] == b["action"]
