"""COORDINATION-02B directed tests: self-experience -> future behavior.

Only the new autonomous protocol is run. Old 32 tests and old seeds are not
rerun. No test writes a semantic rule or manually sets support/counter.
"""
from __future__ import annotations

import json

import pytest

from coordination02b.experiment import (NEG, NEG_SEED, POS, POS_SEED, replay,
                                         revision, run_one)


def test_natural_negative_experience_changes_action(tmp_path):
    result = run_one(tmp_path / "negative", NEG, NEG_SEED)
    assert result["t0_action"] == "PUSH_UP"
    assert result["t2_action"] == "PUSH_RIGHT"
    assert result["t0"]["memory"] == [0.0] * 5
    rule = result["t1"]["rules"]["coord02/contact/0/-1/0/0"]["PUSH_UP"]
    assert rule["support"] == 0 and rule["counter"] >= 1
    assert result["t2"]["memory"][3] < 0
    on = replay(tmp_path / "negative", NEG, NEG_SEED, memory=True)
    off = replay(tmp_path / "negative", NEG, NEG_SEED, memory=False)
    assert on["action"] == result["t2_action"] == "PUSH_RIGHT"
    assert off["action"] == result["t0_action"] == "PUSH_UP"
    assert on["scores"]["PUSH_UP"] != off["scores"]["PUSH_UP"]


def test_natural_positive_experience_changes_score(tmp_path):
    result = run_one(tmp_path / "positive", POS, POS_SEED)
    assert result["good"] is True
    assert result["t0_action"] == "PUSH_LEFT"
    rule = result["t1"]["rules"]["coord02/contact/0/-1/0/0"]["PUSH_LEFT"]
    assert rule["support"] >= 1 and rule["counter"] == 0
    assert result["t2"]["memory"][1] > 0
    # Positive evidence changes the score in this trajectory, not the action.
    assert result["t2_action"] == result["t0_action"] == "PUSH_LEFT"
    on = replay(tmp_path / "positive", POS, POS_SEED, memory=True)
    off = replay(tmp_path / "positive", POS, POS_SEED, memory=False)
    assert on["action"] == off["action"] == "PUSH_LEFT"
    assert on["scores"]["PUSH_LEFT"] > off["scores"]["PUSH_LEFT"]


def test_environment_change_revises_old_weight(tmp_path):
    result = revision(tmp_path / "revision", POS_SEED)
    good = [t for t in result["trials"] if t["phase"] == "good"]
    bad = [t for t in result["trials"] if t["phase"] == "bad"]
    assert good and bad
    first_rule = good[-1]["rules"]["coord02/contact/0/-1/0/0"]["PUSH_LEFT"]
    assert first_rule["support"] >= 1 and first_rule["counter"] == 0
    assert first_rule["confidence"] == 1.0
    final_rule = result["final"]["rules"]["coord02/contact/0/-1/0/0"]["PUSH_LEFT"]
    assert final_rule["counter"] >= 1
    assert final_rule["confidence"] < first_rule["confidence"]
    assert result["final"]["action"] != "PUSH_LEFT"
