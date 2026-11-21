"""COORDINATION-02C directed tests: transfer, negative transfer, revision.

No test writes memory evidence; source experience comes only from Subject.step.
Old tests/seeds are not rerun.
"""
from __future__ import annotations

from coordination02c.experiment import (CENTER, SOURCE, TARGET, SEED, branch,
                                        evaluator_fork, revision, source)


def test_positive_transfer_level4(tmp_path):
    src = source(tmp_path / "source")
    assert src["source_rule"]["support"] == 5
    assert src["source_rule"]["counter"] == 0
    off = branch(tmp_path / "source", TARGET, memory=False)
    on = branch(tmp_path / "source", TARGET, memory=True)
    assert off["action"] == "PUSH_RIGHT"
    assert on["action"] == "PUSH_LEFT"
    assert on["memory"][1] > 0
    assert on["progress"] > off["progress"]
    fork_off = evaluator_fork(tmp_path / "source", "PUSH_RIGHT", TARGET)
    fork_on = evaluator_fork(tmp_path / "source", "PUSH_LEFT", TARGET)
    assert fork_on["progress"] > fork_off["progress"]
    assert off["scope"] == on["scope"] == "coord02/contact/0/-1/0/0"


def test_negative_transfer_is_detected(tmp_path):
    source(tmp_path / "source")
    off = branch(tmp_path / "source", TARGET, memory=False, reverse=True)
    on = branch(tmp_path / "source", TARGET, memory=True, reverse=True)
    assert off["action"] == "PUSH_RIGHT"
    assert on["action"] == "PUSH_LEFT"
    assert on["progress"] < off["progress"]
    assert on["memory"][1] > 0


def test_environment_revision(tmp_path):
    source(tmp_path / "source")
    rev = revision(tmp_path / "source")
    assert rev["trials"][0]["action"] == "PUSH_LEFT"
    assert rev["trials"][0]["good"] is False
    assert rev["final"]["action"] == "PUSH_RIGHT"
    rule = rev["final"]["rules"]["coord02/contact/0/-1/0/0"]["PUSH_LEFT"]
    assert rule["counter"] >= 1
    assert rule["confidence"] < 1.0
