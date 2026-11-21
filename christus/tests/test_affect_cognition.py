"""CHRISTUS-AFFECT-02 tests: affect -> attention -> core -> readout -> action.

The tests load naturally induced LOW/HIGH affect snapshots onto one common
SOURCE checkpoint. They do not write affect values or actions.
"""
from __future__ import annotations

import numpy as np
import pytest

from affect02.experiment import induce, replay


@pytest.fixture(scope="module")
def snaps():
    return induce("low"), induce("high")


def _run(snaps, level, direct=False, target="T1"):
    low, high = snaps
    return replay(low_snapshot=low["snapshot"], high_snapshot=high["snapshot"],
                  level=level, target_name=target, direct_affect=direct)


def test_affect_changes_attention_when_enabled(snaps):
    lo = _run(snaps, "LOW")
    hi = _run(snaps, "HIGH")
    assert hi["attention_before"] > lo["attention_before"]
    assert hi["attention_before"] != lo["attention_before"]


def test_affect_changes_core_state_when_attention_changes(snaps):
    lo = _run(snaps, "LOW")
    hi = _run(snaps, "HIGH")
    h0, h1 = np.array(lo["h_after"]), np.array(hi["h_after"])
    c0, c1 = np.array(lo["c_after"]), np.array(hi["c_after"])
    assert np.linalg.norm(h1 - h0) > 0
    assert np.linalg.norm(c1 - c0) == 0  # c unchanged; only h carries attention effect
    cos = float(np.dot(h0, h1) / (np.linalg.norm(h0) * np.linalg.norm(h1)))
    assert cos < 1.0


def test_core_readout_receives_changed_core_state(snaps):
    lo = _run(snaps, "LOW")
    hi = _run(snaps, "HIGH")
    assert lo["readout_before"] == hi["readout_before"]
    assert hi["readout_after"] != lo["readout_after"]
    assert abs(hi["readout_after"] - lo["readout_after"]) > 0


def test_direct_affect_score_is_isolated(snaps):
    lo_off = _run(snaps, "LOW", direct=False)
    lo_on = _run(snaps, "LOW", direct=True)
    assert lo_off["contributions"]["affect"] == [0.0] * 5
    assert lo_on["contributions"]["affect"] != [0.0] * 5
    assert lo_off["action"] == lo_on["action"] == "PUSH_LEFT"
    assert lo_on["scores"] != lo_off["scores"]


def test_affect_counterfactual_starts_from_same_checkpoint(snaps):
    lo = _run(snaps, "LOW")
    hi = _run(snaps, "HIGH")
    assert lo["memory_hash"] == hi["memory_hash"]
    assert lo["predictor_hash"] == hi["predictor_hash"]
    assert lo["core_before_hash"] == hi["core_before_hash"]
    assert lo["affect_hash"] != hi["affect_hash"]


def test_core_readout_is_or_is_not_connected_to_final_action(snaps):
    lo = _run(snaps, "LOW")
    assert lo["readout_action_used"] is False
    assert lo["final_action_source"] == "Subject.action_hook -> CoordinationModule.action"
    # readout.predict_state is used, but its 2.7e-6 effect does not flip action.
    assert lo["readout_pred"] is not None
    assert lo["action"] == "PUSH_LEFT"
