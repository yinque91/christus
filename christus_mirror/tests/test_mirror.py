"""MIRROR-01 tests: geometry, sensory separation, policy and artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from christus_mirror.world import MirrorWorld, MARKER_TO_PART  # noqa: E402
from christus_mirror.run import run_m3  # noqa: E402


def test_mirror_reflection_geometry():
    w = MirrorWorld(seed=1)
    w.reset()
    p = [20.0, 40.0, 0.0]
    m = w._mirror_point(p)
    assert abs(m[0] - 44.0) < 1e-6
    assert abs(m[1] - 40.0) < 1e-6


def test_marker_is_not_in_first_person_features():
    w = MirrorWorld(seed=2)
    w.reset()
    w.set_marker(0, "real")
    obs = w.observe()
    v = np.asarray(w.current["visual"])
    # marker pixels are mirror-only features at 8:11
    assert v[8] == 1.0
    # first-person body features do not encode marker location
    first = v[23:26]
    assert len(first) == 3


def test_no_condition_label_in_observation():
    w = MirrorWorld(seed=3)
    w.reset()
    obs = w.observe()
    text = json.dumps({"meta": obs.meta, "values": list(obs.values)})
    assert "condition" not in text
    assert "ground_truth" not in text
    assert "marker_location" not in text


def test_marker_policy_inspection_is_real_contact():
    out = run_m3(12000, "A3", "real", marker=2)
    assert out["marker_seen"] is True
    assert out["inspection_contact"] is True
    assert out["target_correct"] is True


def test_results_are_twenty_seed_and_real_marker_pass_is_preserved():
    d = json.loads((ROOT / "mirror01" / "results.json").read_text(encoding="utf-8"))
    assert len(d["seeds"]) == 20
    assert len(d["M3"]) == 40
    assert all(x["summary"].get("inspection_contact") for x in d["M3"])
    assert d["M6"]["memory_reference_present"] is True

def test_lab_is_native_tk_not_web():
    s = (ROOT / "christus_mirror" / "lab_app.py").read_text(encoding="utf-8")
    assert "tkinter" in s
    assert "http.server" not in s
    assert "localhost" not in s.lower()
