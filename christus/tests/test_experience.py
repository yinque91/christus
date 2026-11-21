"""Stage 02 only. One short contact trajectory and checkpoint fork."""
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from christus.coordination_runner import make_coordinated_subject, save_coordinated_checkpoint
from christus03.world import PhaseSpec
from wphr.core.event import Event


def test_observed_transitions_and_continuation(tmp_path):
    phases = [PhaseSpec("before", 0, 3, "S1"), PhaseSpec("changed", 3, 100, "T2")]
    subject, predictor, readout, coord, world = make_coordinated_subject(tmp_path, 82002, phases=phases)
    world.p[:] = (39, 32)
    world.v[:] = (-1.5, 0)
    for _ in range(3):
        subject.step()
    assert coord.processed and subject.memory.semantic.rules
    ck = save_coordinated_checkpoint(subject, world, "pending at physical regime change")
    assert coord.pending["tid"] == predictor.pending["transition_id"]
    assert "outcome" in coord.pending
    assert "hidden" not in ck.meta["world_state"]
    expected = []
    for _ in range(8):
        result = subject.step()
        expected.append((result.action, dict(coord.last["scores"]), world.last_values.copy(),
                         subject.core.h.copy(), subject.core.c.copy(), subject.affect_values(),
                         dict(subject.shared.get("volition.scores")), dict(subject.shared.get("concept.beliefs"))))
    subject.flush_events()
    with sqlite3.connect(subject.config.resolve_path("paths.database")) as db:
        records = [json.loads(x[0]) for x in db.execute("select payload from events order by timestamp,seq")]
    trace = os.getenv("COORD02_TRACE")
    if trace:
        Path(trace).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records), encoding="utf8")
    experiences = [r["observation"] for r in records if r["modality"] == "experience"]
    assert len(experiences) == len({e["tid"] for e in experiences}) == coord.stats["memory_records"]
    for e in experiences:
        np.testing.assert_allclose(e["residual"], np.subtract(e["actual"], e["prediction"]))
        assert e["outcome"]["action"] == e["action"]
    assert not any(m.last_error for m in subject.registry)
    final_predictor = predictor.predictor.snapshot()
    final_memory = subject.memory.semantic.snapshot()
    subject.close()
    subject, predictor, readout, coord, world = make_coordinated_subject(tmp_path, 82002, restore=True)
    try:
        assert subject.memory.semantic.snapshot() == ck.meta["extra"]["memory"]["semantic"]
        assert coord.pending["tid"] == predictor.pending["transition_id"]
        verification = []
        for action, scores, values, h, c, affect, goals, beliefs in expected:
            assert subject.step().action == action
            assert coord.last["scores"] == scores
            assert world.last_values == values
            np.testing.assert_array_equal(subject.core.h, h)
            np.testing.assert_array_equal(subject.core.c, c)
            assert (subject.affect_values(), subject.shared.get("volition.scores"),
                    subject.shared.get("concept.beliefs")) == (affect, goals, beliefs)
            verification.append({"step": subject.clock.step, "action": action, "scores": scores,
                                 "h_sha256": hashlib.sha256(h.tobytes()).hexdigest(),
                                 "c_sha256": hashlib.sha256(c.tobytes()).hexdigest()})
        assert final_predictor == predictor.predictor.snapshot()
        assert final_memory == subject.memory.semantic.snapshot()
        if trace:
            Path(trace).with_name("restore.json").write_text(json.dumps({
                "checkpoint_step": ck.step, "checkpoint_sha256": ck.sha256,
                "pending_id": ck.meta["modules"]["coordination"]["pending"]["tid"],
                "matched_continuation": verification,
                "final_predictor_equal": True, "final_semantic_memory_equal": True
            }, indent=2), encoding="utf8")
        assert not any(m.last_error for m in subject.registry)
    finally:
        subject.close()



def test_recurring_contact_after_drift_reversal(tmp_path):
    # Each trial starts at the same measured pose with a fresh visual history.
    # A forced WAIT consumes the next frame and leaves no pending decision
    # across the evaluator's repositioning boundary.
    subject, predictor, readout, coord, world = make_coordinated_subject(
        tmp_path, 82003, phases=[PhaseSpec("contact", 0, 100, "S1")])
    chosen = []
    try:
        for trial in range(24):
            world.p[:] = (39, 32)
            world.v[:] = (-2.5 if trial < 8 else 2.5, 0)
            world.tracker.reset()
            chosen.append(subject.step().action)
            subject.step(forced_action="WAIT")
            assert coord.pending is None
        subject.flush_events()
        with sqlite3.connect(subject.config.resolve_path("paths.database")) as db:
            events = [json.loads(x[0]) for x in db.execute(
                "select payload from events order by timestamp,seq")]
        decisions = [x["observation"] for x in events if x["modality"] == "decision"]
        records = [x["observation"] for x in events if x["modality"] == "experience"]
        trace = os.getenv("COORD02_TRACE")
        if trace:
            Path(trace).with_name("drift.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in events), encoding="utf8")
        assert len(decisions) == len(records) == 24
        assert len({d["experience_scope"] for d in decisions}) == 1
        assert any(any(d["contributions"]["memory"]) for d in decisions[1:])
        assert all(r["progress"] > 0 for r in records[:8])
        assert all(r["progress"] < 0 for r in records[8:])
        assert any(r["rule"]["support"] and r["rule"]["counter"] for r in records[8:])
        assert decisions[3]["chosen"] != decisions[3]["chosen_without_memory"]
        assert chosen[8] != chosen[9]
        assert records[8]["rule"]["counter"] == 1
        assert decisions[9]["confidence_by_action"][4] < decisions[8]["confidence_by_action"][4]
        assert any(any(d["contributions"]["cognitive"]) and any(d["contributions"]["affect"])
                   and any(d["contributions"]["memory"]) for d in decisions)
        assert any(x["prediction_error"] > 0 for x in events if x["event_type"] == "prediction")
        # Redelivery of a real feedback event must leave semantic evidence intact.
        prediction = next(x for x in events if x["event_type"] == "prediction")
        before = subject.memory.semantic.snapshot()
        coord.on_event(Event.from_dict(prediction))
        assert subject.memory.semantic.snapshot() == before
        assert coord.stats["duplicates"] == 1
        base = predictor.current_obs
        preds, _ = predictor.predictor.predict_all(base, subject.core.h, subject.core.c)
        scope = decisions[-1]["experience_scope"]
        scores, confidence = coord._experience_scores(subject, scope, preds)
        assert np.any(scores < 0)
        limit = np.linalg.norm(preds[:, :, :2] - .5, axis=2).std(axis=1)
        assert np.all(np.abs(scores) <= limit)
        assert not np.any(coord._experience_scores(subject, scope + "/unseen", preds)[0])
        subject.clock.step += 600
        assert max(coord._experience_scores(subject, scope, preds)[1]) < max(confidence)
        assert not any(m.last_error for m in subject.registry)
    finally:
        subject.close()
