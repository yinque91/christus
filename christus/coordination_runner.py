"""Runner for CHRISTUS-COORDINATION-01: one Subject, one action exit."""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

from wphr.config import Config
from wphr.memory.store import Store
from christus.core_readout import CoreActionReadout
from christus.coordination import CoordinationModule
from christus03.policy import Christus03Policy
from christus03.predictor import PredictorModule
from christus03.runner import ROOT, Christus03Subject, replay_consistency
from christus03.visual import VisualObservation
from christus03.world import HiddenParams, BallWorld, PhaseSpec, SealedWorldPort, smoke_phases

ACTIONS = ("WAIT", "PUSH_LEFT", "PUSH_RIGHT", "PUSH_UP", "PUSH_DOWN")


def build_config(run_dir: Path, seed: int) -> Config:
    cfg = Config.load(root=ROOT)
    for key in cfg.section("paths"):
        cfg.set("paths." + key, str((run_dir / cfg.get("paths." + key)).resolve()))
    cfg.set("world.id", "christus03_ball")
    cfg.set("runtime.seed", int(seed))
    cfg.set("runtime.auto_reset_episodes", False)
    cfg.set("runtime.status_every", 0)
    cfg.set("sensors.enabled", False)
    cfg.set("memory.enabled", True)
    cfg.set("memory.consolidate_every", 50)
    cfg.set("memory.retention.auto_archive_every", 0)
    cfg.set("planning.enabled", False)
    cfg.set("language.enabled", True)
    cfg.set("language.teacher_enabled", False)
    cfg.set("affect.ablation", "full")
    cfg.set("modules.disabled", ["world", "practice", "social"])
    cfg.set("scheduler.fast", 1)
    cfg.set("scheduler.affect", 1)
    cfg.set("scheduler.concept", 1)
    cfg.set("scheduler.slow", 1)
    cfg.set("safety.allowed_actions", list(ACTIONS))
    cfg.set("concepts.interaction_actions", ["PUSH_LEFT", "PUSH_RIGHT", "PUSH_UP", "PUSH_DOWN"])
    cfg.set("coordination", {
        "enabled": True,
        "world_weight": 1.0,
        "uncertainty_weight": 40.0,
        "cognitive_weight": 0.8,
        "affect_explore_weight": 0.15,
        "affect_risk_weight": 0.25,
        "affect_fatigue_weight": 0.08,
        "memory_weight": 0.45,
        "memory_conf_threshold": 0.20,
        "memory_external_reliability": 0.5,
        "memory_reality_gate": False,
        "memory_reality_margin": 0.05,
        "memory_reality_fraction": 0.25,
        "memory_conflict_discount": 0.5,
        "memory_allowed_sources": ["coordination.observed", "coordination.observed_wait"],
        "volition_weight": 0.20,
        "concept_weight": 0.30,
        "concept_conf_threshold": 0.20,
        "affect_override_margin": 1e-9,
    })
    cfg.ensure_dirs()
    return cfg


def save_coordinated_checkpoint(subject, world, note=""):
    """Evaluator owns the companion file. The subject NPZ stays observable-only."""
    ck = subject.save_checkpoint(note)
    tracker = world.tracker
    data = {"subject_sha256": ck.sha256, "physics": world.state(),
            "phases": [asdict(p) for p in world.phases],
            "combos": {k: asdict(v) for k, v in world.combos.items()},
            "visual": asdict(tracker.previous) if tracker.previous else None,
            "history": tracker._history}
    target = ck.path.with_suffix(".evaluator.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf8")
    temporary.replace(target)
    return ck


def make_coordinated_subject(run_dir: Path, seed: int, *, restore: bool = False,
                             checkpoint_id: str | None = None,
                             observed_wait: bool = False,
                             reality_gate: bool | None = None,
                             phases: list[PhaseSpec] | None = None,
                             steps: int = 100):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(run_dir, seed)
    world = BallWorld(seed=seed, phases=phases or smoke_phases(seed), combos=None)
    port = SealedWorldPort(world)
    store = Store(cfg.resolve_path("paths.database"))
    subject = Christus03Subject(cfg, world=port, store=store, seed=seed, persist=True)
    subject.initialise()
    predictor = PredictorModule(subject.registry.get("practice").ctx)
    subject.registry.register(predictor)
    predictor.attach(subject.core, Christus03Policy("A2", seed))
    readout = CoreActionReadout(subject.registry.get("practice").ctx)
    subject.registry.register(readout)
    readout.attach(subject.core, subject.registry.get("practice"))
    readout.enabled = True
    coordination = CoordinationModule(subject.registry.get("practice").ctx)
    if observed_wait:
        coordination.cfg["observed_wait"] = True
    subject.registry.register(coordination)
    coordination.attach(subject, predictor, readout)
    if reality_gate is not None:
        coordination.cfg["memory_reality_gate"] = bool(reality_gate)
        cfg.set("coordination.memory_reality_gate", bool(reality_gate))
    subject.action_hook = coordination.action
    subject.pre_core_hook = coordination.finish_experience
    if restore or checkpoint_id is not None:
        try:
            ck = subject.restore(checkpoint_id)
            data = json.loads(ck.path.with_suffix(".evaluator.json").read_text(encoding="utf8"))
            if data["subject_sha256"] != ck.sha256 or data["physics"]["t"] != ck.step:
                raise ValueError("Evaluator and subject checkpoints do not match")
            world = BallWorld.from_state(
                data["physics"], phases=[PhaseSpec(**x) for x in data["phases"]],
                combos={k: HiddenParams(**v) for k, v in data["combos"].items()})
            world.tracker._history = [tuple(pair) for pair in data["history"]]
            world.tracker.previous = VisualObservation(**data["visual"]) if data["visual"] else None
            world.last_visual = world.tracker.previous
            subject.world = SealedWorldPort(world)
            subject.world.restore(ck.meta["world_state"])
        except Exception:
            subject.close()
            raise
    if reality_gate is not None:
        coordination.cfg["memory_reality_gate"] = bool(reality_gate)
        cfg.set("coordination.memory_reality_gate", bool(reality_gate))
    return subject, predictor, readout, coordination, world


def record_row(subject, result, predictor, readout, coordination, world) -> dict:
    meta = dict(subject.world.last_meta)
    values = list(subject.world.last_values)
    return {
        **result.to_dict(),
        "arm": "COORD",
        "seed": int(subject.seed),
        "visible": bool(meta.get("visible", False)),
        "quality": float(meta.get("quality", 0.0)),
        "age_since_seen": float(meta.get("age_since_seen", 0.0)),
        "contact_estimate": float(meta.get("contact_estimate", 0.0)),
        "phase": world.phase_name,
        "obs_x": float(values[0]),
        "obs_y": float(values[1]),
        "obs_vx": float(values[3]),
        "obs_vy": float(values[4]),
        "legal_action": result.action in ACTIONS,
        "predictor": predictor.metrics(),
        "readout": readout.metrics(),
        "coordination": coordination.metrics(),
        "affect": subject.affect_values(),
        "volition": dict(subject.shared.get("volition.scores", {}) or {}),
        "goal": str(subject.shared.get("volition.goal", "")),
        "concept_beliefs": dict(subject.shared.get("concept.beliefs", {}) or {}),
        "concept_confidence": dict(subject.shared.get("concept.confidence", {}) or {}),
        "memory": subject.memory.stats(),
    }


def summarize(rows: list[dict], seed: int) -> dict:
    actions = [str(r["action"]) for r in rows]
    legal = sum(1 for r in rows if r.get("legal_action"))
    visible = sum(1 for r in rows if r.get("visible"))
    last_coord = rows[-1].get("coordination", {}).get("last", {}) if rows else {}
    return {
        "arm": "COORD",
        "seed": int(seed),
        "steps": len(rows),
        "action_counts": {a: actions.count(a) for a in ACTIONS},
        "action_validity_rate": legal / len(rows) if rows else None,
        "visible_rate": visible / len(rows) if rows else None,
        "occlusion_rate": (len(rows) - visible) / len(rows) if rows else None,
        "memory_records": rows[-1]["coordination"]["memory_records"] if rows else 0,
        "conflicts": sum(1 for r in rows if (r.get("coordination", {}).get("last") or {}).get("conflict")),
        "affect_overrides_blocked": sum(1 for r in rows if (r.get("coordination", {}).get("last") or {}).get("affect_override_blocked")),
        "readout_updates": max(
            [int((r.get("coordination", {}) or {}).get("readout_updates", 0)) for r in rows]
        ) if rows else 0,
        "memory_gate_rejected": max(
            [int((r.get("coordination", {}) or {}).get("memory_gate_rejected", 0)) for r in rows]
        ) if rows else 0,
        "memory_current_conflicts": sum(
            1 for r in rows
            if any((e or {}).get("conflict") for e in
                   (((r.get("coordination", {}) or {}).get("last") or {})
                    .get("memory_gate_meta") or {}).get("entries", {}).values())
        ),
        "memory_reality_gate": bool(
            ((last_coord.get("memory_gate_meta") or {}).get("reality_gate"))
        ) if last_coord else False,
        "last_coordination": last_coord,
        "final_memory": rows[-1].get("memory") if rows else None,
        "final_affect": rows[-1].get("affect") if rows else None,
        "final_goal": rows[-1].get("goal") if rows else None,
        "final_concept": rows[-1].get("concept_beliefs") if rows else None,
        "final_concept_confidence": rows[-1].get("concept_confidence") if rows else None,
    }


def run(out: Path, seed: int, steps: int = 100, *, reality_gate: bool = True
        ) -> tuple[dict, list[dict]]:
    run_dir = out / "COORD" / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    subject = None
    rows: list[dict] = []
    try:
        subject, predictor, readout, coordination, world = make_coordinated_subject(
            run_dir, seed, steps=steps, reality_gate=reality_gate)
        subject.config.save(run_dir / "config.json")
        with (run_dir / "events.jsonl").open("x", encoding="utf-8") as handle:
            for _ in range(steps):
                result = subject.step()
                row = record_row(subject, result, predictor, readout, coordination, world)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        save_coordinated_checkpoint(subject, world, "coordination run end")
        summary = summarize(rows, seed)
        summary["health"] = [h.to_dict() for h in subject.registry.health_report()]
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "actions.json").write_text(
            json.dumps([r["action"] for r in rows], ensure_ascii=False, indent=2),
            encoding="utf-8")
        return summary, rows
    except Exception:
        failure = {"seed": seed, "traceback": traceback.format_exc()}
        (run_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return failure, rows
    finally:
        if subject is not None:
            try:
                subject.close()
            except Exception:
                pass
