"""The subject: one continuous digital subject and its step loop.

Loop order per step (documented so results stay interpretable):

1. advance the clock;
2. world -> observation event (a measurement, never ground truth);
3. modules react to the observation; M1 reports its one-step prediction error,
   which is published as its own derived event;
4. features are assembled from observation + affect + module statistics and fed
   to the SM-CTRN core (one Euler update);
5. modules due on the scheduler (M2 selects the action, M7 updates the goal);
6. the world executes the action; an outcome event carries reward and state;
7. M2 updates its value estimate from the outcome.

Events are buffered and written to SQLite in batches; on pause/stop/checkpoint
the buffer is flushed first.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from wphr.cognition.smctrn import SparseModularCTRN
from wphr.config import Config
from wphr.core.bus import EventBus
from wphr.core.clock import Clock
from wphr.core.event import Event, EventType, PrivacyClass, Source
from wphr.core.module import ModuleContext
from wphr.core.registry import ModuleRegistry
from wphr.core.scheduler import Scheduler
from wphr.core.state import SharedState
from wphr.core.subject_state import SubjectState
from wphr.logging_setup import AuditLog, get_logger, setup_logging
from wphr.memory.checkpoint import Checkpoint, CheckpointManager
from wphr.memory.layers import MemoryStore
from wphr.memory.store import Store
from wphr.modules import MODULE_CLASSES
from wphr.planning import Planner
from wphr.runtime.control import ControlFile
from wphr.runtime.features import assemble_inputs, build_percept, percept_to_features
from wphr.sensors import SensorSuite
from wphr.util.ids import new_id, utc_iso, utc_now
from wphr.version import __version__
from wphr.world import create_world

HARDWARE_FLAGS = (
    "real_camera",
    "real_microphone",
    "screen_capture",
    "keyboard_capture",
    "network_access",
    "private_files",
    "robot_actuators",
    "paid_services",
)


class SafetyError(RuntimeError):
    """Raised when configuration asks for a capability that is not permitted."""


@dataclass
class StepResult:
    step: int
    sim_time: float
    action: str
    reward: float
    reached: bool
    prediction_error: float
    affect: dict[str, float]
    goal: str
    h_norm: float
    faults: int
    cue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "sim_time": round(self.sim_time, 4),
            "action": self.action,
            "reward": round(self.reward, 4),
            "reached": self.reached,
            "prediction_error": round(self.prediction_error, 6),
            "affect": {k: round(v, 4) for k, v in self.affect.items()},
            "goal": self.goal,
            "cue": self.cue,
            "h_norm": round(self.h_norm, 6),
            "faults": self.faults,
        }


@dataclass
class RunSummary:
    started_at: float
    finished_at: float
    start_step: int
    end_step: int
    steps: int
    state: str
    events_written: int
    checkpoints: int = 0
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "steps": self.steps,
            "state": self.state,
            "events_written": self.events_written,
            "checkpoints": self.checkpoints,
            "wall_seconds": round(self.wall_seconds, 3),
            "steps_per_second": round(self.steps / self.wall_seconds, 2) if self.wall_seconds else 0.0,
            "notes": self.notes,
        }


class Subject:
    """A persistent digital subject: config + core + modules + memory + world."""

    def __init__(
        self,
        config: Config,
        *,
        world: Any | None = None,
        store: Store | None = None,
        seed: int | None = None,
        persist: bool = True,
        audit: bool = True,
        log_level: str | None = None,
        teacher: Any | None = None,
    ) -> None:
        self.config = config
        self._check_safety()
        setup_logging(
            config.resolve_path("paths.logs"),
            log_level or config.get("runtime.log_level", "INFO"),
            run_id=new_id("")[:6],
            console=bool(config.get("runtime.console_log", False)),
        )
        self.log = get_logger("subject")
        self.seed = int(seed if seed is not None else config.get("runtime.seed"))
        self.rng = np.random.default_rng(self.seed)
        self.clock = Clock(dt=float(config.get("runtime.dt")))
        self.bus = EventBus(history_size=256)
        self.shared = SharedState()
        self.subject_state = SubjectState()
        self.episode_id = new_id("ep_")
        self.persist = persist
        self._pending: list[Event] = []
        self._events_written = 0
        self._closed = False

        self.audit = (
            AuditLog(config.resolve_path("paths.logs") / "audit.jsonl") if audit else None
        )
        self.store = store
        if self.store is not None:
            self.store.init()
        self.checkpoints = (
            CheckpointManager(self.store, config) if (self.store is not None and persist) else None
        )

        self.core = SparseModularCTRN(
            config.section("core"),
            seed=self.seed,
            dt=float(config.get("runtime.dt")),
            finite_policy=config.get("runtime.finite_policy"),
        )
        self.memory = MemoryStore(self.store if persist else None, config)
        ctx = ModuleContext(
            config=config,
            clock=self.clock,
            bus=self.bus,
            shared=self.shared,
            rng=self.rng,
            audit=self.audit,
            memory=self.memory,
        )
        self.registry = ModuleRegistry()
        for module_cls in MODULE_CLASSES:
            self.registry.register(module_cls(ctx))
        missing = self.registry.missing_required()
        if missing:
            raise RuntimeError(f"missing required modules: {missing}")
        # ablation support: a module listed here keeps its slot (health = DISABLED)
        # but receives no events and does not update.
        disabled_cfg = config.get("modules.disabled", []) or []
        if isinstance(disabled_cfg, str):
            disabled_cfg = [disabled_cfg]
        for name in disabled_cfg:
            if name in self.registry:
                self.registry.get(name).enabled = False
        self.disabled_modules = list(disabled_cfg)

        # modules observe every event; the bus returns emitted events for persistence
        self.bus.subscribe("*", self.registry.dispatch)

        self.scheduler = Scheduler(config.section("scheduler"))
        world_id = config.get("world.id", "demo")
        self.world = world or create_world(
            world_id, seed=self.seed, config=config.get(f"world.{world_id}", {})
        )
        self.suite: SensorSuite | None = None
        if config.get("sensors.enabled", False) and hasattr(self.world, "visible_cells"):
            self.suite = SensorSuite(
                config.section("sensors"), np.random.default_rng(self.seed + 7919)
            )
            self.suite.start()
            if self.audit:
                self.audit.record(
                    "sensors.started", sensors=sorted(self.suite.sensors), simulated=True
                )
        self._check_action_whitelist()
        self.planner = Planner(config)
        self.control = ControlFile(config.resolve_path("paths.control"))
        self._run_id: str | None = None
        self._flush_every = 50
        self.last_offline_gap_s: float = 0.0
        self.teacher = teacher
        # Optional post-core action readout hook. ``None`` keeps the historical
        # path bit-identical: the action is exactly ``practice.action``.
        self.action_hook: Callable[["Subject"], tuple[str, str] | None] | None = None
        # Optional hook run immediately before core.step. CHRISTUS-03 A4 uses
        # it to clear the recurrent state so the next state is formed only from
        # the current observation; ``None`` keeps the historical path unchanged.
        self.pre_core_hook: Callable[["Subject"], None] | None = None
        self._produce_every = int(config.get("language.produce_every", 50) or 50)
        self._archive_every = int(config.get("memory.retention.auto_archive_every", 0) or 0)
        self._last_production: Any = None
        if self.audit:
            self.audit.record(
                "subject.created",
                subject_id=self.subject_state.subject_id,
                seed=self.seed,
                config_hash=config.hash(),
                world=world_id,
            )

    # ----------------------------------------------------------------- safety
    def _check_safety(self) -> None:
        blocked = [flag for flag in HARDWARE_FLAGS if self.config.get(f"safety.{flag}", False)]
        if blocked:
            raise SafetyError(
                "configuration enables capabilities that require explicit user approval and "
                f"have no implemented adapter yet: {blocked}"
            )

    def _check_action_whitelist(self) -> None:
        """Refuse to run if the world exposes an action outside the whitelist."""
        allowed = self.config.get("safety.allowed_actions", None)
        if allowed is None:
            return
        permitted = {str(action) for action in allowed}
        actions = {str(action) for action in self.world.actions()}
        illegal = actions - permitted
        if illegal:
            raise SafetyError(f"world exposes actions outside the safety whitelist: {illegal}")

    # -------------------------------------------------------------- lifecycle
    def initialise(self) -> "Subject":
        world_id = str(self.config.get("world.id", "demo"))
        if self.store is not None and self.persist:
            crashed = self.store.recover_runs()
            if crashed:
                self.log.warning("marked %d previously running run(s) as crashed", crashed)
            if bool(self.config.get("memory.retention.prune_on_start", False)):
                max_events = int(self.config.get("memory.retention.max_events", 200000))
                removed = self.store.prune_events(max_events)
                if removed:
                    self.log.warning("retention policy removed %d old events", removed)
            self.episode_id = self.store.start_episode(
                episode_index=self.subject_state.episode_index,
                world_id=world_id,
                seed=self.seed,
                config_hash=self.config.hash(),
                notes=f"v{__version__}",
            )
            self._run_id = self.store.start_run(
                version=__version__,
                config_hash=self.config.hash(),
                seed=self.seed,
                args={"world": world_id, "persist": True},
            )
        self.clock.reset(0)
        self.subject_state.step = 0
        self.subject_state.sim_time = 0.0
        return self

    def close(self, status: str = "completed") -> None:
        if self._closed:
            return
        self.flush_events()
        if self.store is not None and self.persist:
            self.store.end_episode(self.episode_id)
            if self._run_id:
                self.store.finish_run(self._run_id, steps=self.clock.step, status=status)
        if self.audit:
            self.audit.record(
                "subject.closed", subject_id=self.subject_state.subject_id, step=self.clock.step
            )
        self._closed = True

    # ------------------------------------------------------------------- step
    def step(self, forced_action: str | None = None) -> StepResult:
        self.clock.tick()
        step = self.clock.step
        obs = self.world.observe()
        readings = (
            self.suite.sample_all(self.world, self.clock.sim_time) if self.suite is not None else {}
        )
        percept = build_percept(
            observation_values=list(obs.values),
            energy=float(obs.energy),
            readings=readings,
        )
        # Optional world-provided derived fields. Old worlds do not set these,
        # so their observation path is unchanged. CHRISTUS-03 uses them to make
        # the visual ball cue/scent explicit to ConceptLanguage/Practice.
        meta = obs.meta or {}
        if "cue" in meta:
            percept["cue"] = str(meta["cue"])
        if "scent" in meta:
            scent = meta["scent"] or (0.0, 0.0)
            percept["scent"] = (float(scent[0]), float(scent[1]))
        payload = obs.to_payload()
        payload["derived"] = {
            "cue": percept["cue"],
            "patch": list(percept["patch"]),
            "depth": percept["depth"],
            "scent": list(percept["scent"]),
            "loudness": percept["loudness"],
            "agent": percept.get("agent"),
            "valid_ratio": percept["valid_ratio"],
            "missing": list(percept["missing"]),
        }
        if readings:
            payload["sensor_status"] = {
                sensor_id: {
                    "valid": reading.is_valid,
                    "confidence": round(reading.confidence, 4),
                    "missing_reason": reading.missing_reason,
                }
                for sensor_id, reading in readings.items()
            }
        self.publish(
            self._event(
                EventType.OBSERVATION.value,
                obs.modality,
                source=Source.SIMULATED.value,
                source_device=obs.source_device,
                confidence=obs.confidence * percept["valid_ratio"]
                if readings
                else obs.confidence,
                observation=payload,
                provenance=f"world.{self.world.world_id}.observe",
            )
        )

        world_module = self.registry.get("world")
        error = float(world_module.last_prediction_error)
        if world_module.events_seen > 1:
            self.publish(
                self._event(
                    EventType.INTERNAL.value,
                    "derived",
                    observation={"prediction_error": error},
                    prediction_error=error,
                    provenance="world.one_step_predictor",
                    confidence=world_module.confidence(),
                )
            )

        for group in self.scheduler.due(step):
            self.scheduler.time_group(group, self.registry.update_group, group, step)

        if self.teacher is not None:
            claim = self.teacher.claim(percept["cue"], step)
            if claim is not None:
                self.publish(
                    self._event(
                        EventType.LANGUAGE_CLAIM.value,
                        "language",
                        source=Source.USER_PROVIDED.value,
                        source_device="teacher",
                        source_reliability=float(claim.get("reliability", 0.7)),
                        confidence=float(claim.get("reliability", 0.7)),
                        observation={
                            "text": claim["text"],
                            "words": claim["words"],
                            "polarity": claim["polarity"],
                            "speaker": claim.get("speaker", "teacher"),
                        },
                        provenance="teacher.claim",
                    )
                )

        features = self._features(percept, error)
        # affect -> attention: scales how strongly the percept drives the core
        attention = float(self.shared.get("affect.attention_gain", 1.0))
        inputs = assemble_inputs(self.core, features) * attention
        if self.pre_core_hook is not None:
            self.pre_core_hook(self)
        metrics = self.core.step(inputs)

        action: str
        provenance: str
        if forced_action is not None:
            action, provenance = str(forced_action), "experiment.forced"
        else:
            goal = str(self.shared.get("volition.goal", "explore"))
            planned = None
            if self.planner.enabled:
                if self.planner.handles(goal):
                    if self.planner.should_repropose(percept):
                        self.planner.propose(goal, percept, step)
                    planned = self.planner.next_action()
                elif self.planner.plan is not None:
                    self.planner.abandon()
            if planned is not None:
                action, provenance = planned, f"planner.{goal}"
            else:
                action = str(self.shared.get("practice.action", "wait"))
                provenance = "practice.contextual_epsilon_greedy"
                # The readout runs after core.step, so it sees the current h/c.
                # It may only re-rank Practice's already-computed scores; an
                # empty/zero readout returns None and leaves the path unchanged.
                if self.action_hook is not None:
                    override = self.action_hook(self)
                    if override is not None:
                        action, provenance = override
        affect = self.affect_values()
        self.publish(
            self._event(
                EventType.ACTION.value,
                "motor",
                action={
                    "name": action,
                    "goal": self.shared.get("volition.goal", ""),
                    "cue": percept["cue"],
                },
                provenance=provenance,
            )
        )
        result = self.world.execute(action)
        if forced_action is None:
            self.planner.observe(action, result)
        self.publish(
            self._event(
                EventType.OUTCOME.value,
                "proprioception",
                actual_outcome=result.to_payload(),
                provenance=f"world.{self.world.world_id}.execute",
            )
        )
        self.memory.remember_working(
            {
                "step": step,
                "cue": percept["cue"],
                "action": action,
                "reward": float(result.reward),
            }
        )
        if self.memory.enabled and step - self.memory.last_consolidate_step >= self.memory.consolidate_every:
            self.memory.maybe_consolidate(
                step=step,
                subject_id=self.subject_state.subject_id,
                module_metrics={
                    "actions": self.registry.get("practice").metrics().get("actions", 0),
                    "goal": self.shared.get("volition.goal", ""),
                },
            )

        # world -> language: name a cue once its concept is confident enough.
        production = self.shared.get("language.production")
        if (
            production
            and production != self._last_production
            and step % max(1, self._produce_every) == 0
        ):
            self._last_production = dict(production)
            self.publish(
                self._event(
                    EventType.LANGUAGE_CLAIM.value,
                    "language",
                    observation={
                        "text": f"{production['word']}",
                        "words": [production["word"]],
                        "polarity": None,
                        "speaker": "self",
                        "cue": production.get("cue", ""),
                    },
                    provenance="self.naming",
                    confidence=float(production.get("confidence", 0.5)),
                )
            )

        self.subject_state.step = step
        self.subject_state.sim_time = self.clock.sim_time
        self.subject_state.fault_count = self.core.faults
        self.subject_state.updated_at = utc_now()
        if len(self._pending) >= self._flush_every:
            self.flush_events()
        # Retention belongs to the step path, not to run(): scripts and
        # experiments drive step() directly, and a long run that bypasses run()
        # must still stay bounded. (Learned the hard way: a 100k-step run grew
        # to 875 MB because archival only lived in run().)
        if (
            self.store is not None
            and self.persist
            and self._archive_every
            and step % self._archive_every == 0
        ):
            self._archive_events()
        if getattr(self.world, "done", False) and bool(
            self.config.get("runtime.auto_reset_episodes", True)
        ):
            self.end_episode()

        return StepResult(
            step=step,
            sim_time=self.clock.sim_time,
            action=action,
            reward=float(result.reward),
            reached=bool(result.reached),
            prediction_error=error,
            affect=affect,
            goal=str(self.shared.get("volition.goal", "")),
            h_norm=float(metrics.h_norm),
            faults=int(metrics.faults),
            cue=str(percept["cue"]),
        )

    def _archive_events(self) -> dict[str, Any]:
        """Lossless retention: gzip the oldest events, then drop them from SQLite."""
        if self.store is None:
            return {}
        keep = int(self.config.get("memory.retention.max_events", 20000))
        archive_dir = self.config.resolve_path("paths.storage") / "archive"
        self.flush_events()
        result = self.store.archive_and_prune(keep, archive_dir)
        if result.get("archived"):
            self.log.info(
                "archived %d events to %s (kept %d)",
                result["archived"], result.get("path"), result.get("kept", 0),
            )
            if self.audit:
                self.audit.record("events.archived", **result)
        return result

    def end_episode(self) -> None:
        self.flush_events()
        if self.store is not None and self.persist:
            self.store.end_episode(self.episode_id)
        self.subject_state.episode_index += 1
        if hasattr(self.world, "reset_objects"):
            self.world.reset_objects()
        else:
            self.world.reset()
        if self.store is not None and self.persist:
            self.episode_id = self.store.start_episode(
                episode_index=self.subject_state.episode_index,
                world_id=str(self.config.get("world.id", "grid")),
                seed=self.seed,
                config_hash=self.config.hash(),
                notes=f"auto episode {self.subject_state.episode_index}",
            )

    def _features(self, percept: dict[str, Any], error: float) -> dict[str, float]:
        practice_state = self.registry.get("practice").feature_state()
        return percept_to_features(
            percept,
            affect=self.affect_values(),
            practice={
                **practice_state,
                "trust_mean": self.shared.get("social.trust_mean", 0.5),
                "predicted_dx": (self.shared.get("social.hint", [0.0, 0.0]) or [0.0, 0.0])[0],
            },
            temporal={
                "events_per_step": self.shared.get("temporal.events_per_step", 0.0),
            },
            concept={"norm": self.shared.get("concept.norm", 0.0)},
            volition={"value": self.shared.get("volition.score", 0.0)},
            error_ema=float(self.shared.get("world.error_ema", 0.0)),
            step=self.clock.step,
            episode_index=self.subject_state.episode_index,
        )

    # ------------------------------------------------------------------ events
    def _event(self, event_type: str, modality: str, **kwargs: Any) -> Event:
        kwargs.setdefault("affect_snapshot", self.affect_values())
        kwargs.setdefault("privacy_class", PrivacyClass.PUBLIC_SIM.value)
        return Event.create(
            event_type,
            step=self.clock.step,
            episode_id=self.episode_id,
            actor_id=self.subject_state.subject_id,
            modality=modality,
            **kwargs,
        )

    def publish(self, event: Event) -> list[Event]:
        # Fail fast: nothing classified USER_PRIVATE may enter a persisted loop.
        # There is no consent mechanism yet, so accepting it and dropping it at
        # flush time would silently lose data; refusing here is auditable.
        if self.persist:
            event.assert_persistable()
        emitted = self.bus.publish(event)
        self._pending.extend(emitted)
        return emitted

    def flush_events(self) -> int:
        if not self._pending:
            return 0
        if self.store is None or not self.persist:
            self._pending.clear()
            return 0
        batch = self._pending
        self._pending = []
        try:
            self.store.append_events_batch(batch)
        except Exception:
            self._pending = batch + self._pending
            raise
        self._events_written += len(batch)
        return len(batch)

    def affect_values(self) -> dict[str, float]:
        try:
            module = self.registry.get("affect")
        except KeyError:
            return {}
        return module.snapshot_values()

    def reset_affect(self) -> dict[str, float]:
        """Manual recovery hook (MASTER SPEC 7.3): return affect to baseline."""
        module = self.registry.get("affect")
        module.reset_to_baseline()
        if self.audit:
            self.audit.record("affect.reset", step=self.clock.step)
        return self.affect_values()

    # ----------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        health = self.registry.health_report()
        status: dict[str, Any] = {
            "version": __version__,
            "spec": self.config.get("project.spec", ""),
            "config_hash": self.config.hash(),
            "seed": self.seed,
            "subject_id": self.subject_state.subject_id,
            "episode_id": self.episode_id,
            "episode_index": self.subject_state.episode_index,
            "step": self.clock.step,
            "sim_time": round(self.clock.sim_time, 4),
            "world": self.world.world_id,
            "core": self.core.sanity(),
            "core_layout": {
                "modules": list(self.core.module_names),
                "total_units": self.core.total_units,
                "concept_dim": self.core.concept_dim,
                "total_inputs": self.core.total_inputs,
            },
            "sparsity": self.core.density_report(),
            "modules": [h.to_dict() for h in health],
            "overall_status": self.registry.overall_status(),
            "affect": self.affect_values(),
            "goal": self.shared.get("volition.goal", ""),
            "action": self.shared.get("practice.action", ""),
            "concepts": self.shared.get("concept.count", 0),
            "scheduler": self.scheduler.report(),
            "memory_layers": self.memory.stats(),
            "planner": self.planner.state(),
            "bus": self.bus.stats(),
            "sensors": (
                {
                    "enabled": True,
                    "overall": self.suite.overall_status(),
                    "health": [h.to_dict() for h in self.suite.health()],
                    "coverage": self.suite.coverage(),
                    "readings": self.suite.reading_count,
                    "invalid_readings": self.suite.invalid_count,
                    "faults": dict(self.suite.faults),
                }
                if self.suite is not None
                else {"enabled": False}
            ),
            "pending_events": len(self._pending),
            "events_written": self._events_written,
            "control": self.control.state(),
        }
        if self.store is not None:
            status["memory"] = self.store.quick_stats()
        return status

    def write_status_snapshot(self) -> Path:
        """Persist the current status so ``wphr status`` can read it cross-process."""
        target = self.config.resolve_path("paths.status")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.status()
        payload["written_at"] = utc_iso()
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def health(self) -> dict[str, Any]:
        return {
            "overall": self.registry.overall_status(),
            "modules": [h.to_dict() for h in self.registry.health_report()],
            "core": self.core.sanity(),
            "bus_errors": len(self.bus.errors),
        }

    # ------------------------------------------------------------ checkpoint
    def save_checkpoint(self, note: str = "") -> Checkpoint:
        if self.store is None or self.checkpoints is None:
            raise RuntimeError("persistence is disabled for this subject")
        self.flush_events()
        self.subject_state.module_states = self.registry.snapshot()
        self.subject_state.shared = self.shared.snapshot()
        self.subject_state.rng_state = json.dumps(self.rng.bit_generator.state, sort_keys=True)
        checkpoint = self.checkpoints.create(
            subject=self.subject_state,
            core=self.core,
            registry=self.registry,
            scheduler_stats=self.scheduler.report(),
            bus_stats=self.bus.stats(),
            world_state=self.world.snapshot(),
            extra_meta={
                "sensors": self.suite.snapshot() if self.suite is not None else {},
                "memory": self.memory.snapshot(),
                "planner": self.planner.snapshot(),
            },
            note=note,
        )
        if self.audit:
            self.audit.record(
                "checkpoint.created",
                checkpoint_id=checkpoint.checkpoint_id,
                step=checkpoint.step,
                sha256=checkpoint.sha256,
            )
        return checkpoint

    def restore(self, checkpoint_id: str | None = None, *, allow_config_mismatch: bool = False) -> Checkpoint:
        if self.store is None or self.checkpoints is None:
            raise RuntimeError("persistence is disabled for this subject")
        checkpoint = self.checkpoints.load(
            checkpoint_id, allow_config_mismatch=allow_config_mismatch
        )
        checkpoint.restore_into(
            self.core,
            self.registry,
            self.subject_state,
            allow_config_mismatch=allow_config_mismatch,
        )
        world_state = checkpoint.meta.get("world_state")
        if world_state:
            self.world.restore(world_state)
        extra = checkpoint.meta.get("extra") or {}
        if self.suite is not None and extra.get("sensors"):
            self.suite.restore(extra["sensors"])
        if extra.get("memory"):
            self.memory.restore(extra["memory"])
        if extra.get("planner"):
            self.planner.restore(extra["planner"])
        # Offline gap: recorded, never filled with invented events.
        self.last_offline_gap_s = max(0.0, utc_now() - float(checkpoint.created_at))
        if self.audit:
            self.audit.record(
                "offline_gap.recorded",
                seconds=round(self.last_offline_gap_s, 3),
                checkpoint_id=checkpoint.checkpoint_id,
            )
        self.clock.reset(self.subject_state.step)
        self.clock.sim_time = self.subject_state.sim_time
        self.shared.restore(self.subject_state.shared)
        if self.subject_state.rng_state:
            self.rng.bit_generator.state = json.loads(self.subject_state.rng_state)
        # The episode counter is part of the feature vector, so restore must not
        # change it; only the persistence bookkeeping starts a new episode row.
        if self.store is not None and self.persist:
            self.store.end_episode(self.episode_id, notes="restored")
            self.episode_id = self.store.start_episode(
                episode_index=self.subject_state.episode_index,
                world_id=str(self.config.get("world.id", "grid")),
                seed=self.seed,
                config_hash=self.config.hash(),
                notes=f"restored from {checkpoint.checkpoint_id}",
            )
        if self.audit:
            self.audit.record(
                "checkpoint.restored",
                checkpoint_id=checkpoint.checkpoint_id,
                step=checkpoint.step,
            )
        return checkpoint

    # ------------------------------------------------------------------- run
    def run(
        self,
        steps: int,
        *,
        control: ControlFile | None = None,
        checkpoint_every: int | None = None,
        on_step: Callable[[StepResult], None] | None = None,
        stop_on_control: bool = True,
    ) -> RunSummary:
        control = control or self.control
        started_at = utc_now()
        start_step = self.clock.step
        start_wall = time.perf_counter()
        checkpoint_every = (
            int(self.config.get("runtime.checkpoint_every", 0))
            if checkpoint_every is None
            else int(checkpoint_every)
        )
        max_wall = float(self.config.get("runtime.max_wall_clock_s", 0) or 0)
        if max_wall <= 0:
            max_wall = float(self.config.get("safety.max_wall_clock_s", 86400))
        sleep_s = float(self.config.get("runtime.sleep_between_steps_s", 0.0) or 0.0)
        status_every = int(self.config.get("runtime.status_every", 0) or 0)
        limit = min(steps, int(self.config.get("runtime.max_steps", 10**9)) - start_step)
        state = "completed"
        checkpoints = 0
        last_checkpoint_step = start_step
        try:
            for _ in range(max(0, limit)):
                if stop_on_control and self.clock.step % 10 == 0:
                    command = control.pending()
                    if command in ("pause", "stop"):
                        state = {"pause": "paused", "stop": "stopped"}[command]
                        self.log.info("control request received: %s", command)
                        break
                    if command == "checkpoint":
                        control.clear()
                        self.save_checkpoint(note="control")
                        checkpoints += 1
                        last_checkpoint_step = self.clock.step
                        self.log.info("checkpoint requested via control channel")
                    if command == "reset_affect":
                        control.clear()
                        self.reset_affect()
                        self.log.info("affect reset requested via control channel")
                result = self.step()
                if on_step is not None:
                    on_step(result)
                if checkpoint_every and self.clock.step - last_checkpoint_step >= checkpoint_every:
                    self.save_checkpoint(note="auto")
                    last_checkpoint_step = self.clock.step
                    checkpoints += 1
                if status_every and self.clock.step % status_every == 0:
                    self.write_status_snapshot()
                if sleep_s:
                    time.sleep(sleep_s)
                if time.perf_counter() - start_wall > max_wall:
                    state = "wall_clock_limit"
                    self.log.warning("wall-clock limit reached; stopping the run")
                    break
        except KeyboardInterrupt:
            state = "interrupted"
            self.log.warning("interrupted; flushing and saving a checkpoint")
            self.flush_events()
            if self.checkpoints is not None:
                self.save_checkpoint(note="interrupt")
                checkpoints += 1
        finally:
            self.flush_events()
            written = self._events_written
        elapsed = time.perf_counter() - start_wall
        return RunSummary(
            started_at=started_at,
            finished_at=utc_now(),
            start_step=start_step,
            end_step=self.clock.step,
            steps=self.clock.step - start_step,
            state=state,
            events_written=written,
            checkpoints=checkpoints,
            wall_seconds=elapsed,
        )
