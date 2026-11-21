"""Existing Subject adapted to the mirror environment.

The action hook is the only decision path: actions are chosen by the Subject's
core state and the mirror policy. The environment never inserts an action.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wphr.config import Config
from wphr.memory.store import Store
from wphr.runtime.features import FEATURE_NAMES
from wphr.runtime.subject import Subject
from christus_mirror.world import ACTIONS, MirrorWorld


class MirrorPolicy:
    """Functional action selection for the mirror task.

    Actions are selected from the live Subject core state. When a visible marker
    is present and the learned action鈥搈irror contingency is high, the policy may
    request a body-part inspection. No evaluator condition or truth is read.
    """
    def __init__(self, seed: int = 11000, use_body: bool = True, use_self: bool = True,
                 shuffle: bool = False):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.015, (13, 200))
        self.use_body = use_body
        self.use_self = use_self
        self.shuffle = shuffle
        self.sync_score = 0.0
        self.marker_seen = False
        self.inspect_requested = False
        self.last_action = None
        self.last_visual = None
        self.history = []

    def _core_scores(self, subject):
        h = np.asarray(subject.core.h, dtype=np.float64)
        return np.tanh(self.W @ h)

    def _inspection(self, visual):
        # Marker pixels occupy indices 8:11. The mapping to touch actions is a
        # body schema, not an evaluator label.
        if not self.use_body:
            return None
        marker = np.asarray(visual[8:11], dtype=float)
        if marker.max() <= 0.5 or self.sync_score < 0.02:
            return None
        part = int(np.argmax(marker))
        return ("touch_forehead", "touch_left_hand", "touch_right_hand")[part]

    def act(self, subject):
        visual = np.asarray(subject.world.current.get("visual", np.zeros(39)), dtype=float)
        if self.last_visual is not None and not self.shuffle:
            dv = float(np.linalg.norm(visual[:8] - self.last_visual[:8]))
            self.sync_score = 0.85 * self.sync_score + 0.15 * min(1.0, dv / 4.0)
        elif self.shuffle:
            self.sync_score = 0.0
        self.last_visual = visual.copy()
        self.marker_seen = bool(visual[8:11].max() > 0.5)
        inspect = self._inspection(visual)
        self.inspect_requested = inspect is not None
        if inspect is not None:
            action = inspect
        else:
            scores = self._core_scores(subject)
            if self.rng.random() < 0.18:
                action = ACTIONS[int(self.rng.integers(0, len(ACTIONS)))]
            else:
                action = ACTIONS[int(np.argmax(scores))]
        self.last_action = action
        self.history.append({"action": action, "sync": self.sync_score,
                             "marker": self.marker_seen, "inspect": self.inspect_requested})
        self.history = self.history[-200:]
        return action, "mirror.policy"


class MirrorSubject(Subject):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.policy = None
        self.current_action = "look_mirror"

    def _features(self, percept, error):
        w = self.world
        visual = np.asarray(w.current.get("visual", np.zeros(39)), dtype=float)
        body = np.asarray(w.current.get("body", np.zeros(8)), dtype=float)
        src = np.concatenate([visual, body])
        out = {}
        k = 0
        for module, names in FEATURE_NAMES.items():
            for name in names:
                if name == "bias":
                    out[name] = 1.0
                else:
                    out[name] = float(src[k % len(src)])
                    k += 1
        out["energy"] = 1.0
        out["social_motivation"] = 0.5
        out["trust_mean"] = 0.5
        return out


def build_config(out_dir: Path) -> Config:
    cfg = Config.load(root=ROOT)
    for key in cfg.section("paths"):
        cfg.set("paths." + key, str((out_dir / cfg.get("paths." + key)).resolve()))
    cfg.set("world.id", "mirror_env")
    cfg.set("runtime.seed", 11000)
    cfg.set("runtime.auto_reset_episodes", False)
    cfg.set("runtime.status_every", 0)
    cfg.set("sensors.enabled", False)
    cfg.set("memory.enabled", True)
    cfg.set("memory.retention.auto_archive_every", 0)
    cfg.set("language.enabled", False)
    cfg.set("affect.ablation", "full")
    cfg.set("planning.enabled", False)
    cfg.set("modules.disabled", ["world", "practice", "social", "concept_language"])
    cfg.set("scheduler.fast", 1)
    cfg.set("scheduler.affect", 1)
    cfg.set("scheduler.slow", 1)
    cfg.set("safety.allowed_actions", list(ACTIONS))
    cfg.ensure_dirs()
    return cfg


class MirrorRuntime:
    def __init__(self, out_dir: Path, seed: int = 11000, *, use_body: bool = True,
                 use_self: bool = True, shuffle: bool = False):
        self.out_dir = Path(out_dir)
        cfg = build_config(self.out_dir)
        self.world = MirrorWorld(seed=seed)
        self.store = Store(cfg.resolve_path("paths.database"))
        self.subject = MirrorSubject(cfg, world=self.world, store=self.store,
                                     seed=seed, persist=True)
        self.subject.initialise()
        self.policy = MirrorPolicy(seed=seed, use_body=use_body, use_self=use_self,
                                   shuffle=shuffle)
        self.subject.policy = self.policy
        self.subject.action_hook = lambda s: self.policy.act(s)
        self.trace = []

    def step(self, action_override: str | None = None):
        # The experiment never overrides the action; retained only for smoke tests.
        result = self.subject.step()
        w = self.world.last
        v = np.asarray(self.world.current.get("visual", np.zeros(39)), dtype=float)
        self.trace.append({
            "step": self.subject.clock.step,
            "action": result.action,
            "sync": float(self.policy.sync_score),
            "marker_seen": bool(self.policy.marker_seen),
            "inspect_requested": bool(self.policy.inspect_requested),
            "body_motion": float(w.get("body_motion", 0)),
            "mirror_motion": float(w.get("mirror_motion", 0)),
            "inspected": bool(w.get("inspected", False)),
            "marker_contact": bool(w.get("marker_contact", False)),
            "false_inspection": bool(w.get("false_inspection", False)),
            "visual": v.tolist(),
            "body": self.world.current.get("body", np.zeros(8)).tolist(),
            "affect": self.subject.affect_values(),
        })
        return self.trace[-1]

    def save(self, note: str):
        ck = self.subject.save_checkpoint(note)
        return {"path": str(ck.path), "sha256": ck.sha256,
                "step": self.subject.clock.step}

    def close(self):
        try:
            self.subject.close()
        except Exception:
            pass


