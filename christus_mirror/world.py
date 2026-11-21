"""2D mirror environment with a rendered mirror, body sensors and hidden markers.

The subject receives mirror/first-person feature vectors and body sensors. Ground
truth (condition, marker location, whether the image is self) stays in
`EvaluatorTruth` and is never passed to the subject.
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from wphr.world.base import WorldObservation, WorldResult

ACTIONS = ("head_left", "head_right", "hand_left", "hand_right",
           "turn_left", "turn_right", "step_left", "step_right",
           "look_mirror", "look_away",
           "touch_forehead", "touch_left_hand", "touch_right_hand")
PARTS = ("head", "left_hand", "right_hand", "torso")
MARKER_LOCATIONS = {0: "forehead", 1: "left_hand", 2: "right_hand"}
MARKER_TO_PART = {0: "head", 1: "left_hand", 2: "right_hand"}


@dataclass
class EvaluatorTruth:
    condition: str
    marker: int | None
    marker_mode: str  # real, none, mirror_only, other_agent, unseen
    seed: int


@dataclass
class MirrorWorld:
    world_id: str = "mirror_env"
    seed: int = 11000
    mirror_condition: str = "realtime"  # realtime, delay_0.5, delay_2, other, none
    delay_steps: int = 0
    mirror_scale: float = 1.0
    mirror_angle: float = 0.0
    lighting: float = 1.0
    body_scale: float = 1.0
    marker: int | None = None
    marker_mode: str = "none"
    other_agent: bool = False
    frame_dir: str | None = None
    truth: EvaluatorTruth | None = None
    step: int = 0
    body: dict = field(default_factory=dict)
    visual_history: list = field(default_factory=list)
    current: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)
    marker_seen_at: int | None = None
    marker_found: bool = False
    marker_contact: bool = False
    inspected: bool = False
    false_inspection: bool = False

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.mirror_queue = []
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(seed)
        self.body = {
            "head": [32.0, 38.0, 0.0],
            "left_hand": [24.0, 52.0, 0.0],
            "right_hand": [40.0, 52.0, 0.0],
            "torso": [32.0, 58.0, 0.0],
        }
        self.step = 0
        self.mirror_queue = []
        self.marker = None
        self.marker_mode = "none"
        self.marker_seen_at = None
        self.marker_found = False
        self.marker_contact = False
        self.inspected = False
        self.false_inspection = False
        self.current = {}
        self.last = {}
        self.visual_history = []

    def actions(self):
        return ACTIONS

    def set_marker(self, location: int, mode: str = "real"):
        self.marker = location
        self.marker_mode = mode

    def clear_marker(self):
        self.marker = None
        self.marker_mode = "none"

    def _apply_action(self, action: str):
        h = self.body["head"]
        lh = self.body["left_hand"]
        rh = self.body["right_hand"]
        if action == "head_left":
            h[2] -= 0.22
        elif action == "head_right":
            h[2] += 0.22
        elif action == "hand_left":
            lh[1] -= 3.0
        elif action == "hand_right":
            rh[1] -= 3.0
        elif action == "turn_left":
            for part in self.body.values():
                part[0] -= 1.5
        elif action == "turn_right":
            for part in self.body.values():
                part[0] += 1.5
        elif action == "step_left":
            for part in self.body.values():
                part[0] -= 2.0
        elif action == "step_right":
            for part in self.body.values():
                part[0] += 2.0
        elif action == "look_mirror":
            self.body["head"][2] = 0.0
        elif action == "look_away":
            self.body["head"][2] = math.pi
        elif action == "touch_forehead":
            self.inspected = True
            self.marker_contact = self.marker == 0 and self.marker_mode in ("real", "unseen")
            if self.marker is not None and not self.marker_contact:
                self.false_inspection = True
        elif action == "touch_left_hand":
            self.inspected = True
            self.marker_contact = self.marker == 1 and self.marker_mode in ("real", "unseen")
            if self.marker is not None and not self.marker_contact:
                self.false_inspection = True
        elif action == "touch_right_hand":
            self.inspected = True
            self.marker_contact = self.marker == 2 and self.marker_mode in ("real", "unseen")
            if self.marker is not None and not self.marker_contact:
                self.false_inspection = True
        # clamp / keep body in front of mirror
        for part in self.body.values():
            part[0] = float(np.clip(part[0], 8.0, 56.0))
            part[1] = float(np.clip(part[1], 20.0, 65.0))

    def _mirror_point(self, p):
        # Reflection around x=32, with scale/angle and a small perspective offset.
        dx = (p[0] - 32.0) * self.mirror_scale
        dy = p[1] * self.body_scale
        ang = self.mirror_angle
        x = 32.0 - dx * math.cos(ang) + dy * math.sin(ang) * 0.15
        y = dy - dx * math.sin(ang) * 0.15
        return [float(x), float(y)]

    def _marker_on(self, part_name: str):
        if self.marker is None:
            return False
        if self.marker_mode == "none":
            return False
        if self.marker_mode == "other_agent":
            return False
        target = MARKER_TO_PART.get(self.marker)
        if self.marker_mode == "mirror_only":
            return part_name == target
        return part_name == target

    def _visual_features(self):
        mirror = {k: self._mirror_point(v) for k, v in self.body.items()}
        first = {k: list(v) for k, v in self.body.items()}
        marker_pixels = [0.0, 0.0, 0.0]
        if self.marker is not None:
            target = MARKER_TO_PART[self.marker]
            if self.marker_mode == "other_agent":
                # The other body is visible only in the mirror, offset to the left.
                pass
            elif self.marker_mode in ("real", "mirror_only", "unseen"):
                marker_pixels[self.marker] = 1.0
        # Other-agent mirror: a second silhouette is drawn and may carry a marker.
        if self.mirror_condition == "other":
            other = {"head": [18.0, 38.0], "left_hand": [12.0, 52.0],
                     "right_hand": [24.0, 52.0], "torso": [18.0, 58.0]}
            other_marker = [0.0, 0.0, 0.0]
            if self.marker is not None and self.marker_mode == "other_agent":
                other_marker[self.marker] = 1.0
        else:
            other = None
            other_marker = [0.0, 0.0, 0.0]
        order = ("head", "left_hand", "right_hand", "torso")
        v = []
        for k in order:
            v += mirror[k][:2]
        v += marker_pixels
        v += [self.visual_history[-1][i] - v[i] if self.visual_history else 0.0
              for i in range(min(8, len(v)))]
        v += [self.lighting, float(self.mirror_scale), float(self.mirror_angle),
              float(self.other_agent)]
        # first-person visual body: marker is never visible on one's own forehead
        v += [first["left_hand"][0], first["right_hand"][0], first["head"][2]]
        if other:
            v += [other["head"][0], other["torso"][0]]
            v += other_marker
        else:
            v += [0.0] * 5
        return np.asarray(v, dtype=np.float64), marker_pixels

    def observe(self):
        v, marker_pixels = self._visual_features()
        # No-mirror condition: the subject has no mirror visual stream.
        if self.mirror_condition == "none":
            v = v.copy()
            v[:11] = 0.0
            marker_pixels = [0.0, 0.0, 0.0]
        # Other-agent condition: mirror silhouette is another body, not the self.
        if self.mirror_condition == "other":
            v = v.copy()
            v[:8] = [18.0, 38.0, 12.0, 52.0, 24.0, 52.0, 18.0, 58.0]
            v[8:11] = [0.0, 0.0, 0.0]
        # Delayed mirror: return a delayed visual vector while body sensors stay live.
        if self.mirror_condition in ("delay_0.5", "delay_2"):
            self.mirror_queue.append(v.copy())
            keep = 8 if self.mirror_condition == "delay_2" else 4
            if len(self.mirror_queue) > keep:
                v = self.mirror_queue.pop(0)
                marker_pixels = [float(v[i]) for i in range(8, 11)]
        self.visual_history.append(v.copy())
        self.visual_history = self.visual_history[-12:]
        body_values = []
        for k in ("head", "left_hand", "right_hand", "torso"):
            body_values += self.body[k][:2]
        values = list(v) + body_values
        self.current = {
            "visual": v, "marker_pixels": marker_pixels, "body": np.asarray(body_values),
            "mirror_visible": self.mirror_condition != "none",
        }
        self.step += 1
        return WorldObservation(modality="mirror", values=values, energy=1.0,
                                step=self.step, meta={"visible": True,
                                                      "cue": "mirror",
                                                      "scent": [0.0, 0.0]})

    def execute(self, action: str):
        if action not in ACTIONS:
            action = "look_mirror"
        before = {k: list(v) for k, v in self.body.items()}
        self._apply_action(action)
        after = {k: list(v) for k, v in self.body.items()}
        motion = sum(abs(after[k][0] - before[k][0]) + abs(after[k][1] - before[k][1]) +
                     abs(after[k][2] - before[k][2]) for k in after)
        # Mirror condition: no mirror gives no mirror change; delay/other use
        # their own visual stream (handled by the policy's visual history).
        mirror_motion = motion if self.mirror_condition == "realtime" else 0.0
        if self.mirror_condition in ("delay_0.5", "delay_2"):
            mirror_motion = motion
        self.last = {
            "action": action, "body_motion": motion, "mirror_motion": mirror_motion,
            "marker_contact": bool(self.marker_contact),
            "inspected": bool(self.inspected),
            "false_inspection": bool(self.false_inspection),
            "marker": self.marker, "marker_mode": self.marker_mode,
            "condition": self.mirror_condition,
        }
        return WorldResult(action=action, success=True, reward=0.0,
                           energy_after=1.0, info={"cue": "mirror",
                                                   "body_motion": motion,
                                                   "mirror_motion": mirror_motion})

    def snapshot(self):
        return {"body": copy.deepcopy(self.body), "step": self.step,
                "marker": self.marker, "marker_mode": self.marker_mode,
                "inspected": self.inspected, "marker_contact": self.marker_contact,
                "false_inspection": self.false_inspection}

    def restore(self, data):
        self.body = copy.deepcopy(data.get("body", self.body))
        self.step = int(data.get("step", 0))
        self.marker = data.get("marker")
        self.marker_mode = data.get("marker_mode", "none")
        self.inspected = bool(data.get("inspected", False))
        self.marker_contact = bool(data.get("marker_contact", False))
        self.false_inspection = bool(data.get("false_inspection", False))

    def render(self, path: str, marker_location=None):
        img = Image.new("RGB", (480, 320), (18, 22, 25))
        d = ImageDraw.Draw(img)
        # mirror frame
        d.rectangle([80, 30, 400, 290], outline=(120, 190, 185), width=3)
        d.line([240, 30, 240, 290], fill=(50, 65, 68), width=1)
        for name, p in self.body.items():
            mp = self._mirror_point(p)
            x = 240 + (mp[0] - 32) * 4.0
            y = 35 + (mp[1] - 20) * 4.2
            color = (230, 230, 225)
            d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=color)
            if self.marker is not None and MARKER_TO_PART[self.marker] == name:
                if self.marker_mode in ("real", "mirror_only", "unseen"):
                    d.ellipse([x - 4, y - 4, x + 4, y + 4],
                              fill=(235, 70, 70))
        d.text((12, 8), f"mirror={self.mirror_condition} step={self.step}",
               fill=(150, 170, 172))
        img.save(path)
        return path



