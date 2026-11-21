"""Embodied world: camera + network + filesystem as the subject's real environment.

Actions keep the grid-world names (move_*, interact, wait) so the existing
safety whitelist and module policies apply unchanged.

- observe(): real camera preview (screen path) -> brightness / motion / ROI
- move_*: shift the ROI window (real effect on the next observation)
- interact: perform a real HTTP fetch from a queue; reward depends on success
- wait: idle one tick
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wphr.world.base import WorldObservation, WorldResult

WORKSPACE = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
QUEUE = [
    "https://example.com",
    "https://www.python.org",
    "https://docs.python.org/3/library/index.html",
    "https://en.wikipedia.org/wiki/Embodied_cognition",
]


class EmbodiedWorld:
    world_id = "embodied"

    def __init__(self, log_path: Path | None = None):
        self.step_i = 0
        self.roi = (0, 0)          # x, y offset into the frame
        self.last_frame = None
        self.prev_frame = None
        self.queue = list(QUEUE)
        self.fetched: list[dict] = []
        self.log_path = log_path or (WORKSPACE / "logs" / "embodied_world.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._energy = 1.0

    # ------------------------------------------------------------- observation
    def _grab(self):
        from wphr.sensors.camera_screen import capture
        p = WORKSPACE / "frames" / "embodied.png"
        m = capture(str(p))
        if not m.get("ok"):
            return None
        from PIL import Image
        img = np.asarray(Image.open(p).convert("L"), float)
        return img

    def _battery(self) -> float:
        try:
            import psutil
            b = psutil.sensors_battery()
            return float(b.percent) / 100.0 if b else 1.0
        except Exception:
            return 1.0

    def observe(self) -> WorldObservation:
        img = self._grab()
        if img is None:
            return WorldObservation(modality="vision", values=[0.0, 0.0, 0.0],
                                    energy=self._energy, step=self.step_i,
                                    source_device="camera_screen", valid=False,
                                    meta={"reason": "no_frame"})
        x, y = self.roi
        h, w = img.shape
        crop = img[y:y + 300, x:x + 400]
        brightness = float(crop.mean()) / 255.0
        motion = 0.0
        cx, cy = 0.0, 0.0
        if self.prev_frame is not None and self.prev_frame.shape == img.shape:
            d = np.abs(img - self.prev_frame)
            mask = d > 12
            motion = float(mask.sum()) / mask.size
            if mask.sum() > 50:
                ys, xs = np.nonzero(mask)
                cx, cy = float(xs.mean()) / w, float(ys.mean()) / h
        self.prev_frame = img
        self.last_frame = img
        self._energy = self._battery()
        return WorldObservation(
            modality="vision",
            values=[round(brightness, 3), round(motion, 4), round(cx, 3), round(cy, 3)],
            energy=self._energy, step=self.step_i,
            source_device="camera_screen",
            confidence=0.8, valid=True,
            meta={"roi": self.roi, "queue_left": len(self.queue)},
        )

    # ----------------------------------------------------------------- actions
    def actions(self) -> tuple[str, ...]:
        return ("move_north", "move_south", "move_east", "move_west", "interact", "wait")

    def execute(self, action: str) -> WorldResult:
        info: dict = {"action": action}
        reward = -0.005
        success = True
        if action.startswith("move_"):
            dx, dy = {"move_east": (40, 0), "move_west": (-40, 0),
                      "move_north": (0, -40), "move_south": (0, 40)}[action]
            self.roi = (max(0, min(500, self.roi[0] + dx)),
                        max(0, min(380, self.roi[1] + dy)))
            info["roi"] = self.roi
        elif action == "interact":
            if self.queue:
                url = self.queue.pop(0)
                try:
                    from wphr.tools.network import NetworkTool
                    r = NetworkTool().get(url, timeout=10)
                    ok = bool(r.get("ok"))
                    success = ok
                    reward = 1.0 if ok else -0.2
                    info.update({"url": url, "status": r.get("status"),
                                 "bytes": r.get("bytes")})
                    self.fetched.append({"url": url, "ok": ok,
                                         "bytes": r.get("bytes")})
                except Exception as e:
                    success = False
                    reward = -0.2
                    info["error"] = repr(e)
            else:
                success = False
                reward = -0.05
                info["reason"] = "queue_empty"
        elif action == "wait":
            time.sleep(0.05)
        self.step_i += 1
        rec = {"ts": time.time(), "step": self.step_i, "action": action,
               "reward": reward, "success": success, "info": info}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return WorldResult(action=action, success=success, reward=reward,
                           energy_after=self._energy, reached=False, info=info)

    def reset(self, seed: int | None = None) -> None:
        self.step_i = 0
        self.roi = (0, 0)
        self.queue = list(QUEUE)

    def snapshot(self) -> dict:
        return {"roi": self.roi, "queue": self.queue, "fetched": self.fetched}

    def restore(self, data: dict) -> None:
        self.roi = tuple(data.get("roi", (0, 0)))
        self.queue = list(data.get("queue", QUEUE))
        self.fetched = list(data.get("fetched", []))
