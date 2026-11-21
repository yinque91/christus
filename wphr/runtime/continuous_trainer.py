"""Continuous trainer: camera preview -> motion tracking -> prediction -> memory.

Runs unattended. Control files (in WORKSPACE/control):
  STOP   -> exit cleanly
  PAUSE  -> idle until removed
Logs: WORKSPACE/logs/trainer.jsonl
Checkpoint: WORKSPACE/state/trainer_state.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
CTRL = WORKSPACE / "control"
LOGS = WORKSPACE / "logs"
STATE = WORKSPACE / "state"
FRAMES = WORKSPACE / "frames"
for d in (CTRL, LOGS, STATE, FRAMES):
    d.mkdir(parents=True, exist_ok=True)

INTERVAL = 1.0
CKPT_EVERY = 10


def capture():
    from wphr.sensors.camera_screen import capture as cap
    p = FRAMES / "latest.png"
    meta = cap(str(p))
    if not meta.get("ok"):
        return None
    from PIL import Image
    return np.asarray(Image.open(p).convert("L"), float)


def motion_centroid(prev, cur, thresh=12.0):
    if prev is None:
        return None, 0.0
    d = np.abs(cur - prev)
    mask = d > thresh
    w = float(mask.sum())
    if w < 30:
        return None, w
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())), w


class Predictor:
    def __init__(self):
        self.prev = None
        self.vel = (0.0, 0.0)
        self.errs = []

    def step(self, pos):
        if pos is None:
            return None
        pred = None
        if self.prev is not None:
            pred = (self.prev[0] + self.vel[0], self.prev[1] + self.vel[1])
            err = ((pred[0] - pos[0]) ** 2 + (pred[1] - pos[1]) ** 2) ** 0.5
            self.errs.append(err)
        if self.prev is not None:
            self.vel = (pos[0] - self.prev[0], pos[1] - self.prev[1])
        self.prev = pos
        return pred


def main():
    import sys
    sys.path.insert(0, r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")
    state_path = STATE / "trainer_state.json"
    step = 0
    if state_path.exists():
        try:
            step = json.loads(state_path.read_text(encoding="utf-8")).get("step", 0)
        except Exception:
            step = 0
    pred = Predictor()
    prev = None
    log = (LOGS / "trainer.jsonl").open("a", encoding="utf-8")
    while True:
        if (CTRL / "STOP").exists():
            log.write(json.dumps({"t": time.time(), "event": "stop"}) + "\n")
            log.close()
            break
        if (CTRL / "PAUSE").exists():
            time.sleep(1.0)
            continue
        cur = capture()
        if cur is None:
            log.write(json.dumps({"t": time.time(), "step": step, "event": "no_frame"}) + "\n")
            log.flush()
            time.sleep(INTERVAL)
            continue
        pos, weight = motion_centroid(prev, cur)
        p = pred.step(pos)
        err = None
        if p is not None and pos is not None:
            err = float(((p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2) ** 0.5)
        rec = {"t": time.time(), "step": step, "pos": pos, "motion_px": weight,
               "pred": p, "err": err, "shape": list(cur.shape)}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        prev = cur
        step += 1
        if step % CKPT_EVERY == 0:
            state_path.write_text(json.dumps({
                "step": step, "last_pos": pos, "last_err": err,
                "mean_err": float(np.mean(pred.errs[-50:])) if pred.errs else None,
                "saved_at": time.time()}, ensure_ascii=False), encoding="utf-8")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
