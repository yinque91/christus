"""Subject service loop: persistence + camera/prediction + health + graceful stop."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WS = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")

sys.path.insert(0, r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")

from wphr.persistence.store import COMPONENTS, SubjectStore  # noqa: E402
from wphr.runtime.service_manager import ServiceManager  # noqa: E402

SAVE_EVERY = 10
TICK = 2.0


def default_state() -> dict:
    st = {c: {} for c in COMPONENTS}
    st["identity_state"] = {"subject_id": "wphr-local", "created": time.time()}
    st["cognitive_state"] = {"predictor": {"prev": None, "vel": [0.0, 0.0]},
                             "step": 0, "mean_err": None}
    st["affect_state"] = {"values": {"curiosity": 0.5, "frustration": 0.0},
                          "last_event": None}
    st["world_model_state"] = {"last_pos": None, "last_err": None}
    st["sensor_configuration"] = {"camera": {"mode": "SCREEN_OF_CAMERA_APP",
                                             "enabled": True}}
    st["episodic_memory"] = {"recent": []}
    return st


def capture():
    try:
        import numpy as np
        from PIL import Image
        from wphr.sensors.camera_screen import capture as cap
        p = WS / "frames" / "service_latest.png"
        m = cap(str(p))
        if not m.get("ok"):
            return None
        return np.asarray(Image.open(p).convert("L"), float)
    except Exception:
        return None


def motion(prev, cur, thresh=12.0):
    if prev is None:
        return None
    import numpy as np
    d = np.abs(cur - prev)
    mask = d > thresh
    if mask.sum() < 30:
        return None
    ys, xs = np.nonzero(mask)
    return [float(xs.mean()), float(ys.mean())]


def run_once(sm: ServiceManager, store: SubjectStore, state: dict) -> dict:
    cog = state["cognitive_state"]
    world = state["world_model_state"]
    prev = cog["predictor"].get("prev")
    cur = capture()
    pos = motion(None if prev is None else prev, cur) if cur is not None else None
    pred = None
    err = None
    if pos is not None and prev is not None:
        vel = cog["predictor"].get("vel", [0.0, 0.0])
        pred = [prev[0] + vel[0], prev[1] + vel[1]]
        err = float(((pred[0] - pos[0]) ** 2 + (pred[1] - pos[1]) ** 2) ** 0.5)
    if pos is not None:
        if prev is not None:
            cog["predictor"]["vel"] = [pos[0] - prev[0], pos[1] - prev[1]]
        cog["predictor"]["prev"] = pos
    cog["step"] = int(cog.get("step", 0)) + 1
    cog["mean_err"] = None if err is None else round(
        0.9 * (cog.get("mean_err") or err) + 0.1 * err, 3)
    world["last_pos"] = pos
    world["last_err"] = err
    if err is not None:
        state["affect_state"]["values"]["frustration"] = min(
            1.0, float(state["affect_state"]["values"].get("frustration", 0.0)) * 0.95 + 0.01 * min(err, 5.0))
        state["affect_state"]["last_event"] = {"ts": time.time(), "err": round(err, 3)}
    if cur is not None:
        state["episodic_memory"]["recent"] = (
            state["episodic_memory"].get("recent", [])[-20:]
            + [{"ts": time.time(), "pos": pos, "err": err}])
    return {"step": cog["step"], "pos": pos, "err": err, "mean_err": cog["mean_err"]}


def main():
    sm = ServiceManager(WS / "service")
    if not sm.acquire():
        print(json.dumps({"ok": False, "reason": "already_running"}))
        return
    sm.install_signals()
    store = SubjectStore(WS / "subject")
    if store.manifest_path.exists():
        state = store.load()
        sm.log("state_loaded")
    else:
        state = default_state()
        sm.log("state_init")
    sm.log("service_start")
    try:
        while not sm.stop_requested():
            if sm.paused():
                sm.write_status({"mode": "paused"})
                time.sleep(TICK)
                continue
            rec = run_once(sm, store, state)
            step = rec["step"]
            if step % SAVE_EVERY == 0:
                store.save(state, reason="periodic")
            sm.write_status({"mode": "running", **rec})
            sm.log(f"tick step={step} err={rec['err']}")
            time.sleep(TICK)
    except Exception as e:
        sm.log("fatal_error: " + repr(e))
        raise
    finally:
        store.save(state, reason="shutdown")
        sm.log("service_stop")
        sm.release()


if __name__ == "__main__":
    main()
