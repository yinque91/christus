"""Run the real Subject against the embodied world (camera + network)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR")
sys.path.insert(0, str(ROOT))

from wphr.config import Config  # noqa: E402
from wphr.memory.store import Store  # noqa: E402
from wphr.runtime.subject import Subject  # noqa: E402
from wphr.world.embodied import EmbodiedWorld  # noqa: E402

OUT = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace\logs\embodied_subject.jsonl")


def main(steps: int = 40, delay: float = 0.5):
    cfg = Config.load()
    cfg.set("safety.allowed_actions",
            ["move_north", "move_south", "move_east", "move_west", "interact", "wait"])
    store = Store(cfg.resolve_path("paths.database"))
    world = EmbodiedWorld()
    subj = Subject(cfg, world=world, store=store, seed=None, persist=True)
    subj.initialise()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as log:
        for i in range(steps):
            r = subj.step()
            rec = {"ts": time.time(), "step": r.step, "action": r.action,
                   "goal": r.goal, "cue": r.cue, "reward": round(r.reward, 4),
                   "pred_err": round(r.prediction_error, 4),
                   "affect": {k: round(v, 3) for k, v in r.affect.items()},
                   "energy": round(getattr(world, "_energy", 1.0), 3)}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(rec, ensure_ascii=False))
            time.sleep(delay)
    subj.close(status="completed")
    print("done")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(n)
