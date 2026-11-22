"""Episodic log and a small novelty/transition model for the autonomy loop."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path


class ExperienceStore:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "experience.jsonl"

    def record(self, observation: dict, *, action: str | None = None,
               result: dict | None = None, novelty: float = 0.0) -> dict:
        row = {"ts": time.time(), "observation": observation,
               "action": action, "result": result, "novelty": novelty}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        rows = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in rows if x.strip()]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for x in self.path.open(encoding="utf-8") if x.strip())


class NoveltyModel:
    """Counts window transitions; novel transitions get a higher learning signal."""

    def __init__(self, state_dir: str | Path):
        self.path = Path(state_dir) / "novelty.json"
        self.transitions: Counter[str] = Counter()
        self.windows: Counter[str] = Counter()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.transitions.update(data.get("transitions", {}))
                self.windows.update(data.get("windows", {}))
            except (OSError, ValueError):
                pass
        self.last = ""

    def score(self, observation: dict) -> float:
        key = f"{observation.get('process','')}|{observation.get('title','')}"
        transition = f"{self.last}->{key}"
        known = self.transitions.get(transition, 0)
        novel = 1.0 / (1.0 + known)
        self.last = key
        self.windows[key] += 1
        self.transitions[transition] += 1
        return float(novel)

    def save(self):
        self.path.write_text(json.dumps({
            "windows": dict(self.windows),
            "transitions": dict(self.transitions),
        }, ensure_ascii=False), encoding="utf-8")

    def snapshot(self) -> dict:
        return {"transition_kinds": len(self.transitions),
                "window_kinds": len(self.windows)}
