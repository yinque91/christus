"""Atomic, versioned persistence for WPHR subject state (11 components)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

VERSION = 2
COMPONENTS = (
    "identity_state", "cognitive_state", "affect_state", "motivation_state",
    "episodic_memory", "semantic_memory", "concept_state", "world_model_state",
    "active_goals", "sensor_configuration", "pending_tasks",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SubjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backups = self.root / "backups"
        self.backups.mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.log_path = self.root / "recovery.log"
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------- load/save
    def load(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        migrated = False
        for name in COMPONENTS:
            p = self.root / f"{name}.json"
            if not p.exists():
                state[name] = {}
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            ver = int(data.get("_version", 1))
            payload = data.get("payload", {})
            if ver < VERSION:
                payload = self._migrate(name, ver, payload)
                migrated = True
            state[name] = payload
        self._loaded = state
        if migrated:
            self._log("migrated to version %d" % VERSION)
            self.save(state)
        return state

    def _migrate(self, name: str, from_version: int, payload: dict) -> dict:
        if from_version == 1 and VERSION >= 2:
            payload.setdefault("schema", "v2")
        return payload

    def save(self, state: dict[str, Any], *, reason: str = "periodic") -> dict:
        for name in COMPONENTS:
            if name not in state:
                raise ValueError(f"missing component: {name}")
        manifest: dict[str, Any] = {"version": VERSION, "saved_at": time.time(),
                                    "reason": reason, "files": {}}
        tmp_paths: list[tuple[Path, Path]] = []
        for name in COMPONENTS:
            final = self.root / f"{name}.json"
            tmp = self.root / f".{name}.json.tmp"
            body = {"_version": VERSION, "payload": state[name]}
            tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            tmp_paths.append((tmp, final))
        for tmp, final in tmp_paths:
            os.replace(tmp, final)
        for name in COMPONENTS:
            p = self.root / f"{name}.json"
            manifest["files"][name] = {"sha256": _sha(p), "bytes": p.stat().st_size}
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
        self._prune_backups()
        self._loaded = state
        return manifest

    # ------------------------------------------------------------- integrity
    def verify(self) -> dict:
        if not self.manifest_path.exists():
            return {"ok": False, "reason": "no_manifest"}
        man = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bad = []
        for name, meta in man.get("files", {}).items():
            p = self.root / f"{name}.json"
            if not p.exists() or _sha(p) != meta["sha256"]:
                bad.append(name)
        return {"ok": not bad, "corrupt": bad, "version": man.get("version"),
                "saved_at": man.get("saved_at")}

    # -------------------------------------------------------------- backups
    def backup(self, tag: str | None = None) -> Path:
        tag = tag or time.strftime("%Y%m%d-%H%M%S")
        dst = self.backups / f"subject-{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        for name in COMPONENTS:
            src = self.root / f"{name}.json"
            if src.exists():
                shutil.copy2(src, dst / src.name)
        if self.manifest_path.exists():
            shutil.copy2(self.manifest_path, dst / "manifest.json")
        return dst

    def restore(self, backup_dir: str | Path) -> dict:
        src = Path(backup_dir)
        if not (src / "manifest.json").exists():
            return {"ok": False, "reason": "invalid_backup"}
        for name in COMPONENTS:
            p = src / f"{name}.json"
            if p.exists():
                shutil.copy2(p, self.root / f"{name}.json")
        shutil.copy2(src / "manifest.json", self.manifest_path)
        self._log("restored from %s" % src.name)
        return {"ok": True, "from": str(src)}

    def _prune_backups(self, keep: int = 10) -> None:
        dirs = sorted([d for d in self.backups.iterdir() if d.is_dir()],
                      key=lambda d: d.stat().st_mtime, reverse=True)
        for d in dirs[keep:]:
            shutil.rmtree(d, ignore_errors=True)

    def _log(self, msg: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "msg": msg}) + "\n")
