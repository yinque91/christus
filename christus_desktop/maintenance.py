"""Backup/restore for the local desktop app (stdlib only)."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "christus_platform" / "state"
BACKUPS = ROOT / "backups"


def backup():
    dest = BACKUPS / time.strftime("desktop-%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("platform.db", "runtime_state.json"):
        src = STATE / name
        if src.exists():
            shutil.copy2(src, dest / name)
    subj = STATE / "subject"
    if subj.exists():
        shutil.copytree(subj, dest / "subject", dirs_exist_ok=True)
    print(dest)


def restore(path: str | None):
    choices = sorted([p for p in BACKUPS.iterdir() if p.is_dir()]) if BACKUPS.exists() else []
    if path:
        src = Path(path)
    elif choices:
        src = choices[-1]
    else:
        raise SystemExit("no backup")
    for name in ("platform.db", "runtime_state.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, STATE / name)
    if (src / "subject").exists():
        shutil.copytree(src / "subject", STATE / "subject", dirs_exist_ok=True)
    print(src)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: maintenance.py backup|restore [path]")
    if sys.argv[1] == "backup":
        backup()
    elif sys.argv[1] == "restore":
        restore(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        raise SystemExit("unknown action")
