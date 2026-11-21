"""Watchdog: restart the subject service if its PID disappears."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR_workspace")
PY = r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR\.venv\Scripts\python.exe"
ROOT = r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR"
PID = WS / "service" / "run" / "wphr.pid"
LOG = WS / "logs" / "watchdog.jsonl"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    restarts = 0
    while True:
        time.sleep(5.0)
        if (WS / "service" / "run" / "STOP").exists():
            log({"ts": time.time(), "event": "watchdog_stop"})
            return
        pid = None
        if PID.exists():
            try:
                pid = int(PID.read_text().strip())
            except Exception:
                pid = None
        if pid and alive(pid):
            continue
        log({"ts": time.time(), "event": "restart", "old_pid": pid})
        subprocess.Popen([PY, "-m", "wphr.runtime.subject_service"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        restarts += 1


if __name__ == "__main__":
    main()
