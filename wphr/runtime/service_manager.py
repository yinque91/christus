"""Service manager: single instance, health, metrics, rotation, graceful stop."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path


class ServiceManager:
    def __init__(self, workspace: str | Path):
        self.ws = Path(workspace)
        self.run = self.ws / "run"
        self.logs = self.ws / "logs"
        self.run.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.run / "wphr.pid"
        self.stop_file = self.run / "STOP"
        self.pause_file = self.run / "PAUSE"
        self.status_file = self.run / "status.json"
        self.log_file = self.logs / "service.log"

    # ------------------------------------------------------------ singleton
    def acquire(self) -> bool:
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, 0)
                return False
            except (ValueError, OSError):
                pass
        self.pid_file.write_text(str(os.getpid()))
        return True

    def release(self) -> None:
        try:
            if self.pid_file.exists() and int(self.pid_file.read_text()) == os.getpid():
                self.pid_file.unlink()
        except Exception:
            pass

    # -------------------------------------------------------------- control
    def paused(self) -> bool:
        return self.pause_file.exists()

    def stop_requested(self) -> bool:
        return self.stop_file.exists()

    def request_stop(self) -> None:
        self.stop_file.write_text("1")
        self.log("stop requested")

    def request_pause(self, on: bool = True) -> None:
        if on:
            self.pause_file.write_text("1")
        elif self.pause_file.exists():
            self.pause_file.unlink()

    # --------------------------------------------------------------- health
    def _proc_metrics(self) -> dict:
        try:
            import psutil
            p = psutil.Process()
            mem = p.memory_info()
            return {"rss_mb": round(mem.rss / 1e6, 1),
                    "cpu_pct": p.cpu_percent(interval=None),
                    "threads": p.num_threads()}
        except Exception:
            return {}

    def _disk(self) -> dict:
        u = shutil.disk_usage(self.ws)
        return {"free_gb": round(u.free / 1e9, 2),
                "used_pct": round(100 * u.used / u.total, 1)}

    def write_status(self, extra: dict | None = None) -> dict:
        st = {"ts": time.time(), "pid": os.getpid(), "paused": self.paused(),
              "stop": self.stop_requested(), "disk": self._disk()}
        st.update(self._proc_metrics())
        if extra:
            st.update(extra)
        self.status_file.write_text(json.dumps(st, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        return st

    def read_status(self) -> dict:
        if self.status_file.exists():
            return json.loads(self.status_file.read_text(encoding="utf-8"))
        return {}

    # ----------------------------------------------------------------- logs
    def log(self, msg: str) -> None:
        line = json.dumps({"ts": time.time(), "pid": os.getpid(), "msg": msg},
                          ensure_ascii=False)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.rotate()

    def rotate(self, max_bytes: int = 2_000_000, keep: int = 5) -> None:
        if not self.log_file.exists() or self.log_file.stat().st_size < max_bytes:
            return
        for i in range(keep - 1, 0, -1):
            src = self.logs / f"service.log.{i}"
            dst = self.logs / f"service.log.{i+1}"
            if src.exists():
                shutil.move(str(src), str(dst))
        shutil.move(str(self.log_file), str(self.logs / "service.log.1"))

    # --------------------------------------------------------------- signal
    def install_signals(self) -> None:
        def handler(signum, frame):
            self.request_stop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
