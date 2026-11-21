"""Start/stop the native desktop app; no web server, no browser."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "christus_desktop"
STATE = DESKTOP / "state"
LOGS = DESKTOP / "logs"
PID = STATE / "desktop.pid"
PY = Path(r"C:\Users\13394\AppData\Local\Programs\Python\Python312\python.exe")
PYW = PY.with_name("pythonw.exe")
SITE = Path(r"C:\Users\13394\Documents\deepseek-harness\default-workspace\WPHR\.venv\Lib\site-packages")


def alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=5)
        return str(pid) in out.stdout
    except Exception:
        return False


def kill(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, text=True, timeout=8)
    except Exception:
        pass


def stop_web():
    p = ROOT / "christus_platform" / "state" / "server.pid"
    if p.exists():
        try:
            pid = int(p.read_text().strip())
            kill(pid)
        except Exception:
            pass
        try:
            p.unlink()
        except Exception:
            pass


def start():
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            old = int(PID.read_text().strip())
            if alive(old):
                print(f"Christus desktop is already running (pid {old}).")
                return 0
        except Exception:
            pass
    stop_web()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SITE) + os.pathsep + str(ROOT) + os.pathsep + str(ROOT / "CHRISTUS-DATA-01")
    out = (LOGS / "desktop.out.log").open("a", encoding="utf-8")
    err = (LOGS / "desktop.err.log").open("a", encoding="utf-8")
    exe = str(PYW if PYW.exists() else PY)
    p = subprocess.Popen([exe, str(DESKTOP / "app.py")], cwd=str(ROOT),
                         env=env, stdout=out, stderr=err,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    PID.write_text(str(p.pid), encoding="ascii")
    time.sleep(1.2)
    if p.poll() is not None:
        print("Desktop app failed to start. See christus_desktop/logs/desktop.err.log")
        return 1
    print(f"Christus desktop started (pid {p.pid}).")
    print("A native window should open; no browser or localhost page is used.")
    return 0


def stop():
    if PID.exists():
        try:
            pid = int(PID.read_text().strip())
            kill(pid)
            print(f"Stopped Christus desktop pid {pid}.")
        except Exception as exc:
            print("Stop error:", exc)
        try:
            PID.unlink()
        except Exception:
            pass
    else:
        print("No desktop pid file.")
    stop_web()
    return 0


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "start"
    raise SystemExit(start() if action == "start" else stop())
