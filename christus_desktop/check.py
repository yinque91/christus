"""Environment / data / desktop health check for CHECK_CHRISTUS.bat."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CHRISTUS-DATA-01"))


def desktop_alive() -> int | None:
    p = ROOT / "christus_desktop" / "state" / "desktop.pid"
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=5)
        if str(pid) in out.stdout:
            return pid
    except Exception:
        pass
    return None


def main():
    out = {"python": sys.version.split()[0], "root": str(ROOT)}
    out["tkinter"] = bool(importlib.util.find_spec("tkinter"))
    for name in ("numpy", "requests"):
        out[name] = bool(importlib.util.find_spec(name))
    running = desktop_alive()
    out["desktop_running"] = bool(running)
    out["desktop_pid"] = running
    out["web_server_file_present"] = (ROOT / "christus_platform" / "server.py").exists()
    out["web_server_running"] = False
    if running:
        out["runtime_check"] = "skipped: desktop owns the Subject (do not create a second Subject)"
    else:
        try:
            from christus_platform.runtime import PlatformRuntime
            rt = PlatformRuntime()
            s = rt.public_state()
            out.update({
                "subject_id": s.get("subject_id"), "step": s.get("step"),
                "turns": s.get("turns"),
                "core_weights_unchanged": s.get("core_weights_unchanged"),
                "language": s.get("language_source"),
                "transformer": s.get("transformer"),
                "memory": s.get("memory"),
                "checkpoint": bool(s.get("last_checkpoint")),
            })
            rt.close()
        except Exception as exc:
            out["runtime_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("tkinter") and "runtime_error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
