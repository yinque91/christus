"""Persistent autonomy supervisor and emergency hotkey.

The supervisor is intentionally conservative. It observes continuously and can
execute an external action queue; it does not invent arbitrary clicks. The action
queue is the integration point for a Christus policy.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .actuator import DesktopActuator
from .memory import ExperienceStore, NoveltyModel
from .perception import foreground, list_windows, screenshot
from .safety import SafetyBroker, SafetyError

user32 = ctypes.WinDLL("user32", use_last_error=True)
MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
MOD_SHIFT = 0x0004
WM_HOTKEY = 0x0312


class MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD), ("pt", wintypes.POINT)]


class EmergencyHotkey(threading.Thread):
    """Ctrl+Alt+Shift+Q writes the STOP file from a background message loop."""

    def __init__(self, stop_path: Path):
        super().__init__(daemon=True)
        self.stop_path = stop_path
        self._registered = False

    def run(self):
        try:
            self._registered = bool(user32.RegisterHotKey(
                None, 1, MOD_CONTROL | MOD_ALT | MOD_SHIFT, 0x51
            ))
            if not self._registered:
                return
            msg = MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0):
                if msg.message == WM_HOTKEY:
                    self.stop_path.write_text("emergency hotkey", encoding="utf-8")
        finally:
            if self._registered:
                user32.UnregisterHotKey(None, 1)


class AutonomySupervisor:
    def __init__(self, *, state_dir: str | Path | None = None,
                 mode: str = "observe", interval: float = 2.0,
                 screenshots: bool = False):
        self.state_dir = Path(state_dir or Path(os.environ.get(
            "CHRISTUS_AUTONOMY_STATE", str(Path.home() / ".christus_autonomy"))))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.state_dir / "supervisor.pid"
        self.stop_path = self.state_dir / "STOP"
        self.queue_path = self.state_dir / "actions.jsonl"
        self.result_path = self.state_dir / "action-results.jsonl"
        self.broker = SafetyBroker(state_dir=self.state_dir)
        self.actuator = DesktopActuator(self.broker)
        self.memory = ExperienceStore(self.state_dir)
        self.novelty = NoveltyModel(self.state_dir)
        self.mode = mode
        self.interval = max(0.25, float(interval))
        self.screenshots = screenshots
        self._queue_offset = 0
        self._running = False
        self._hotkey = None

    def _alive(self, pid: int) -> bool:
        import subprocess
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True, timeout=5)
            return str(pid) in out.stdout
        except Exception:
            return False

    def status(self) -> dict:
        pid = None
        if self.pid_path.exists():
            try:
                pid = int(self.pid_path.read_text(encoding="utf-8").strip())
            except ValueError:
                pid = None
        return {
            "pid": pid,
            "alive": bool(pid and self._alive(pid)),
            "stop": self.stop_path.exists(),
            "mode": self.mode,
            "experience_rows": self.memory.count(),
            "novelty": self.novelty.snapshot(),
        }

    def _drain_queue(self):
        if not self.queue_path.exists():
            return
        size = self.queue_path.stat().st_size
        if size <= self._queue_offset:
            return
        with self.queue_path.open("r", encoding="utf-8") as f:
            f.seek(self._queue_offset)
            lines = f.readlines()
            self._queue_offset = f.tell()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                action = row.pop("action")
                if action == "move":
                    self.actuator.move(int(row["x"]), int(row["y"]))
                elif action == "click":
                    self.actuator.click(row.get("x"), row.get("y"),
                                         button=row.get("button", "left"))
                elif action == "doubleclick":
                    self.actuator.click(row.get("x"), row.get("y"), double=True)
                elif action == "scroll":
                    self.actuator.scroll(int(row["amount"]))
                elif action == "type":
                    self.actuator.type_text(str(row["text"]))
                elif action == "key":
                    self.actuator.key(str(row["key"]), int(row.get("presses", 1)))
                elif action == "hotkey":
                    self.actuator.hotkey(str(row["combo"]))
                elif action == "activate":
                    self.actuator.activate(str(row["title"]))
                elif action == "launch":
                    self.actuator.launch(str(row["executable"]))
                else:
                    raise ValueError(f"unknown queued action: {action}")
                result = {"ts": time.time(), "action": action, "ok": True}
            except Exception as exc:
                result = {"ts": time.time(), "action": "error", "ok": False,
                          "error": str(exc)}
            with self.result_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def run(self):
        if self.pid_path.exists():
            try:
                old = int(self.pid_path.read_text(encoding="utf-8").strip())
                if self._alive(old):
                    raise RuntimeError(f"supervisor already running (pid {old})")
            except ValueError:
                pass
        if self.stop_path.exists():
            raise RuntimeError("emergency stop is active; run `christus_autonomy clear-stop`")
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self._hotkey = EmergencyHotkey(self.stop_path)
        self._hotkey.start()
        self._running = True
        try:
            while not self.stop_path.exists():
                obs = foreground()
                novelty = self.novelty.score({
                    "process": obs.process, "title": obs.title, "pid": obs.pid,
                    "visible_windows": len(list_windows()),
                })
                self.memory.record({"process": obs.process, "title": obs.title,
                                    "pid": obs.pid, "mode": self.mode},
                                   novelty=novelty)
                if self.screenshots:
                    screenshot(self.state_dir / "screens" / f"{int(time.time())}.png")
                if self.mode == "control":
                    self._drain_queue()
                self.novelty.save()
                time.sleep(self.interval)
        finally:
            self._running = False
            self.novelty.save()
            self.pid_path.unlink(missing_ok=True)

    def stop(self):
        self.stop_path.write_text("manual stop", encoding="utf-8")
