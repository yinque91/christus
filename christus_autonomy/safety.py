"""Permission broker and audit trail for the desktop actuator.

The broker is deliberately conservative: keyboard/mouse actions are allowed only
when the foreground process is on the allowlist, the title is not a blocked
sensitive surface, the emergency-stop file is absent, and the rate limit is not
exceeded.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SafetyError(PermissionError):
    """Raised when an action is outside the current permission policy."""


@dataclass
class WindowInfo:
    hwnd: int = 0
    title: str = ""
    pid: int = 0
    process: str = ""


class SafetyBroker:
    INPUT_ACTIONS = {"move", "click", "doubleclick", "scroll", "type", "key", "hotkey"}

    def __init__(self, config_path: str | Path | None = None,
                 state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir or Path(os.environ.get(
            "CHRISTUS_AUTONOMY_STATE", str(Path.home() / ".christus_autonomy"))))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path or (Path(__file__).with_name("permissions.json")))
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.stop_file = self.state_dir / "STOP"
        self.audit_file = self.state_dir / "audit.jsonl"
        self._actions: deque[float] = deque()
        self.allowed = {x.casefold() for x in self.config.get("allowed_processes", [])}
        self.blocked = {x.casefold() for x in self.config.get("blocked_processes", [])}
        self.launch_allowed = {x.casefold() for x in self.config.get("allowed_launch", [])}
        self.blocked_titles = [x.casefold() for x in self.config.get("blocked_titles", [])]
        self.blocked_shortcuts = {x.casefold().replace(" ", "") for x in self.config.get("blocked_shortcuts", [])}

    # ------------------------------------------------------------------ state
    def emergency_stopped(self) -> bool:
        return self.stop_file.exists()

    def stop(self, reason: str = "manual"):
        self.stop_file.write_text(reason, encoding="utf-8")
        self.audit("emergency_stop", allowed=False, reason=reason)

    def clear_stop(self):
        if self.stop_file.exists():
            self.stop_file.unlink()

    # ------------------------------------------------------------------ audit
    def audit(self, action: str, *, allowed: bool, reason: str = "", **extra: Any):
        row = {
            "ts": time.time(),
            "action": action,
            "allowed": bool(allowed),
            "reason": reason,
            "extra": extra,
        }
        try:
            with self.audit_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ checks
    def check(self, action: str, *, foreground: WindowInfo | None = None,
              shortcut: str = "", executable: str = "", **extra: Any) -> None:
        if self.emergency_stopped():
            self.audit(action, allowed=False, reason="emergency_stop", **extra)
            raise SafetyError("emergency stop is active; run `christus_autonomy clear-stop`")

        if action in self.INPUT_ACTIONS:
            if foreground is None:
                from .perception import foreground as get_foreground
                foreground = get_foreground()
            process = (foreground.process or "").casefold()
            title = (foreground.title or "").casefold()
            if not process:
                self.audit(action, allowed=False, reason="no_foreground", **extra)
                raise SafetyError("no foreground window")
            if process in self.blocked:
                self.audit(action, allowed=False, reason=f"blocked_process:{process}", **extra)
                raise SafetyError(f"process is blocked: {foreground.process}")
            if process not in self.allowed:
                self.audit(action, allowed=False, reason=f"process_not_allowed:{process}", **extra)
                raise SafetyError(f"process is not allowlisted: {foreground.process}")
            if any(p in title for p in self.blocked_titles):
                self.audit(action, allowed=False, reason="blocked_title", **extra)
                raise SafetyError("the foreground title is a protected surface")
            if action == "hotkey":
                combo = shortcut.casefold().replace(" ", "")
                if combo in self.blocked_shortcuts:
                    self.audit(action, allowed=False, reason=f"blocked_shortcut:{combo}", **extra)
                    raise SafetyError(f"shortcut is blocked: {shortcut}")

        if action == "launch":
            exe = Path(executable).name.casefold()
            if exe not in self.launch_allowed:
                self.audit(action, allowed=False, reason=f"launch_not_allowed:{exe}", **extra)
                raise SafetyError(f"launch is not allowlisted: {executable}")
            if exe in self.blocked:
                self.audit(action, allowed=False, reason=f"blocked_process:{exe}", **extra)
                raise SafetyError(f"process is blocked: {executable}")

        if action in self.INPUT_ACTIONS:
            self._rate_limit()
        self.audit(action, allowed=True, reason="allowed", **extra)

    def _rate_limit(self):
        now = time.monotonic()
        while self._actions and now - self._actions[0] > 60:
            self._actions.popleft()
        limit = int(self.config.get("max_actions_per_minute", 90))
        if len(self._actions) >= limit:
            self.audit("rate_limit", allowed=False, reason=f"limit={limit}")
            raise SafetyError(f"action rate limit exceeded ({limit}/min)")
        self._actions.append(now)
