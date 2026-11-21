"""Control channel: pause/stop requests from another process.

``wphr run`` polls a small JSON file so that ``wphr pause`` / ``wphr stop`` from
a second shell can act on a running subject. The file is read with mtime caching
so polling costs nothing in the step loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wphr.util.ids import utc_now

VALID_COMMANDS = ("pause", "stop", "checkpoint", "reset_affect")


class ControlFile:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mtime: float | None = None
        self._state: dict[str, Any] = {"command": None}

    def request(self, command: str, note: str = "") -> dict[str, Any]:
        if command not in VALID_COMMANDS:
            raise ValueError(f"invalid control command {command!r}; use one of {VALID_COMMANDS}")
        payload = {"command": command, "note": note, "requested_at": utc_now()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self._mtime = None
        return payload

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            self._state = {"command": None}
            return self._state
        mtime = self.path.stat().st_mtime
        if self._mtime == mtime:
            return self._state
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {"command": None}
        self._mtime = mtime
        self._state = {
            "command": payload.get("command"),
            "note": payload.get("note", ""),
            "requested_at": payload.get("requested_at"),
        }
        return self._state

    def pending(self) -> str | None:
        command = self.read().get("command")
        return command if command in VALID_COMMANDS else None

    def clear(self) -> None:
        if self.path.is_file():
            self.path.unlink()
        self._mtime = None
        self._state = {"command": None}

    def state(self) -> dict[str, Any]:
        return dict(self.read())
