"""Permission tests; no real keyboard/mouse input is sent."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from christus_autonomy.safety import SafetyBroker, SafetyError, WindowInfo

ROOT = Path(__file__).resolve().parents[2]


def broker(tmp):
    cfg = json.loads((ROOT / "christus_autonomy" / "permissions.json").read_text(encoding="utf-8"))
    return SafetyBroker(config_path=ROOT / "christus_autonomy" / "permissions.json",
                        state_dir=tmp)


def test_allowed_notepad_action():
    with tempfile.TemporaryDirectory() as d:
        b = broker(d)
        b.check("move", foreground=WindowInfo(title="Untitled - Notepad", process="notepad.exe"))
        assert (Path(d) / "audit.jsonl").exists()


def test_blocked_process():
    with tempfile.TemporaryDirectory() as d:
        b = broker(d)
        try:
            b.check("click", foreground=WindowInfo(title="PowerShell", process="powershell.exe"))
        except SafetyError:
            pass
        else:
            raise AssertionError("powershell should be blocked")


def test_blocked_title():
    with tempfile.TemporaryDirectory() as d:
        b = broker(d)
        try:
            b.check("type", foreground=WindowInfo(title="Bank login", process="chrome.exe"))
        except SafetyError:
            pass
        else:
            raise AssertionError("bank login title should be blocked")


def test_stop_blocks_everything():
    with tempfile.TemporaryDirectory() as d:
        b = broker(d)
        b.stop("test")
        try:
            b.check("move", foreground=WindowInfo(title="Untitled - Notepad", process="notepad.exe"))
        except SafetyError:
            pass
        else:
            raise AssertionError("stop must block actions")
        b.clear_stop()
        b.check("move", foreground=WindowInfo(title="Untitled - Notepad", process="notepad.exe"))


def test_rate_limit():
    with tempfile.TemporaryDirectory() as d:
        b = broker(d)
        limit = int(b.config["max_actions_per_minute"])
        fg = WindowInfo(title="Untitled - Notepad", process="notepad.exe")
        for _ in range(limit):
            b.check("move", foreground=fg)
        try:
            b.check("move", foreground=fg)
        except SafetyError:
            pass
        else:
            raise AssertionError("rate limit not enforced")

