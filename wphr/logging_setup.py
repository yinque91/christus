"""Logging setup: console + rotating text log + JSONL audit log.

Audit log lines are append-only JSON objects with an explicit ``audit`` flag so
that safety-relevant actions (device access, sandbox writes, checkpoint
restore, permission changes) remain traceable even if the main log is rotated.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


def setup_logging(
    log_dir: Path | str,
    level: str = "INFO",
    run_id: str = "run",
    console: bool = False,
) -> logging.Logger:
    """Configure the ``wphr`` logger tree once per process.

    Console output is opt-in (``runtime.console_log``) because the CLI already
    prints user-facing results; duplicating every log line onto stderr makes the
    tool awkward to script against.
    """
    global _CONFIGURED
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wphr")
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric)
    if _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path / f"wphr_{run_id}.log", maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"wphr.{name}")


class AuditLog:
    """Append-only JSONL audit trail for safety- and state-relevant actions."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, **fields: Any) -> dict[str, Any]:
        from wphr.util.ids import utc_iso, utc_now

        entry = {"ts": utc_now(), "iso": utc_iso(), "audit": True, "action": action}
        for key, value in fields.items():
            if isinstance(value, Path):
                value = str(value)
            entry[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        get_logger("audit").info("%s %s", action, fields or "")
        return entry

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
