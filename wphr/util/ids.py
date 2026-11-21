"""Identifier and clock helpers."""

from __future__ import annotations

import time
import uuid


def new_id(prefix: str = "") -> str:
    """A 32-hex-character identifier, optionally prefixed (``evt_...``)."""
    body = uuid.uuid4().hex
    return f"{prefix}{body}" if prefix else body


def utc_now() -> float:
    """Wall-clock UNIX time in seconds (UTC)."""
    return time.time()


def utc_iso(ts: float | None = None) -> str:
    """ISO-8601 UTC timestamp with millisecond precision."""
    t = time.time() if ts is None else ts
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"
