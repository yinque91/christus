"""Tool-layer permission enforcement + audit log."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class PermissionDenied(PermissionError):
    pass


class ApprovalRequired(PermissionError):
    pass


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
]


def redact(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub(r"\1=***REDACTED***", text)
    return text


class Policy:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "permission_policy.yaml"
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) if (yaml and self.path.exists()) else {}
        if not data:
            data = {
                "workspace_root": r"C:/Users/13394/Documents/deepseek-harness/default-workspace/WPHR_workspace",
                "auto_allowed": ["fs.read_workspace", "fs.write_workspace", "fs.list_workspace",
                                  "proc.python", "proc.powershell", "net.http_get", "net.dns_lookup",
                                  "net.download_to_workspace", "device.enumerate", "device.camera_read"],
                "approval_required": ["fs.read_outside_workspace", "net.http_post", "net.upload",
                                       "proc.install", "proc.admin", "device.microphone", "account.create_user"],
                "denied": ["credential.theft", "security.bypass", "net.scan_lan", "audit.delete"],
            }
        self.auto = set(data.get("auto_allowed", []))
        self.approval = set(data.get("approval_required", []))
        self.denied = set(data.get("denied", []))
        self.workspace = Path(data.get("workspace_root",
                                       Path.home() / "WPHR_workspace")).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def check(self, action: str) -> None:
        if action in self.denied:
            raise PermissionDenied("denied by policy: " + action)
        if action in self.auto:
            return
        if action in self.approval:
            raise ApprovalRequired("requires user approval: " + action)
        raise PermissionDenied("unknown action, default deny: " + action)

    def inside(self, p: str | Path) -> bool:
        try:
            Path(p).resolve().relative_to(self.workspace)
            return True
        except Exception:
            return False


class Audit:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "logs" / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, tool: str, target: str, result: str, *, ok: bool,
            detail: str = "", artifacts: list[str] | None = None) -> dict[str, Any]:
        arts = []
        for a in (artifacts or []):
            pa = Path(a)
            digest = None
            if pa.exists() and pa.is_file():
                digest = hashlib.sha256(pa.read_bytes()).hexdigest()
            arts.append({"path": a, "sha256": digest})
        rec: dict[str, Any] = {
            "ts": time.time(),
            "tool": tool,
            "target": redact(str(target)),
            "ok": bool(ok),
            "result": redact(result)[:2000],
            "detail": redact(detail)[:2000],
            "artifacts": arts,
        }
        payload = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        rec["record_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
