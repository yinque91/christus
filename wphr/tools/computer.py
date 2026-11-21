"""Computer tools: sandboxed filesystem + process execution (audited)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from wphr.tools.policy import Audit, Policy, PermissionDenied


class ComputerTool:
    def __init__(self, policy: Policy | None = None, audit: Audit | None = None):
        self.policy = policy or Policy()
        self.audit = audit or Audit()

    # ------------------------------------------------------------- filesystem
    def _resolve(self, path: str) -> Path:
        q = Path(path)
        return (q if q.is_absolute() else self.policy.workspace / q).resolve()

    def read(self, path: str) -> str:
        p = self._resolve(path)
        if not self.policy.inside(p):
            self.policy.check("fs.read_outside_workspace")
        self.policy.check("fs.read_workspace")
        text = p.read_text(encoding="utf-8", errors="replace")
        self.audit.log("fs.read", str(p), f"{len(text)} chars", ok=True)
        return text

    def write(self, path: str, content: str) -> Path:
        p = self._resolve(path)
        if not self.policy.inside(p):
            self.policy.check("fs.read_outside_workspace")
        self.policy.check("fs.write_workspace")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.audit.log("fs.write", str(p), f"{len(content)} chars", ok=True, artifacts=[str(p)])
        return p

    def listdir(self, path: str | None = None) -> list[str]:
        p = self._resolve(path) if path else self.policy.workspace
        if not self.policy.inside(p):
            self.policy.check("fs.read_outside_workspace")
        self.policy.check("fs.list_workspace")
        out = sorted(x.name for x in p.iterdir())
        self.audit.log("fs.list", str(p), f"{len(out)} entries", ok=True)
        return out

    # --------------------------------------------------------------- processes
    def run_python(self, code: str, timeout: float = 30.0) -> dict:
        self.policy.check("proc.python")
        return self._run([sys.executable, "-c", code], timeout, "proc.python")

    def run_powershell(self, script: str, timeout: float = 30.0) -> dict:
        self.policy.check("proc.powershell")
        return self._run(["powershell", "-NoProfile", "-Command", script], timeout,
                         "proc.powershell")

    def _run(self, cmd: list[str], timeout: float, tool: str) -> dict:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               cwd=str(self.policy.workspace))
            out = {"ok": r.returncode == 0, "code": r.returncode,
                   "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            out = {"ok": False, "code": None, "stdout": "", "stderr": "timeout"}
        self.audit.log(tool, " ".join(cmd)[:300], str(out.get("code")),
                       ok=out["ok"], detail=out.get("stderr", ""))
        return out
