"""Browser control via dedicated Chrome profile (headless page fetch + screenshots)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from wphr.tools.policy import Audit, Policy

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


class Browser:
    def __init__(self, policy: Policy | None = None, audit: Audit | None = None):
        self.policy = policy or Policy()
        self.audit = audit or Audit()
        self.exe = next((p for p in CHROME_CANDIDATES if Path(p).exists()), None)
        self.profile = Path(__file__).resolve().parents[2] / "storage" / "browser_profile"
        self.profile.mkdir(parents=True, exist_ok=True)
        if self.exe is None:
            raise RuntimeError("no supported browser found")

    def _run(self, args: list[str], timeout: float = 45.0) -> subprocess.CompletedProcess:
        cmd = [self.exe, f"--user-data-dir={self.profile}", "--no-first-run",
               "--disable-gpu", "--headless=new", *args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def fetch(self, url: str, timeout: float = 45.0) -> dict:
        self.policy.check("net.http_get")
        r = self._run(["--dump-dom", url], timeout)
        ok = r.returncode == 0 and bool(r.stdout)
        self.audit.log("browser.fetch", url, f"{len(r.stdout)} chars", ok=ok,
                       detail=(r.stderr or "")[-500:])
        return {"ok": ok, "url": url, "html": r.stdout[:200_000],
                "chars": len(r.stdout)}

    def screenshot(self, url: str, out: str, size: str = "1280,800") -> dict:
        self.policy.check("net.http_get")
        p = Path(out).resolve() if Path(out).is_absolute() else (self.policy.workspace / out).resolve()
        if not self.policy.inside(p):
            raise PermissionError("screenshot must stay inside workspace")
        p.parent.mkdir(parents=True, exist_ok=True)
        r = self._run([f"--window-size={size}", f"--screenshot={p}", url])
        ok = p.exists() and p.stat().st_size > 0
        self.audit.log("browser.screenshot", url, str(p), ok=ok, artifacts=[str(p)])
        return {"ok": ok, "path": str(p), "bytes": p.stat().st_size if p.exists() else 0}

    def download(self, url: str, dest: str) -> dict:
        from wphr.tools.network import NetworkTool
        return NetworkTool(self.policy, self.audit).download(url, dest)
