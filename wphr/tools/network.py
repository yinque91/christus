"""Network tools: outbound-only HTTP(S), DNS, downloads into workspace."""

from __future__ import annotations

import socket
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

from wphr.tools.policy import Audit, Policy

USER_AGENT = "WPHR-Autonomy/1.0 (+local research agent)"


class NetworkTool:
    def __init__(self, policy: Policy | None = None, audit: Audit | None = None):
        self.policy = policy or Policy()
        self.audit = audit or Audit()

    def dns(self, host: str) -> dict:
        self.policy.check("net.dns_lookup")
        try:
            ip = socket.gethostbyname(host)
            out = {"ok": True, "host": host, "ip": ip}
        except Exception as e:
            out = {"ok": False, "host": host, "error": str(e)}
        self.audit.log("net.dns", host, str(out.get("ip") or out.get("error")), ok=out["ok"])
        return out

    def get(self, url: str, timeout: float = 15.0, max_bytes: int = 2_000_000) -> dict:
        self.policy.check("net.http_get")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise PermissionError("only http/https egress allowed")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                body = r.read(max_bytes)
                out = {"ok": True, "status": r.status, "url": url,
                       "content_type": r.headers.get("Content-Type", ""),
                       "bytes": len(body), "body": body}
        except Exception as e:
            out = {"ok": False, "url": url, "error": str(e)}
        self.audit.log("net.get", url, str(out.get("status") or out.get("error")),
                       ok=out["ok"])
        return out

    def download(self, url: str, dest: str, timeout: float = 60.0) -> dict:
        self.policy.check("net.download_to_workspace")
        p = Path(dest).resolve() if Path(dest).is_absolute() else (self.policy.workspace / dest).resolve()
        if not self.policy.inside(p):
            raise PermissionError("downloads must stay inside the workspace")
        res = self.get(url, timeout=timeout, max_bytes=50_000_000)
        if not res.get("ok"):
            return res
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(res["body"])
        self.audit.log("net.download", url, str(p), ok=True, artifacts=[str(p)])
        return {"ok": True, "path": str(p), "bytes": len(res["body"])}
