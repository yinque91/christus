"""Local HTTP server for the Christus platform (stdlib only)."""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from christus_platform.runtime import PlatformRuntime  # noqa: E402

STATIC = ROOT / "static"
RUNTIME: PlatformRuntime | None = None


def get_runtime():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = PlatformRuntime()
    return RUNTIME


def json_response(handler, code, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    server_version = "ChristusPlatform/1.0"

    def log_message(self, fmt, *args):
        line = "%s - %s\n" % (self.address_string(), fmt % args)
        try:
            (ROOT / "logs" / "server.log").open("a", encoding="utf-8").write(line)
        except Exception:
            pass
        if os.environ.get("CHRISTUS_VERBOSE"):
            sys.stderr.write(line)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _serve_static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            json_response(self, 404, {"error": "not found"})
            return
        data = target.read_bytes()
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path.startswith("/api/"):
                rt = get_runtime()
                if u.path == "/api/health":
                    return json_response(self, 200, {"ok": True, "service": "christus",
                                                     "time": time.time()})
                if u.path == "/api/state":
                    return json_response(self, 200, rt.public_state())
                if u.path == "/api/sessions":
                    return json_response(self, 200, {"sessions": rt.sessions(
                        q.get("user_id", ["local-user"])[0])})
                if u.path == "/api/history":
                    sid = q.get("session_id", [""])[0]
                    return json_response(self, 200, {"messages": rt.history(sid)})
                if u.path == "/api/learning":
                    return json_response(self, 200, {"events": rt.learning()})
                if u.path == "/api/knowledge":
                    return json_response(self, 200, {"items": rt.knowledge_list()})
                if u.path == "/api/settings":
                    return json_response(self, 200, {
                        "language": rt.language.name,
                        "external_bridge_configured": rt.external_language.enabled,
                        "internet": "Wikipedia/DuckDuckGo public endpoints",
                        "port": int(os.environ.get("CHRISTUS_PORT", "8765")),
                    })
                return json_response(self, 404, {"error": "unknown endpoint"})
            return self._serve_static(u.path)
        except Exception as exc:
            json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}",
                                      "trace": traceback.format_exc()})

    def do_POST(self):
        try:
            u = urlparse(self.path)
            body = self._read_json()
            rt = get_runtime()
            if u.path == "/api/chat":
                text = str(body.get("text", ""))
                out = rt.chat(text, session_id=body.get("session_id"),
                              user_id=str(body.get("user_id", "local-user")))
                return json_response(self, 200, out)
            if u.path == "/api/chat/stream":
                return self._stream_chat(body, rt)
            if u.path == "/api/checkpoint":
                return json_response(self, 200, {"checkpoint": rt.checkpoint()})
            if u.path == "/api/pause":
                return json_response(self, 200, {"paused": rt.pause(bool(body.get("paused", True)))})
            if u.path == "/api/backup":
                dest = ROOT / "backups" / time.strftime("backup-%Y%m%d-%H%M%S")
                dest.mkdir(parents=True, exist_ok=True)
                for name in ("platform.db", "runtime_state.json"):
                    src = ROOT / "state" / name
                    if src.exists():
                        shutil.copy2(src, dest / name)
                subj = ROOT / "state" / "subject"
                if subj.exists():
                    shutil.copytree(subj, dest / "subject", dirs_exist_ok=True)
                return json_response(self, 200, {"backup": str(dest)})
            return json_response(self, 404, {"error": "unknown endpoint"})
        except Exception as exc:
            json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}",
                                      "trace": traceback.format_exc()})

    def _stream_chat(self, body, rt):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            out = rt.chat(str(body.get("text", "")),
                          session_id=body.get("session_id"),
                          user_id=str(body.get("user_id", "local-user")))
            reply = out["reply"]
            # Chunk after the real Subject turn; this is a transport stream, not
            # a second generation pass.
            for i in range(0, len(reply), 18):
                chunk = reply[i:i + 18]
                self.wfile.write(("data: " + json.dumps(
                    {"delta": chunk, "session_id": out["session_id"]},
                    ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(("data: " + json.dumps(
                {"done": True, "session_id": out["session_id"],
                 "source": out["source"], "citations": out["citations"],
                 "state": out["state"]}, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()
        except Exception as exc:
            self.wfile.write(("data: " + json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n\n").encode("utf-8"))


def main():
    port = int(os.environ.get("CHRISTUS_PORT", "8765"))
    host = os.environ.get("CHRISTUS_HOST", "127.0.0.1")
    rt = get_runtime()
    server = ThreadingHTTPServer((host, port), Handler)
    (ROOT / "state" / "server.pid").write_text(str(os.getpid()), encoding="utf-8")
    print(f"CHRISTUS platform: http://{host}:{port}")
    print(f"language={rt.language.name} subject={rt.public_state().get('subject_id')}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        rt.close()


if __name__ == "__main__":
    main()
