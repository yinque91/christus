"""Local read-only web UI for observing a WPHR subject.

Safety properties, all enforced in code:

* binds to 127.0.0.1 only — never 0.0.0.0, never a LAN address;
* read-only: the server never writes to the database, checkpoints or files;
* no external assets: the dashboard is a single embedded HTML file served from
  disk, so it works offline;
* no browser is launched automatically (it would steal keyboard focus).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from wphr.config import Config, project_root
from wphr.logging_setup import get_logger
from wphr.memory.checkpoint import CheckpointManager
from wphr.memory.store import Store
from wphr.version import __version__

LOOPBACK = "127.0.0.1"

CELL_GLYPHS = {
    0: "#",  # wall
    1: ".",  # floor
    2: "F",  # food
    3: "!",  # hazard
    4: "B",  # switch blue
    5: "O",  # switch orange
    6: "T",  # tool
}
CELL_NAMES = {0: "wall", 1: "floor", 2: "food", 3: "hazard", 4: "switch_blue",
              5: "switch_orange", 6: "tool"}


class Dashboard:
    """Assembles the read-only view model from persisted state."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = Store(
            config.resolve_path("paths.database"),
            timeout=float(config.get("memory.db_timeout_s", 30)),
        )
        self.manager = CheckpointManager(self.store, config)
        self.log = get_logger("ui")

    def status_snapshot(self) -> dict[str, Any] | None:
        path = self.config.resolve_path("paths.status")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def latest_checkpoint_meta(self) -> dict[str, Any] | None:
        try:
            checkpoint = self.manager.load("latest", verify=False, allow_config_mismatch=True)
        except Exception:
            return None
        return checkpoint.meta

    def world_view(self) -> dict[str, Any] | None:
        meta = self.latest_checkpoint_meta()
        if not meta:
            return None
        world = meta.get("world_state") or {}
        grid_map = world.get("map")
        if not grid_map:
            return None
        rows = grid_map["rows"]
        body = world.get("body", {})
        bx, by = int(body.get("x", 0)), int(body.get("y", 0))
        heading = str(body.get("heading", "east"))
        ascii_rows: list[str] = []
        for y, row in enumerate(rows):
            line = []
            for x, code in enumerate(row):
                if (x, y) == (bx, by):
                    line.append(_heading_glyph(heading))
                else:
                    line.append(CELL_GLYPHS.get(int(code), "?"))
            ascii_rows.append("".join(line))
        return {
            "map_id": grid_map.get("map_id"),
            "width": grid_map.get("width"),
            "height": grid_map.get("height"),
            "ascii": ascii_rows,
            "body": {"x": bx, "y": by, "heading": heading,
                     "energy": round(float(body.get("energy", 0.0)), 4)},
            "step": world.get("step"),
            "episode_reward": round(float(world.get("episode_reward", 0.0)), 4),
            "legend": CELL_NAMES,
        }

    def state(self) -> dict[str, Any]:
        snapshot = self.status_snapshot() or {}
        meta = self.latest_checkpoint_meta() or {}
        modules = meta.get("modules") or {}
        concept = modules.get("concept_language") or {}
        graph = concept.get("graph") or {}
        beliefs = (concept.get("state") or {}).get("beliefs") or {}
        payload = {
            "version": __version__,
            "config_hash": self.config.hash(),
            "snapshot": snapshot,
            "database": self.store.quick_stats(),
            "memory_layers": self.store.memory_counts(),
            "summaries": self.store.list_summaries(limit=3),
            "semantic_rules": self.store.list_memories(layer="semantic", limit=10),
            "checkpoints": self.store.list_checkpoints(limit=5),
            "world": self.world_view(),
            "concept_stats": {
                "nodes": len(graph.get("nodes", {})),
                "interventions": int(graph.get("interventions", 0)),
                "splits": int(graph.get("splits", 0)),
                "revisions": int(graph.get("revisions", 0)),
                "observations": int(graph.get("observations", 0)),
            },
            "concept_beliefs": beliefs,
            "checkpoint_meta": {
                "step": meta.get("step"),
                "modules": list(modules),
                "created_at": meta.get("created_at"),
            },
        }
        return payload

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.store.recent_events(limit=limit)]


def _heading_glyph(heading: str) -> str:
    return {"north": "^", "east": ">", "south": "v", "west": "<"}.get(heading, "@")


def make_handler(dashboard: Dashboard, dashboard_html: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"wphr-ui/{__version__}"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, dashboard_html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                self._send(200, json.dumps(dashboard.state(), default=str).encode("utf-8"),
                           "application/json")
                return
            if parsed.path == "/api/events":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["20"])[0])
                self._send(200, json.dumps(dashboard.events(limit), default=str).encode("utf-8"),
                           "application/json")
                return
            if parsed.path == "/api/health":
                self._send(200, b'{"ok": true}', "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter stderr
            get_logger("ui").debug(fmt, *args)

    return Handler


def serve(port: int = 8765, host: str = LOOPBACK) -> int:
    config = Config.load()
    html_path = Path(__file__).with_name("dashboard.html")
    if not html_path.is_file():
        raise FileNotFoundError(f"dashboard asset missing: {html_path}")
    dashboard = Dashboard(config)
    handler = make_handler(dashboard, html_path.read_text(encoding="utf-8"))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"WPHR UI listening on {url}")
    print("read-only; binds loopback only; no browser was opened automatically")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping UI")
    finally:
        server.server_close()
    return 0


def serve_in_background(port: int = 8765) -> ThreadingHTTPServer:  # pragma: no cover - test helper
    config = Config.load()
    html = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    server = ThreadingHTTPServer((LOOPBACK, port), make_handler(Dashboard(config), html))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
