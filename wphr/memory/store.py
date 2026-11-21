"""SQLite persistence: events, episodes, checkpoints, health and runs.

Design rules taken from MASTER SPEC 6:
* the episodic log is append-only; raw records are never silently rewritten;
* every event keeps its source, provenance and privacy class;
* schema version, backup and restore are first-class operations.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from wphr.core.event import Event
from wphr.memory.schema import MIGRATIONS, table_rows
from wphr.util.ids import new_id, utc_now
from wphr.version import SCHEMA_VERSION


class StoreError(RuntimeError):
    pass


class Store:
    """Thin, explicit SQLite wrapper (no ORM, no hidden state)."""

    def __init__(self, db_path: str | Path, timeout: float = 30.0) -> None:
        self.path = Path(db_path)
        self.timeout = float(timeout)

    # ---------------------------------------------------------------- connect
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    # --------------------------------------------------------------- migrate
    def version(self) -> int:
        if not self.exists:
            return 0
        with self.connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self, target: int = SCHEMA_VERSION) -> int:
        """Apply migrations up to ``target``; returns the resulting version."""
        if target < 1 or target > SCHEMA_VERSION:
            raise StoreError(f"unsupported target schema version: {target}")
        current = self.version()
        if current > SCHEMA_VERSION:
            raise StoreError(
                f"database schema v{current} is newer than this code (v{SCHEMA_VERSION}); "
                "refusing to downgrade"
            )
        if current == target:
            return current
        with self.connect() as conn:
            for version in range(current + 1, target + 1):
                for statement in MIGRATIONS[version]:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {version}")
        return target

    def init(self) -> int:
        return self.migrate(SCHEMA_VERSION)

    def require_initialised(self) -> None:
        if not self.exists:
            raise StoreError(f"database not initialised: {self.path} (run `wphr init`)")
        if self.version() < SCHEMA_VERSION:
            raise StoreError(
                f"database schema v{self.version()} is older than code v{SCHEMA_VERSION}; "
                "run `wphr init --migrate`"
            )

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        if not self.exists:
            return default
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    # --------------------------------------------------------------- memories
    def upsert_memory(self, record: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO memories(memory_id, layer, kind, key, content, source, confidence, "
                "importance, support_count, counter_count, scope, created_step, updated_step, "
                "created_at, updated_at, last_access, access_count, active) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET content=excluded.content, "
                "confidence=excluded.confidence, importance=excluded.importance, "
                "support_count=excluded.support_count, counter_count=excluded.counter_count, "
                "updated_step=excluded.updated_step, updated_at=excluded.updated_at, "
                "active=excluded.active",
                record.to_row(),
            )

    def list_memories(
        self, *, layer: str | None = None, limit: int = 50, active_only: bool = True
    ) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        query = "SELECT * FROM memories"
        clauses: list[str] = []
        params: list[Any] = []
        if layer:
            clauses.append("layer = ?")
            params.append(layer)
        if active_only:
            clauses.append("active = 1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def memory_counts(self) -> dict[str, int]:
        if not self.exists:
            return {}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT layer, COUNT(*) AS n FROM memories WHERE active=1 GROUP BY layer"
            ).fetchall()
        return {row["layer"]: int(row["n"]) for row in rows}

    def log_consolidation(self, *, step: int, layer: str, written: int, note: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO consolidation_log(timestamp, step, layer, written, note) "
                "VALUES(?,?,?,?,?)",
                (utc_now(), step, layer, written, note),
            )

    def consolidation_history(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM consolidation_log ORDER BY seq DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def add_summary(self, entry: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO summaries(summary_id, layer, start_step, end_step, text, source, "
                "created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    entry["summary_id"],
                    entry.get("layer", "autobiographical"),
                    int(entry["start_step"]),
                    int(entry["end_step"]),
                    entry["text"],
                    entry.get("source", "consolidation"),
                    float(entry.get("created_at", utc_now())),
                ),
            )

    def list_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------- crash recovery
    def recover_runs(self) -> int:
        """Mark runs left in 'running' by a crashed process as 'crashed'."""
        if not self.exists:
            return 0
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status='crashed', ended_at=? WHERE status='running'",
                (utc_now(),),
            )
            return int(cursor.rowcount or 0)

    def prune_events(self, keep_last: int) -> int:
        """Delete all but the newest ``keep_last`` events. Irreversible."""
        if not self.exists:
            return 0
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(seq) FROM events").fetchone()
            max_seq = int(row[0] or 0)
            cutoff = max_seq - int(keep_last)
            if cutoff <= 0:
                return 0
            cursor = conn.execute("DELETE FROM events WHERE seq <= ?", (cutoff,))
            return int(cursor.rowcount or 0)

    def archive_and_prune(
        self, keep_last: int, archive_dir: str | Path, batch_size: int = 2000
    ) -> dict[str, Any]:
        """Move the oldest events into a gzipped JSONL archive, then delete them.

        Unlike :meth:`prune_events` this is lossless: the archive keeps every
        removed event with its full payload, so a long run stays bounded without
        destroying raw data. The caller decides where archives live.
        """
        if not self.exists:
            return {"archived": 0, "path": None, "kept": 0}
        directory = Path(archive_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            max_seq_row = conn.execute("SELECT MAX(seq) FROM events").fetchone()
            max_seq = int(max_seq_row[0] or 0)
            cutoff = max_seq - int(keep_last)
            if cutoff <= 0:
                return {"archived": 0, "path": None, "kept": max_seq}
            path = directory / f"events_{int(utc_now())}_{cutoff}.jsonl.gz"
            archived = 0
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                cursor = conn.execute(
                    "SELECT seq, payload FROM events WHERE seq <= ? ORDER BY seq ASC",
                    (cutoff,),
                )
                for row in cursor:
                    handle.write(row["payload"] + "\n")
                    archived += 1
            conn.execute("DELETE FROM events WHERE seq <= ?", (cutoff,))
            remaining = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        return {
            "archived": archived,
            "path": str(path),
            "kept": remaining,
            "bytes": path.stat().st_size,
        }

    def archive_files(self, archive_dir: str | Path) -> list[Path]:
        directory = Path(archive_dir)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("events_*.jsonl.gz"))

    # -------------------------------------------------------------- episodes
    def start_episode(
        self, *, episode_index: int, world_id: str, seed: int, config_hash: str, notes: str = ""
    ) -> str:
        episode_id = new_id("ep_")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO episodes(episode_id, episode_index, started_at, world_id, seed, "
                "config_hash, notes) VALUES(?,?,?,?,?,?,?)",
                (episode_id, episode_index, utc_now(), world_id, seed, config_hash, notes),
            )
        return episode_id

    def end_episode(self, episode_id: str, notes: str | None = None) -> None:
        with self.connect() as conn:
            if notes is None:
                conn.execute(
                    "UPDATE episodes SET ended_at=? WHERE episode_id=?", (utc_now(), episode_id)
                )
            else:
                conn.execute(
                    "UPDATE episodes SET ended_at=?, notes=? WHERE episode_id=?",
                    (utc_now(), notes, episode_id),
                )

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------- events
    def append_event(self, event: Event) -> int:
        event.assert_persistable()
        payload = json.dumps(event.to_dict(), ensure_ascii=False)
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events(event_id, timestamp, step, episode_id, actor_id, event_type, "
                "modality, source, source_device, source_reliability, prediction_error, "
                "confidence, privacy_class, provenance, schema_version, payload) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event.event_id,
                    event.timestamp,
                    event.step,
                    event.episode_id,
                    event.actor_id,
                    event.event_type,
                    event.modality,
                    event.source,
                    event.source_device,
                    event.source_reliability,
                    event.prediction_error,
                    event.confidence,
                    event.privacy_class,
                    event.provenance,
                    event.schema_version,
                    payload,
                ),
            )
            return int(cursor.lastrowid)

    def append_events(self, events: Iterable[Event]) -> int:
        batch = list(events)
        if not batch:
            return 0
        rows = []
        for event in batch:
            event.assert_persistable()
            rows.append(
                (
                    event.event_id,
                    event.timestamp,
                    event.step,
                    event.episode_id,
                    event.actor_id,
                    event.event_type,
                    event.modality,
                    event.source,
                    event.source_device,
                    event.source_reliability,
                    event.prediction_error,
                    event.confidence,
                    event.privacy_class,
                    event.provenance,
                    event.schema_version,
                    json.dumps(event.to_dict(), ensure_ascii=False),
                )
            )
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO events(event_id, timestamp, step, episode_id, actor_id, event_type, "
                "modality, source, source_device, source_reliability, prediction_error, "
                "confidence, privacy_class, provenance, schema_version, payload) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def append_events_batch(self, events: Iterable[Event]) -> int:
        return self.append_events(events)

    def iter_events(
        self,
        *,
        limit: int | None = None,
        since_step: int | None = None,
        episode_id: str | None = None,
        event_type: str | None = None,
    ) -> Iterator[Event]:
        query = "SELECT payload FROM events"
        clauses: list[str] = []
        params: list[Any] = []
        if since_step is not None:
            clauses.append("step >= ?")
            params.append(since_step)
        if episode_id is not None:
            clauses.append("episode_id = ?")
            params.append(episode_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY seq ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self.connect() as conn:
            for row in conn.execute(query, params):
                yield Event.from_dict(json.loads(row["payload"]))

    def recent_events(self, limit: int = 10) -> list[Event]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM events ORDER BY seq DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [Event.from_dict(json.loads(r["payload"])) for r in reversed(rows)]

    def count_events(self) -> int:
        if not self.exists:
            return 0
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def event_type_counts(self) -> dict[str, int]:
        if not self.exists:
            return {}
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type"
            ).fetchall()
        return {row["event_type"]: int(row["n"]) for row in rows}

    # ----------------------------------------------------------- checkpoints
    def record_checkpoint(
        self,
        *,
        checkpoint_id: str,
        step: int,
        config_hash: str,
        path: str,
        sha256: str,
        size_bytes: int,
        note: str = "",
        schema_version: int = SCHEMA_VERSION,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO checkpoints(checkpoint_id, created_at, step, schema_version, "
                "config_hash, path, sha256, size_bytes, note) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    checkpoint_id,
                    utc_now(),
                    step,
                    schema_version,
                    config_hash,
                    path,
                    sha256,
                    size_bytes,
                    note,
                ),
            )

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_checkpoints(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_checkpoint(self) -> dict[str, Any] | None:
        rows = self.list_checkpoints(limit=1)
        return rows[0] if rows else None

    # --------------------------------------------------------------- health
    def log_health(self, step: int, healths: Iterable[Any]) -> None:
        with self.connect() as conn:
            for health in healths:
                conn.execute(
                    "INSERT INTO module_health(timestamp, step, module, status, detail, metrics) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        utc_now(),
                        step,
                        health.module,
                        health.status,
                        health.detail,
                        json.dumps(health.metrics, ensure_ascii=False),
                    ),
                )

    def latest_health(self) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM module_health WHERE seq IN "
                "(SELECT MAX(seq) FROM module_health GROUP BY module) ORDER BY module"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ runs
    def start_run(
        self, *, version: str, config_hash: str, seed: int, args: dict[str, Any] | None = None
    ) -> str:
        run_id = new_id("run_")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runs(run_id, version, started_at, config_hash, seed, steps, status, args) "
                "VALUES(?,?,?,?,?,0,'running',?)",
                (run_id, version, utc_now(), config_hash, seed, json.dumps(args or {}, ensure_ascii=False)),
            )
        return run_id

    def finish_run(self, run_id: str, *, steps: int, status: str = "completed") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET ended_at=?, steps=?, status=? WHERE run_id=?",
                (utc_now(), steps, status, run_id),
            )

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- export
    def export_jsonl(self, path: str | Path, *, limit: int | None = None) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with target.open("w", encoding="utf-8") as handle:
            for event in self.iter_events(limit=limit):
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    def backup(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            dest = sqlite3.connect(str(target))
            try:
                conn.backup(dest)
            finally:
                dest.close()
        return target

    def integrity_check(self) -> str:
        if not self.exists:
            return "missing"
        with self.connect() as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def tables(self) -> list[str]:
        if not self.exists:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows]

    def stats(self) -> dict[str, Any]:
        if not self.exists:
            return {"exists": False}
        return {
            "exists": True,
            "path": str(self.path),
            "schema_version": self.version(),
            "events": self.count_events(),
            "event_types": self.event_type_counts(),
            "checkpoints": len(self.list_checkpoints(limit=1000)),
            "runs": len(self.list_runs(limit=1000)),
            "size_bytes": self.path.stat().st_size,
            "integrity": self.integrity_check(),
        }

    def quick_stats(self) -> dict[str, Any]:
        """Cheap statistics suitable for a per-N-steps status snapshot."""
        if not self.exists:
            return {"exists": False}
        return {
            "exists": True,
            "schema_version": self.version(),
            "events": self.count_events(),
            "size_bytes": self.path.stat().st_size,
        }
