"""Schema migrations.

Version 1 (V0.1): meta, episodes, events, checkpoints, module_health, runs.
Version 2 (V0.3): layered memories, consolidation log, autobiographical summaries.

Adding a version means appending to :data:`MIGRATIONS`; already-created databases
are upgraded in one transaction per version.
"""

from __future__ import annotations

from wphr.version import SCHEMA_VERSION

MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE episodes (
            episode_id  TEXT PRIMARY KEY,
            episode_index INTEGER NOT NULL,
            started_at  REAL NOT NULL,
            ended_at    REAL,
            world_id    TEXT,
            seed        INTEGER,
            config_hash TEXT NOT NULL,
            notes       TEXT
        )
        """,
        """
        CREATE TABLE events (
            seq               INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id          TEXT NOT NULL UNIQUE,
            timestamp         REAL NOT NULL,
            step              INTEGER NOT NULL,
            episode_id        TEXT NOT NULL,
            actor_id          TEXT NOT NULL,
            event_type        TEXT NOT NULL,
            modality          TEXT NOT NULL,
            source            TEXT NOT NULL,
            source_device     TEXT NOT NULL,
            source_reliability REAL NOT NULL,
            prediction_error  REAL,
            confidence        REAL NOT NULL,
            privacy_class     TEXT NOT NULL,
            provenance        TEXT NOT NULL,
            schema_version    INTEGER NOT NULL,
            payload           TEXT NOT NULL
        )
        """,
        "CREATE INDEX idx_events_step ON events(step)",
        "CREATE INDEX idx_events_episode ON events(episode_id)",
        "CREATE INDEX idx_events_type ON events(event_type)",
        """
        CREATE TABLE checkpoints (
            checkpoint_id  TEXT PRIMARY KEY,
            created_at     REAL NOT NULL,
            step           INTEGER NOT NULL,
            schema_version INTEGER NOT NULL,
            config_hash    TEXT NOT NULL,
            path           TEXT NOT NULL,
            sha256         TEXT NOT NULL,
            size_bytes     INTEGER NOT NULL,
            note           TEXT
        )
        """,
        """
        CREATE TABLE module_health (
            seq       INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            step      INTEGER NOT NULL,
            module    TEXT NOT NULL,
            status    TEXT NOT NULL,
            detail    TEXT,
            metrics   TEXT
        )
        """,
        """
        CREATE TABLE runs (
            run_id      TEXT PRIMARY KEY,
            version     TEXT NOT NULL,
            started_at  REAL NOT NULL,
            ended_at    REAL,
            config_hash TEXT NOT NULL,
            seed        INTEGER NOT NULL,
            steps       INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL,
            args        TEXT
        )
        """,
        "INSERT INTO meta(key, value) VALUES ('created_by', 'wphr.init')",
    ],
    2: [
        """
        CREATE TABLE memories (
            memory_id     TEXT PRIMARY KEY,
            layer         TEXT NOT NULL,
            kind          TEXT NOT NULL,
            key           TEXT,
            content       TEXT NOT NULL,
            source        TEXT NOT NULL,
            confidence    REAL NOT NULL,
            importance    REAL NOT NULL,
            support_count INTEGER NOT NULL DEFAULT 0,
            counter_count INTEGER NOT NULL DEFAULT 0,
            scope         TEXT,
            created_step  INTEGER NOT NULL,
            updated_step  INTEGER NOT NULL,
            created_at    REAL NOT NULL,
            updated_at    REAL NOT NULL,
            last_access   REAL,
            access_count  INTEGER NOT NULL DEFAULT 0,
            active        INTEGER NOT NULL DEFAULT 1
        )
        """,
        "CREATE INDEX idx_memories_layer ON memories(layer)",
        "CREATE INDEX idx_memories_key ON memories(key)",
        "CREATE INDEX idx_memories_active ON memories(active)",
        """
        CREATE TABLE memory_links (
            seq         INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id     TEXT NOT NULL,
            to_id       TEXT NOT NULL,
            relation    TEXT NOT NULL,
            weight      REAL NOT NULL DEFAULT 1.0,
            created_at  REAL NOT NULL
        )
        """,
        """
        CREATE TABLE summaries (
            summary_id  TEXT PRIMARY KEY,
            layer       TEXT NOT NULL,
            start_step  INTEGER NOT NULL,
            end_step    INTEGER NOT NULL,
            text        TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  REAL NOT NULL
        )
        """,
        """
        CREATE TABLE consolidation_log (
            seq        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  REAL NOT NULL,
            step       INTEGER NOT NULL,
            layer      TEXT NOT NULL,
            written    INTEGER NOT NULL,
            note       TEXT
        )
        """,
    ],
}

EXPECTED_TABLES = (
    "meta",
    "episodes",
    "events",
    "checkpoints",
    "module_health",
    "runs",
)

LAYER_TABLES = ("memories", "memory_links", "summaries", "consolidation_log")


def current_version() -> int:
    return SCHEMA_VERSION


def table_rows(target_version: int = SCHEMA_VERSION) -> list[str]:
    """All statements needed to build a database at ``target_version``."""
    statements: list[str] = []
    for version in range(1, target_version + 1):
        statements.extend(MIGRATIONS[version])
    return statements
