"""Persistent platform store: sessions, messages, knowledge and learning events."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class PlatformStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._schema()

    def _schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
          created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
          role TEXT NOT NULL, text TEXT NOT NULL, source TEXT NOT NULL,
          state_json TEXT, citations_json TEXT, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge (
          id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL,
          title TEXT, url TEXT, license TEXT, summary TEXT,
          snippet TEXT, trust TEXT, retrieved_at REAL, content_hash TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS learning (
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
          kind TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS profiles (
          user_id TEXT PRIMARY KEY, name TEXT, notes TEXT, updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
        """)
        self.conn.commit()

    def ensure_session(self, session_id: str | None, user_id: str) -> str:
        sid = session_id or "s-" + uuid.uuid4().hex[:12]
        now = time.time()
        row = self.conn.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            title = "新的对话"
            self.conn.execute(
                "INSERT INTO sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (sid, user_id, title, now, now))
            self.conn.commit()
        else:
            self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
            self.conn.commit()
        return sid

    def add_message(self, session_id: str, role: str, text: str, source: str,
                    state: dict | None = None, citations: list | None = None):
        now = time.time()
        self.conn.execute(
            "INSERT INTO messages(session_id,role,text,source,state_json,citations_json,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (session_id, role, text, source,
             json.dumps(state or {}, ensure_ascii=False),
             json.dumps(citations or [], ensure_ascii=False), now))
        self.conn.commit()
        # Set a readable title from the first user message.
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id=?", (session_id,)).fetchone()
        if row["n"] == 1 and role == "user":
            title = text.strip().replace("\n", " ")[:40] or "新的对话"
            self.conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
            self.conn.commit()

    def history(self, session_id: str, limit: int = 200):
        rows = self.conn.execute(
            "SELECT role,text,source,state_json,citations_json,created_at"
            " FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ?",
            (session_id, int(limit))).fetchall()
        out = []
        for r in rows:
            out.append({"role": r["role"], "text": r["text"], "source": r["source"],
                        "state": json.loads(r["state_json"] or "{}"),
                        "citations": json.loads(r["citations_json"] or "[]"),
                        "created_at": r["created_at"]})
        return out

    def message_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
        return int(row["n"] if row else 0)

    def recent_context(self, session_id: str, limit: int = 8) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role,text FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, int(limit))).fetchall()
        return [{"role": r["role"], "text": r["text"]} for r in reversed(rows)]

    def sessions(self, user_id: str, limit: int = 30):
        rows = self.conn.execute(
            "SELECT id,title,created_at,updated_at FROM sessions WHERE user_id=?"
            " ORDER BY updated_at DESC LIMIT ?", (user_id, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def add_knowledge(self, *, query, title, url, license, summary, snippet,
                      trust, content_hash):
        self.conn.execute(
            "INSERT OR IGNORE INTO knowledge(query,title,url,license,summary,snippet,"
            "trust,retrieved_at,content_hash) VALUES(?,?,?,?,?,?,?,?,?)",
            (query, title, url, license, summary, snippet, trust, time.time(), content_hash))
        self.conn.commit()

    def knowledge(self, limit: int = 50):
        rows = self.conn.execute(
            "SELECT query,title,url,license,summary,snippet,trust,retrieved_at"
            " FROM knowledge ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def add_learning(self, session_id: str | None, kind: str, content: dict):
        self.conn.execute(
            "INSERT INTO learning(session_id,kind,content,created_at) VALUES(?,?,?,?)",
            (session_id, kind, json.dumps(content, ensure_ascii=False), time.time()))
        self.conn.commit()

    def learning(self, limit: int = 100):
        rows = self.conn.execute(
            "SELECT session_id,kind,content,created_at FROM learning ORDER BY id DESC LIMIT ?",
            (int(limit),)).fetchall()
        return [{"session_id": r["session_id"], "kind": r["kind"],
                 "content": json.loads(r["content"]), "created_at": r["created_at"]}
                for r in rows]

    def get_profile(self, user_id: str):
        row = self.conn.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else {"user_id": user_id, "name": None, "notes": None}

    def set_profile(self, user_id: str, name: str | None = None, notes: str | None = None):
        cur = self.get_profile(user_id)
        self.conn.execute(
            "INSERT OR REPLACE INTO profiles(user_id,name,notes,updated_at) VALUES(?,?,?,?)",
            (user_id, name if name is not None else cur.get("name"),
             notes if notes is not None else cur.get("notes"), time.time()))
        self.conn.commit()

    def close(self):
        self.conn.close()
