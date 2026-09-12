"""Persistence.

SQLite via the stdlib. The data model is small and mostly document-shaped
(analysis and preparation are naturally JSON), so a relational schema would add
migrations without buying anything -- except in one place: the *feedback* table,
which is deliberately relational and append-only, because it is a training set
and wants to be queried by cue kind, by reason code, by time delta.

That table is the human-in-the-loop story made concrete. Every accept, reject and
drag lands in it with the evidence that produced the original recommendation, so
`scripts/refit_weights.py` can fit the scoring model on real decisions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    duration_sec REAL,
    created_at REAL NOT NULL,
    title TEXT, artist TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    track_id TEXT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS preparations (
    track_id TEXT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    payload TEXT NOT NULL,
    config TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    marker_id TEXT NOT NULL,
    marker_class TEXT NOT NULL,     -- 'cue' | 'loop'
    kind TEXT NOT NULL,             -- drop, breakdown, ...
    action TEXT NOT NULL,           -- accepted | rejected | moved | deleted | recoloured
    original_time REAL,
    final_time REAL,
    delta_bars REAL,
    confidence REAL,
    evidence TEXT,                  -- JSON: the features behind the recommendation
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_kind ON feedback(kind, action);
CREATE INDEX IF NOT EXISTS idx_feedback_track ON feedback(track_id);
"""


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    # --- tracks ------------------------------------------------------------
    def create_track(self, filename: str, path: str, duration: float | None = None,
                     title: str | None = None, artist: str | None = None) -> str:
        tid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO tracks (id, filename, path, duration_sec, created_at,"
                " title, artist) VALUES (?,?,?,?,?,?,?)",
                (tid, filename, str(path), duration, time.time(), title, artist))
        return tid

    def get_track(self, track_id: str) -> dict | None:
        r = self._conn().execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        return dict(r) if r else None

    def list_tracks(self, limit: int = 100) -> list[dict]:
        rs = self._conn().execute(
            "SELECT * FROM tracks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rs]

    def delete_track(self, track_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM tracks WHERE id=?", (track_id,))

    # --- analyses / preparations -------------------------------------------
    def save_analysis(self, track_id: str, version: str, payload: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO analyses (track_id, version, payload,"
                      " created_at) VALUES (?,?,?,?)",
                      (track_id, version, json.dumps(payload), time.time()))

    def get_analysis(self, track_id: str) -> dict | None:
        r = self._conn().execute("SELECT payload FROM analyses WHERE track_id=?",
                                 (track_id,)).fetchone()
        return json.loads(r["payload"]) if r else None

    def save_preparation(self, track_id: str, payload: dict,
                         config: dict | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO preparations (track_id, payload, config,"
                      " updated_at) VALUES (?,?,?,?)",
                      (track_id, json.dumps(payload),
                       json.dumps(config) if config else None, time.time()))

    def get_preparation(self, track_id: str) -> dict | None:
        r = self._conn().execute("SELECT payload FROM preparations WHERE track_id=?",
                                 (track_id,)).fetchone()
        return json.loads(r["payload"]) if r else None

    def get_config(self, track_id: str) -> dict | None:
        r = self._conn().execute("SELECT config FROM preparations WHERE track_id=?",
                                 (track_id,)).fetchone()
        return json.loads(r["config"]) if r and r["config"] else None

    # --- feedback ----------------------------------------------------------
    def log_feedback(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = time.time()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO feedback (track_id, marker_id, marker_class, kind, action,"
                " original_time, final_time, delta_bars, confidence, evidence, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(r.get("track_id"), r.get("marker_id"), r.get("marker_class", "cue"),
                  r.get("kind", "custom"), r.get("action", "accepted"),
                  r.get("original_time"), r.get("final_time"), r.get("delta_bars"),
                  r.get("confidence"),
                  json.dumps(r.get("evidence")) if r.get("evidence") else None, now)
                 for r in rows])
        return len(rows)

    def feedback_stats(self) -> list[dict]:
        rs = self._conn().execute(
            "SELECT kind, action, COUNT(*) n, AVG(confidence) avg_conf,"
            " AVG(ABS(delta_bars)) avg_abs_delta_bars"
            " FROM feedback GROUP BY kind, action ORDER BY kind, action").fetchall()
        return [dict(r) for r in rs]

    def export_feedback(self) -> list[dict]:
        rs = self._conn().execute("SELECT * FROM feedback ORDER BY id").fetchall()
        out = []
        for r in rs:
            d = dict(r)
            if d.get("evidence"):
                d["evidence"] = json.loads(d["evidence"])
            out.append(d)
        return out
