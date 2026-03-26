"""Persistent conversation history with full-text search.

Stores all conversations from CLI and API in ~/.ppmlx/history.db using SQLite
with FTS5 for full-text search.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any


def _get_history_path() -> Path:
    try:
        from ppmlx.config import get_ppmlx_dir
        return get_ppmlx_dir() / "history.db"
    except ImportError:
        return Path.home() / ".ppmlx" / "history.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversations_model ON conversations(model);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


ROLE_LABELS: dict[str, str] = {"user": "You", "assistant": "Assistant", "system": "System"}


def _auto_title(content: str, max_len: int = 72) -> str:
    """Generate a short title from the first user message."""
    text = content.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len - 3].rsplit(" ", 1)[0] + "..."


class HistoryManager:
    """Manage persistent conversation history backed by SQLite + FTS5."""

    def __init__(self, path: Path | None = None):
        self._path = path or _get_history_path()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.executescript(_FTS_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Create / Update ──────────────────────────────────────────────

    def create_conversation(self, model: str, title: str = "") -> str:
        """Create a new conversation and return its ID."""
        conn = self._connect()
        conv_id = uuid.uuid4().hex[:12]
        conn.execute(
            "INSERT INTO conversations (id, model, title) VALUES (?, ?, ?)",
            (conv_id, model, title),
        )
        conn.commit()
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> int:
        """Append a message to a conversation. Returns the message rowid."""
        conn = self._connect()
        # If content is structured (list for vision), serialize to JSON
        if not isinstance(content, str):
            content = json.dumps(content)
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
            (conversation_id,),
        )
        # Auto-set title from first user message if empty
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row and not row["title"] and role == "user":
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (_auto_title(content), conversation_id),
            )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ── Read ─────────────────────────────────────────────────────────

    def list_conversations(
        self,
        limit: int = 20,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent conversations, newest first."""
        conn = self._connect()
        conditions: list[str] = []
        params: list[Any] = []
        if model:
            conditions.append("c.model = ?")
            params.append(model)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        rows = conn.execute(
            f"""SELECT c.id, c.model, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                {where}
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get a single conversation's metadata."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Get all messages in a conversation, ordered chronologically."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across all messages. Returns conversations with matching snippets."""
        conn = self._connect()
        # First get matching message IDs and snippets, then deduplicate by conversation.
        rows = conn.execute(
            """SELECT m.conversation_id,
                      snippet(messages_fts, 0, '>>>', '<<<', '...', 48) as snippet
               FROM messages_fts
               JOIN messages m ON m.id = messages_fts.rowid
               WHERE messages_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit * 5),  # fetch extra to allow dedup
        ).fetchall()

        # Deduplicate by conversation, keeping best match
        seen: dict[str, str] = {}
        for r in rows:
            cid = r["conversation_id"]
            if cid not in seen:
                seen[cid] = r["snippet"]

        if not seen:
            return []

        # Fetch conversation metadata for matched IDs
        placeholders = ",".join("?" * len(seen))
        conv_rows = conn.execute(
            f"""SELECT id, model, title, created_at, updated_at
                FROM conversations
                WHERE id IN ({placeholders})
                ORDER BY updated_at DESC
                LIMIT ?""",
            list(seen.keys()) + [limit],
        ).fetchall()

        results = []
        for cr in conv_rows:
            d = dict(cr)
            d["snippet"] = seen.get(d["id"], "")
            results.append(d)
        return results

    # ── Delete ───────────────────────────────────────────────────────

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages. Returns True if found."""
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        conn.commit()
        return cur.rowcount > 0

    # ── Export ────────────────────────────────────────────────────────

    def export_conversation(
        self, conversation_id: str, fmt: str = "json"
    ) -> str | None:
        """Export a conversation as JSON or Markdown. Returns None if not found."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return None
        messages = self.get_messages(conversation_id)

        if fmt == "json":
            return json.dumps(
                {
                    "id": conv["id"],
                    "model": conv["model"],
                    "title": conv["title"],
                    "created_at": conv["created_at"],
                    "updated_at": conv["updated_at"],
                    "messages": [
                        {"role": m["role"], "content": m["content"], "timestamp": m["timestamp"]}
                        for m in messages
                    ],
                },
                indent=2,
            )
        elif fmt == "md":
            lines = [
                f"# {conv['title'] or 'Untitled Conversation'}",
                "",
                f"**Model:** {conv['model']}  ",
                f"**Date:** {conv['created_at']}",
                "",
                "---",
                "",
            ]
            for m in messages:
                role_label = ROLE_LABELS.get(m["role"], m["role"].capitalize())
                lines.append(f"### {role_label}")
                lines.append("")
                lines.append(m["content"])
                lines.append("")
            return "\n".join(lines)
        else:
            return None

    # ── Resolve ──────────────────────────────────────────────────────

    def resolve_id(self, prefix: str) -> str | None:
        """Resolve a conversation ID prefix to a full ID. Returns None if ambiguous or not found."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id FROM conversations WHERE id LIKE ? ORDER BY updated_at DESC",
            (prefix + "%",),
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"]
        return None


# ── Singleton ────────────────────────────────────────────────────────

_instance: HistoryManager | None = None


def get_history(path: Path | None = None) -> HistoryManager:
    """Return the singleton HistoryManager."""
    global _instance
    if _instance is None:
        _instance = HistoryManager(path)
    return _instance


def reset_history() -> None:
    """Reset singleton (for testing)."""
    global _instance
    if _instance is not None:
        _instance.close()
    _instance = None
