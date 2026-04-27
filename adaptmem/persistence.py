"""SQLite-backed corpus store — survives daemon restarts.

Off by default (the daemon's `corpora` dict is in-memory only). Pass
`--persist-dir PATH` or set `ADAPTMEM_PERSIST_DIR` to enable: every
`/reindex` writes both to memory and to disk; on next daemon start
all stored corpora are reloaded.

Schema (SQLite, single file `corpora.db`):

    CREATE TABLE corpora (
        id          TEXT PRIMARY KEY,
        model       TEXT NOT NULL,
        n_docs      INTEGER NOT NULL,
        dim         INTEGER NOT NULL,
        created_at  REAL NOT NULL
    );
    CREATE TABLE documents (
        corpus_id   TEXT NOT NULL,
        doc_id      TEXT NOT NULL,
        text        TEXT NOT NULL,
        embedding   BLOB NOT NULL,  -- float32[] tobytes
        PRIMARY KEY (corpus_id, doc_id),
        FOREIGN KEY (corpus_id) REFERENCES corpora(id) ON DELETE CASCADE
    );
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    id          TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    n_docs      INTEGER NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    corpus_id   TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    PRIMARY KEY (corpus_id, doc_id),
    FOREIGN KEY (corpus_id) REFERENCES corpora(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_corpus ON documents(corpus_id);
"""


class CorpusStore:
    """SQLite-backed corpus persistence. One process per file, locking
    handled by SQLite WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def save_corpus(
        self,
        corpus_id: str,
        model: str,
        documents: Iterable[tuple[str, str]],  # (doc_id, text)
        embeddings: np.ndarray[Any, Any],
    ) -> None:
        """Replace the entire corpus (delete existing rows + reinsert).

        Atomic — wrapped in a transaction. If `embeddings.dtype` isn't
        float32, it's cast first (callers usually already provide
        float32 from sentence-transformers).
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        docs = list(documents)
        if len(docs) != embeddings.shape[0]:
            raise ValueError(
                f"documents ({len(docs)}) and embeddings ({embeddings.shape[0]}) "
                "row counts must match"
            )

        dim = int(embeddings.shape[1])
        with self._conn:
            # Delete in case the corpus already exists.
            self._conn.execute("DELETE FROM corpora WHERE id = ?", (corpus_id,))
            self._conn.execute("DELETE FROM documents WHERE corpus_id = ?", (corpus_id,))
            self._conn.execute(
                "INSERT INTO corpora (id, model, n_docs, dim, created_at) VALUES (?, ?, ?, ?, ?)",
                (corpus_id, model, len(docs), dim, time.time()),
            )
            rows = [
                (corpus_id, doc_id, text, embeddings[i].tobytes())
                for i, (doc_id, text) in enumerate(docs)
            ]
            self._conn.executemany(
                "INSERT INTO documents (corpus_id, doc_id, text, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )

    def load_corpus(
        self, corpus_id: str
    ) -> tuple[list[tuple[str, str]], np.ndarray[Any, Any], str] | None:
        """Return (documents, embeddings, model_name) or None if missing."""
        meta = self._conn.execute(
            "SELECT model, dim FROM corpora WHERE id = ?", (corpus_id,)
        ).fetchone()
        if meta is None:
            return None
        model, dim = meta
        rows = self._conn.execute(
            "SELECT doc_id, text, embedding FROM documents WHERE corpus_id = ? ORDER BY doc_id",
            (corpus_id,),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, dim), dtype=np.float32), model
        documents = [(r[0], r[1]) for r in rows]
        embeddings = np.stack(
            [np.frombuffer(r[2], dtype=np.float32) for r in rows], axis=0
        )
        return documents, embeddings, model

    def list_corpora(self) -> list[str]:
        return [
            row[0]
            for row in self._conn.execute("SELECT id FROM corpora ORDER BY id").fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
