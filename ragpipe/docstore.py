"""Document store — persists full document text and chunk content.

Qdrant stores only vector + reference payloads (doc_id, chunk_id).
Full text lives here, hydrated at query time before reranking.

Backend is configurable via DOCSTORE_BACKEND env var:
  - "postgres" (default): uses psycopg2
  - "sqlite": local file, suitable for dev/testing

Schema:
  chunks table:
    doc_id    TEXT    — UUID of the source document
    chunk_id  INTEGER — stable integer offset within the document
    text      TEXT    — full chunk content
    source    TEXT    — filename or URI for observability
    created_at TEXT   — ISO8601 timestamp
    PRIMARY KEY (doc_id, chunk_id)
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime

log = logging.getLogger("docstore")

DOCSTORE_BACKEND = os.environ.get("DOCSTORE_BACKEND", "postgres")
DOCSTORE_URL = os.environ.get("DOCSTORE_URL", "")
DOCSTORE_SQLITE_PATH = os.environ.get("DOCSTORE_SQLITE_PATH", "/tmp/docstore.db")


class DocstoreBackend(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        """Create tables if they don't exist."""

    @abstractmethod
    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str) -> None:
        """Insert or update a single chunk. Upsert on (doc_id, chunk_id)."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Batch upsert. Each dict has: doc_id, chunk_id, text, source."""

    @abstractmethod
    def get_chunk(self, doc_id: str, chunk_id: int) -> str | None:
        """Return chunk text or None if not found."""

    @abstractmethod
    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
        """Batch get. refs is list of (doc_id, chunk_id). Returns {(doc_id, chunk_id): text}."""

    @abstractmethod
    def delete_doc(self, doc_id: str) -> None:
        """Delete all chunks for a document."""


class PostgresDocstore(DocstoreBackend):
    def __init__(self, url: str):
        import psycopg2

        self._url = url
        self._conn = psycopg2.connect(url)
        self._conn.autocommit = True

    def init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id     TEXT NOT NULL,
                    chunk_id   INTEGER NOT NULL,
                    text       TEXT NOT NULL,
                    source     TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (doc_id, chunk_id)
                )
            """)

    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, chunk_id, text, source, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, chunk_id)
                DO UPDATE SET text = EXCLUDED.text, source = EXCLUDED.source
            """,
                (doc_id, chunk_id, text, source, now),
            )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn.cursor() as cur:
            from psycopg2.extras import execute_values

            values = [(c["doc_id"], c["chunk_id"], c["text"], c["source"], now) for c in chunks]
            execute_values(
                cur,
                """
                INSERT INTO chunks (doc_id, chunk_id, text, source, created_at)
                VALUES %s
                ON CONFLICT (doc_id, chunk_id)
                DO UPDATE SET text = EXCLUDED.text, source = EXCLUDED.source
            """,
                values,
            )

    def get_chunk(self, doc_id: str, chunk_id: int) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT text FROM chunks WHERE doc_id = %s AND chunk_id = %s", (doc_id, chunk_id))
            row = cur.fetchone()
            return row[0] if row else None

    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
        if not refs:
            return {}
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, text FROM chunks
                WHERE (doc_id, chunk_id) IN (
                    SELECT unnest(%s::text[]), unnest(%s::integer[])
                )
            """,
                ([r[0] for r in refs], [r[1] for r in refs]),
            )
            return {(row[0], row[1]): row[2] for row in cur.fetchall()}

    def delete_doc(self, doc_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))


class SQLiteDocstore(DocstoreBackend):
    def __init__(self, path: str):
        import sqlite3

        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                doc_id     TEXT NOT NULL,
                chunk_id   INTEGER NOT NULL,
                text       TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (doc_id, chunk_id)
            )
        """)
        self._conn.commit()

    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO chunks (doc_id, chunk_id, text, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (doc_id, chunk_id)
            DO UPDATE SET text = excluded.text, source = excluded.source
        """,
            (doc_id, chunk_id, text, source, now),
        )
        self._conn.commit()

    def upsert_chunks(self, chunks: list[dict]) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.executemany(
            """
            INSERT INTO chunks (doc_id, chunk_id, text, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (doc_id, chunk_id)
            DO UPDATE SET text = excluded.text, source = excluded.source
        """,
            [(c["doc_id"], c["chunk_id"], c["text"], c["source"], now) for c in chunks],
        )
        self._conn.commit()

    def get_chunk(self, doc_id: str, chunk_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM chunks WHERE doc_id = ? AND chunk_id = ?", (doc_id, chunk_id)
        ).fetchone()
        return row[0] if row else None

    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
        if not refs:
            return {}
        placeholders = ",".join(["(?, ?)"] * len(refs))
        params = [v for r in refs for v in r]
        rows = self._conn.execute(
            f"SELECT doc_id, chunk_id, text FROM chunks WHERE (doc_id, chunk_id) IN ({placeholders})",
            params,
        ).fetchall()
        return {(row[0], row[1]): row[2] for row in rows}

    def delete_doc(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()


def create_docstore(backend: str | None = None) -> DocstoreBackend:
    """Factory: create a docstore backend based on config."""
    backend = backend or DOCSTORE_BACKEND
    if backend == "postgres":
        if not DOCSTORE_URL:
            raise ValueError("DOCSTORE_URL must be set for postgres backend")
        store = PostgresDocstore(DOCSTORE_URL)
    elif backend == "sqlite":
        store = SQLiteDocstore(DOCSTORE_SQLITE_PATH)
    else:
        raise ValueError(f"Unknown DOCSTORE_BACKEND: {backend}")
    store.init_schema()
    log.info("Docstore initialized: %s", backend)
    return store
