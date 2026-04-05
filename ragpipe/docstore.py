"""Document store — persists full document text and chunk content.

Qdrant stores only vector + reference payloads (doc_id, chunk_id).
Full text lives here, hydrated at query time before reranking.

Backend is configurable via DOCSTORE_BACKEND env var:
  - "postgres" (default): uses asyncpg with connection pooling
  - "sqlite": local file, suitable for dev/testing (synchronous)

Schema:
  chunks table:
    doc_id    TEXT    — UUID of the source document
    chunk_id  INTEGER — stable integer offset within the document
    text      TEXT    — full chunk content
    source    TEXT    — filename or URI for observability
    title     TEXT    — document title (denormalised, same for all chunks of a doc)
    created_at TEXT   — ISO8601 timestamp
    PRIMARY KEY (doc_id, chunk_id)
"""

import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import UTC, datetime

log = logging.getLogger("docstore")

DOCSTORE_BACKEND = os.environ.get("DOCSTORE_BACKEND", "postgres")
DOCSTORE_URL = os.environ.get("DOCSTORE_URL", "")
DOCSTORE_SQLITE_PATH = os.environ.get("DOCSTORE_SQLITE_PATH", "/tmp/docstore.db")
CHUNK_CACHE_SIZE = int(os.environ.get("CHUNK_CACHE_SIZE", "2048"))


def _chunk_dict(text: str, title: str = "", source: str = "") -> dict:
    """Normalise a chunk row into the standard dict shape."""
    return {"text": text, "title": title, "source": source}


class DocstoreBackend(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        """Create tables if they don't exist."""

    @abstractmethod
    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str, title: str = "") -> None:
        """Insert or update a single chunk. Upsert on (doc_id, chunk_id)."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Batch upsert. Each dict has: doc_id, chunk_id, text, source, title."""

    @abstractmethod
    def get_chunk(self, doc_id: str, chunk_id: int) -> dict | None:
        """Return chunk dict {text, title, source} or None if not found."""

    @abstractmethod
    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        """Batch get. refs is list of (doc_id, chunk_id). Returns {(doc_id, chunk_id): {text, title, source}}."""

    @abstractmethod
    def delete_doc(self, doc_id: str) -> None:
        """Delete all chunks for a document."""


class PostgresDocstore(DocstoreBackend):
    """Async Postgres backend using asyncpg with connection pooling.

    Exposes sync methods (init_schema, upsert_*) for ingestion compatibility
    and async methods (get_chunks_async) for the hot query path.
    """

    def __init__(self, url: str):
        self._url = url
        self._pool = None
        # Sync connection for init_schema and upsert (ingestion path)
        self._sync_conn = None
        self._schema_ready = False

    def _get_sync_conn(self):
        if self._sync_conn is None:
            import psycopg2

            self._sync_conn = psycopg2.connect(self._url)
            self._sync_conn.autocommit = True
        return self._sync_conn

    def _ensure_schema(self) -> None:
        """Create tables if not already confirmed to exist."""
        if self._schema_ready:
            return
        conn = self._get_sync_conn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id     TEXT NOT NULL,
                    chunk_id   INTEGER NOT NULL,
                    text       TEXT NOT NULL,
                    source     TEXT NOT NULL DEFAULT '',
                    title      TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (doc_id, chunk_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    collection_id TEXT,
                    query_text TEXT,
                    query_hash TEXT,
                    grounding TEXT,
                    cited_chunks TEXT[],
                    total_chunks INT,
                    latency_ms INT,
                    model TEXT,
                    route TEXT,
                    created_at TIMESTAMP DEFAULT now()
                )
            """)
        self._schema_ready = True

    def init_schema(self) -> None:
        self._ensure_schema()

    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str, title: str = "") -> None:
        self._ensure_schema()
        now = datetime.now(UTC).isoformat()
        conn = self._get_sync_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, chunk_id, text, source, title, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id, chunk_id)
                DO UPDATE SET text = EXCLUDED.text, source = EXCLUDED.source, title = EXCLUDED.title
            """,
                (doc_id, chunk_id, text, source, title, now),
            )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        self._ensure_schema()
        now = datetime.now(UTC).isoformat()
        conn = self._get_sync_conn()
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values

            values = [(c["doc_id"], c["chunk_id"], c["text"], c["source"], c.get("title", ""), now) for c in chunks]
            execute_values(
                cur,
                """
                INSERT INTO chunks (doc_id, chunk_id, text, source, title, created_at)
                VALUES %s
                ON CONFLICT (doc_id, chunk_id)
                DO UPDATE SET text = EXCLUDED.text, source = EXCLUDED.source, title = EXCLUDED.title
            """,
                values,
            )

    def get_chunk(self, doc_id: str, chunk_id: int) -> dict | None:
        conn = self._get_sync_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT text, title, source FROM chunks WHERE doc_id = %s AND chunk_id = %s",
                (doc_id, chunk_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _chunk_dict(row[0], row[1], row[2])

    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        """Sync batch get — used by ingestion and tests."""
        if not refs:
            return {}
        conn = self._get_sync_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, text, title, source FROM chunks
                WHERE (doc_id, chunk_id) IN (
                    SELECT unnest(%s::text[]), unnest(%s::integer[])
                )
            """,
                ([r[0] for r in refs], [r[1] for r in refs]),
            )
            return {(row[0], row[1]): _chunk_dict(row[2], row[3], row[4]) for row in cur.fetchall()}

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self._url,
                min_size=2,
                max_size=8,
            )
        if not self._schema_ready:
            self._ensure_schema()

    async def get_chunks_async(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        """Async batch get — used by the query hot path."""
        if not refs:
            return {}
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT doc_id, chunk_id, text, title, source FROM chunks
                WHERE (doc_id, chunk_id) IN (
                    SELECT unnest($1::text[]), unnest($2::integer[])
                )
                """,
                [r[0] for r in refs],
                [r[1] for r in refs],
            )
            return {
                (row["doc_id"], row["chunk_id"]): _chunk_dict(row["text"], row["title"], row["source"]) for row in rows
            }

    def delete_doc(self, doc_id: str) -> None:
        conn = self._get_sync_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    def close(self) -> None:
        if self._sync_conn is not None:
            self._sync_conn.close()
            self._sync_conn = None
        if self._pool is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                self._close_task = loop.create_task(self._pool.close())
            except RuntimeError:
                # No running loop — close synchronously via new loop
                asyncio.run(self._pool.close())
            self._pool = None


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
                title      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (doc_id, chunk_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                collection_id TEXT,
                query_text TEXT,
                query_hash TEXT,
                grounding TEXT,
                cited_chunks TEXT,
                total_chunks INTEGER,
                latency_ms INTEGER,
                model TEXT,
                route TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str, title: str = "") -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT INTO chunks (doc_id, chunk_id, text, source, title, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id, chunk_id)
            DO UPDATE SET text = excluded.text, source = excluded.source, title = excluded.title
        """,
            (doc_id, chunk_id, text, source, title, now),
        )
        self._conn.commit()

    def upsert_chunks(self, chunks: list[dict]) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.executemany(
            """
            INSERT INTO chunks (doc_id, chunk_id, text, source, title, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id, chunk_id)
            DO UPDATE SET text = excluded.text, source = excluded.source, title = excluded.title
        """,
            [(c["doc_id"], c["chunk_id"], c["text"], c["source"], c.get("title", ""), now) for c in chunks],
        )
        self._conn.commit()

    def get_chunk(self, doc_id: str, chunk_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT text, title, source FROM chunks WHERE doc_id = ? AND chunk_id = ?", (doc_id, chunk_id)
        ).fetchone()
        if row is None:
            return None
        return _chunk_dict(row[0], row[1], row[2])

    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        if not refs:
            return {}
        placeholders = ",".join(["(?, ?)"] * len(refs))
        params = [v for r in refs for v in r]
        rows = self._conn.execute(
            f"SELECT doc_id, chunk_id, text, title, source FROM chunks WHERE (doc_id, chunk_id) IN ({placeholders})",
            params,
        ).fetchall()
        return {(row[0], row[1]): _chunk_dict(row[2], row[3], row[4]) for row in rows}

    async def get_chunks_async(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        """SQLite has no async driver — delegate to sync."""
        return self.get_chunks(refs)

    def delete_doc(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class CachedDocstore:
    """LRU cache wrapper around a DocstoreBackend.

    Caches individual chunk dicts by (doc_id, chunk_id). Cache entries are
    invalidated on upsert and delete. The cache is bounded by CHUNK_CACHE_SIZE.
    """

    def __init__(self, backend: DocstoreBackend, maxsize: int = CHUNK_CACHE_SIZE):
        self._backend = backend
        self._maxsize = maxsize
        self._cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def init_schema(self) -> None:
        self._backend.init_schema()

    def upsert_chunk(self, doc_id: str, chunk_id: int, text: str, source: str, title: str = "") -> None:
        self._backend.upsert_chunk(doc_id, chunk_id, text, source, title)
        self._cache_put((doc_id, chunk_id), _chunk_dict(text, title, source))

    def upsert_chunks(self, chunks: list[dict]) -> None:
        self._backend.upsert_chunks(chunks)
        for c in chunks:
            self._cache_put((c["doc_id"], c["chunk_id"]), _chunk_dict(c["text"], c.get("title", ""), c["source"]))

    def get_chunk(self, doc_id: str, chunk_id: int) -> dict | None:
        key = (doc_id, chunk_id)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        result = self._backend.get_chunk(doc_id, chunk_id)
        if result is not None:
            self._cache_put(key, result)
        return result

    def get_chunks(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        return self._get_chunks_with_cache(refs, self._backend.get_chunks)

    async def get_chunks_async(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        if hasattr(self._backend, "get_chunks_async"):
            return await self._get_chunks_with_cache_async(refs)
        return self._get_chunks_with_cache(refs, self._backend.get_chunks)

    def delete_doc(self, doc_id: str) -> None:
        self._backend.delete_doc(doc_id)
        # Evict all cached chunks for this doc
        keys_to_remove = [k for k in self._cache if k[0] == doc_id]
        for k in keys_to_remove:
            del self._cache[k]

    def close(self) -> None:
        self._backend.close()
        self._cache.clear()

    @property
    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }

    def _cache_get(self, key: tuple[str, int]) -> dict | None:
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def _cache_put(self, key: tuple[str, int], value: dict) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def _get_chunks_with_cache(self, refs: list[tuple[str, int]], fetch_fn) -> dict[tuple[str, int], dict]:
        if not refs:
            return {}
        result = {}
        missed = []
        for ref in refs:
            cached = self._cache_get(ref)
            if cached is not None:
                result[ref] = cached
            else:
                missed.append(ref)
        if missed:
            fetched = fetch_fn(missed)
            for key, chunk_data in fetched.items():
                self._cache_put(key, chunk_data)
                result[key] = chunk_data
        return result

    async def _get_chunks_with_cache_async(self, refs: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        if not refs:
            return {}
        result = {}
        missed = []
        for ref in refs:
            cached = self._cache_get(ref)
            if cached is not None:
                result[ref] = cached
            else:
                missed.append(ref)
        if missed:
            fetched = await self._backend.get_chunks_async(missed)
            for key, chunk_data in fetched.items():
                self._cache_put(key, chunk_data)
                result[key] = chunk_data
        return result


def create_docstore(backend: str | None = None, *, url: str | None = None) -> CachedDocstore:
    """Factory: create a cached docstore backend based on config.

    Args:
        backend: "postgres" or "sqlite". Defaults to DOCSTORE_BACKEND env var.
        url: Postgres connection string. Defaults to DOCSTORE_URL env var.
    """
    backend = backend or DOCSTORE_BACKEND
    effective_url = url or DOCSTORE_URL
    if backend == "postgres":
        if not effective_url:
            raise ValueError("DOCSTORE_URL must be set for postgres backend")
        store = PostgresDocstore(effective_url)
    elif backend == "sqlite":
        store = SQLiteDocstore(DOCSTORE_SQLITE_PATH)
    else:
        raise ValueError(f"Unknown DOCSTORE_BACKEND: {backend}")
    try:
        store.init_schema()
        log.info("Docstore ready: %s (cache=%d)", backend, CHUNK_CACHE_SIZE)
    except Exception:
        log.warning("Docstore created but backend unavailable — will retry on first request: %s", backend)
    return CachedDocstore(store, maxsize=CHUNK_CACHE_SIZE)
