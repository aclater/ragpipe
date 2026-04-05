"""Tests for the document store — both SQLite and Postgres backends, plus cache layer.

SQLite tests run unconditionally. Postgres tests require a live database
and are skipped if DOCSTORE_URL is not set or the connection fails.
"""

import os

import pytest

from ragpipe.docstore import CachedDocstore, SQLiteDocstore, create_docstore

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_store(tmp_path):
    store = SQLiteDocstore(str(tmp_path / "test.db"))
    store.init_schema()
    return CachedDocstore(store, maxsize=100)


@pytest.fixture
def pg_store():
    url = os.environ.get("DOCSTORE_URL", "postgresql://litellm:litellm@127.0.0.1:5432/litellm")
    try:
        from ragpipe.docstore import PostgresDocstore

        store = PostgresDocstore(url)
        store.init_schema()
        # Clean up test data
        conn = store._get_sync_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id LIKE 'test-%'")
        cached = CachedDocstore(store, maxsize=100)
        yield cached
        # Clean up after
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id LIKE 'test-%'")
    except Exception:
        pytest.skip("Postgres not available")


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, sqlite_store, pg_store):
    if request.param == "sqlite":
        return sqlite_store
    return pg_store


# ── Tests ────────────────────────────────────────────────────────────────────


def test_upsert_and_get_single(store):
    store.upsert_chunk("test-doc1", 0, "hello world", "test.md")
    assert store.get_chunk("test-doc1", 0)["text"] == "hello world"


def test_upsert_idempotent(store):
    """Re-inserting same (doc_id, chunk_id) updates text, no duplicates."""
    store.upsert_chunk("test-doc2", 0, "version 1", "test.md")
    store.upsert_chunk("test-doc2", 0, "version 2", "test.md")
    assert store.get_chunk("test-doc2", 0)["text"] == "version 2"


def test_get_missing_returns_none(store):
    assert store.get_chunk("nonexistent", 999) is None


def test_batch_upsert(store):
    chunks = [{"doc_id": "test-batch", "chunk_id": i, "text": f"chunk {i}", "source": "batch.md"} for i in range(5)]
    store.upsert_chunks(chunks)
    for i in range(5):
        assert store.get_chunk("test-batch", i)["text"] == f"chunk {i}"


def test_batch_upsert_idempotent(store):
    """Re-ingesting the same doc produces no duplicate entries."""
    chunks = [{"doc_id": "test-idem", "chunk_id": i, "text": f"v1 chunk {i}", "source": "idem.md"} for i in range(3)]
    store.upsert_chunks(chunks)

    chunks_v2 = [{"doc_id": "test-idem", "chunk_id": i, "text": f"v2 chunk {i}", "source": "idem.md"} for i in range(3)]
    store.upsert_chunks(chunks_v2)

    for i in range(3):
        assert store.get_chunk("test-idem", i)["text"] == f"v2 chunk {i}"


def test_batch_get(store):
    chunks = [{"doc_id": "test-bget", "chunk_id": i, "text": f"text {i}", "source": "bg.md"} for i in range(5)]
    store.upsert_chunks(chunks)

    refs = [("test-bget", 0), ("test-bget", 2), ("test-bget", 4)]
    result = store.get_chunks(refs)

    assert len(result) == 3
    assert result[("test-bget", 0)]["text"] == "text 0"
    assert result[("test-bget", 2)]["text"] == "text 2"
    assert result[("test-bget", 4)]["text"] == "text 4"


def test_batch_get_with_missing(store):
    """Missing refs are simply absent from the result dict."""
    store.upsert_chunk("test-miss", 0, "exists", "m.md")
    refs = [("test-miss", 0), ("test-miss", 999)]
    result = store.get_chunks(refs)
    assert len(result) == 1
    assert ("test-miss", 999) not in result


def test_batch_get_empty(store):
    assert store.get_chunks([]) == {}


def test_delete_doc(store):
    chunks = [{"doc_id": "test-del", "chunk_id": i, "text": f"del {i}", "source": "d.md"} for i in range(3)]
    store.upsert_chunks(chunks)
    store.delete_doc("test-del")
    for i in range(3):
        assert store.get_chunk("test-del", i) is None


def test_factory_sqlite(tmp_path):
    os.environ["DOCSTORE_SQLITE_PATH"] = str(tmp_path / "factory.db")
    store = create_docstore(backend="sqlite")
    store.upsert_chunk("test-factory", 0, "works", "f.md")
    assert store.get_chunk("test-factory", 0)["text"] == "works"
    os.environ.pop("DOCSTORE_SQLITE_PATH", None)


# ── Cache tests ──────────────────────────────────────────────────────────────


def test_cache_hit_on_second_get(tmp_path):
    """Second get_chunk should hit the cache, not the DB."""
    backend = SQLiteDocstore(str(tmp_path / "cache.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-c1", 0, "cached text", "c.md")
    store.get_chunk("test-c1", 0)  # miss — populates cache
    store.get_chunk("test-c1", 0)  # hit

    stats = store.cache_stats
    assert stats["hits"] >= 1
    assert stats["size"] == 1


def test_cache_batch_get_partial_hit(tmp_path):
    """Batch get with some cached and some not should only query DB for misses."""
    backend = SQLiteDocstore(str(tmp_path / "partial.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-p1", 0, "text 0", "p.md")
    store.upsert_chunk("test-p1", 1, "text 1", "p.md")
    store.upsert_chunk("test-p1", 2, "text 2", "p.md")

    # Prime cache with chunk 0 only
    store.get_chunk("test-p1", 0)

    # Batch get all three — chunk 0 should be cached, 1 and 2 fetched
    result = store.get_chunks([("test-p1", 0), ("test-p1", 1), ("test-p1", 2)])
    assert len(result) == 3
    assert result[("test-p1", 0)]["text"] == "text 0"


def test_cache_invalidated_on_upsert(tmp_path):
    """Upsert should update the cache so stale data is never returned."""
    backend = SQLiteDocstore(str(tmp_path / "inval.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-inv", 0, "v1", "i.md")
    assert store.get_chunk("test-inv", 0)["text"] == "v1"

    store.upsert_chunk("test-inv", 0, "v2", "i.md")
    assert store.get_chunk("test-inv", 0)["text"] == "v2"


def test_cache_invalidated_on_delete(tmp_path):
    """Delete should evict all chunks for the doc from cache."""
    backend = SQLiteDocstore(str(tmp_path / "del.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunks([{"doc_id": "test-cdel", "chunk_id": i, "text": f"t{i}", "source": "d.md"} for i in range(3)])
    # Prime cache
    for i in range(3):
        store.get_chunk("test-cdel", i)
    assert store.cache_stats["size"] == 3

    store.delete_doc("test-cdel")
    assert store.cache_stats["size"] == 0
    assert store.get_chunk("test-cdel", 0) is None


def test_cache_evicts_lru(tmp_path):
    """Cache should evict least-recently-used entries when full."""
    backend = SQLiteDocstore(str(tmp_path / "lru.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=3)

    for i in range(5):
        store.upsert_chunk("test-lru", i, f"text {i}", "l.md")
        store.get_chunk("test-lru", i)  # populate cache

    # Only last 3 should be cached
    assert store.cache_stats["size"] == 3


@pytest.mark.asyncio
async def test_cache_async_get(tmp_path):
    """get_chunks_async should use the cache layer."""
    backend = SQLiteDocstore(str(tmp_path / "async.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-async", 0, "async text", "a.md")
    # Prime cache
    store.get_chunk("test-async", 0)

    result = await store.get_chunks_async([("test-async", 0)])
    assert result[("test-async", 0)]["text"] == "async text"
    assert store.cache_stats["hits"] >= 2


def test_close_cleans_up(tmp_path):
    """close() should release connections and clear cache."""
    backend = SQLiteDocstore(str(tmp_path / "close.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-close", 0, "text", "c.md")
    store.get_chunk("test-close", 0)
    assert store.cache_stats["size"] == 1

    store.close()
    assert store.cache_stats["size"] == 0


# ── Title hydration tests ────────────────────────────────────────────────────


def test_docstore_hydration_includes_title(tmp_path):
    """get_chunks() returns {text, title, source} for each key."""
    backend = SQLiteDocstore(str(tmp_path / "title.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-t1", 0, "chunk text", "src.md", title="My Document")
    result = store.get_chunks([("test-t1", 0)])

    assert ("test-t1", 0) in result
    chunk = result[("test-t1", 0)]
    assert chunk["text"] == "chunk text"
    assert chunk["title"] == "My Document"
    assert chunk["source"] == "src.md"


def test_docstore_hydration_title_fallback(tmp_path):
    """When title is not provided, returns empty strings without crashing."""
    backend = SQLiteDocstore(str(tmp_path / "fallback.db"))
    backend.init_schema()
    store = CachedDocstore(backend, maxsize=100)

    store.upsert_chunk("test-t2", 0, "no title text", "src.md")
    result = store.get_chunks([("test-t2", 0)])

    chunk = result[("test-t2", 0)]
    assert chunk["text"] == "no title text"
    assert chunk["title"] == ""
    assert chunk["source"] == "src.md"
