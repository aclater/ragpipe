"""Tests for the document store — both SQLite and Postgres backends.

SQLite tests run unconditionally. Postgres tests require a live database
and are skipped if DOCSTORE_URL is not set or the connection fails.
"""

import os

import pytest

from ragpipe.docstore import SQLiteDocstore, create_docstore

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sqlite_store(tmp_path):
    store = SQLiteDocstore(str(tmp_path / "test.db"))
    store.init_schema()
    return store


@pytest.fixture
def pg_store():
    url = os.environ.get("DOCSTORE_URL", "postgresql://litellm:litellm@127.0.0.1:5432/litellm")
    try:
        from ragpipe.docstore import PostgresDocstore

        store = PostgresDocstore(url)
        store.init_schema()
        # Clean up test data
        with store._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id LIKE 'test-%'")
        yield store
        # Clean up after
        with store._conn.cursor() as cur:
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
    assert store.get_chunk("test-doc1", 0) == "hello world"


def test_upsert_idempotent(store):
    """Re-inserting same (doc_id, chunk_id) updates text, no duplicates."""
    store.upsert_chunk("test-doc2", 0, "version 1", "test.md")
    store.upsert_chunk("test-doc2", 0, "version 2", "test.md")
    assert store.get_chunk("test-doc2", 0) == "version 2"


def test_get_missing_returns_none(store):
    assert store.get_chunk("nonexistent", 999) is None


def test_batch_upsert(store):
    chunks = [{"doc_id": "test-batch", "chunk_id": i, "text": f"chunk {i}", "source": "batch.md"} for i in range(5)]
    store.upsert_chunks(chunks)
    for i in range(5):
        assert store.get_chunk("test-batch", i) == f"chunk {i}"


def test_batch_upsert_idempotent(store):
    """Re-ingesting the same doc produces no duplicate entries."""
    chunks = [{"doc_id": "test-idem", "chunk_id": i, "text": f"v1 chunk {i}", "source": "idem.md"} for i in range(3)]
    store.upsert_chunks(chunks)

    chunks_v2 = [{"doc_id": "test-idem", "chunk_id": i, "text": f"v2 chunk {i}", "source": "idem.md"} for i in range(3)]
    store.upsert_chunks(chunks_v2)

    for i in range(3):
        assert store.get_chunk("test-idem", i) == f"v2 chunk {i}"


def test_batch_get(store):
    chunks = [{"doc_id": "test-bget", "chunk_id": i, "text": f"text {i}", "source": "bg.md"} for i in range(5)]
    store.upsert_chunks(chunks)

    refs = [("test-bget", 0), ("test-bget", 2), ("test-bget", 4)]
    result = store.get_chunks(refs)

    assert len(result) == 3
    assert result[("test-bget", 0)] == "text 0"
    assert result[("test-bget", 2)] == "text 2"
    assert result[("test-bget", 4)] == "text 4"


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
    assert store.get_chunk("test-factory", 0) == "works"
    os.environ.pop("DOCSTORE_SQLITE_PATH", None)
