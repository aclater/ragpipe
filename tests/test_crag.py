"""Tests for Corrective RAG (CRAG) — query rewrite on low retrieval confidence.

Verifies the CRAG pattern in retrieve_and_rerank():
- Rewrite triggered when reranking filters all chunks (low confidence)
- No rewrite when reranking passes chunks (high confidence)
- Maximum 1 retry
- CRAG metadata fields populated correctly
- Fallback to general when retry also fails
- rag_metadata in API response includes CRAG fields (fixes #50)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ragpipe.app as app

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_globals(monkeypatch):
    """Patch module-level singletons needed by retrieve_and_rerank."""
    monkeypatch.setattr(app, "_http_client", MagicMock())
    monkeypatch.setattr(app, "MODEL_URL", "http://test:8080")


def _make_chunk(doc_id="abc-123", chunk_id=0, text="test text", score=5.0):
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "title": "Test",
        "source": "test://source",
        "reranker_score": score,
    }


# ── test_crag_no_rewrite_on_high_confidence ──────────────────────────────────


@pytest.mark.asyncio
async def test_crag_no_rewrite_on_high_confidence():
    """When reranking returns chunks, CRAG should NOT trigger a rewrite."""
    chunks = [_make_chunk(score=5.0)]

    with (
        patch.object(app, "_search_qdrant_sync", return_value=[{"doc_id": "abc-123", "chunk_id": 0}]),
        patch.object(app, "_hydrate", new_callable=AsyncMock, return_value=chunks),
        patch.object(app, "_rerank_sync", return_value=chunks),
        patch.object(app, "_rewrite_query", new_callable=AsyncMock) as mock_rewrite,
    ):
        ranked, _candidates, crag_meta = await app.retrieve_and_rerank("test query")

    assert len(ranked) == 1
    assert crag_meta["retrieval_attempts"] == 1
    assert crag_meta["query_rewritten"] is False
    assert "original_query" not in crag_meta
    mock_rewrite.assert_not_awaited()


# ── test_crag_rewrite_on_low_confidence ──────────────────────────────────────


@pytest.mark.asyncio
async def test_crag_rewrite_on_low_confidence():
    """When reranking filters ALL chunks, CRAG should rewrite and retry."""
    candidates = [_make_chunk(score=-10.0)]
    rewritten_chunks = [_make_chunk(doc_id="def-456", score=5.0)]

    call_count = 0

    def search_side_effect(query, **kwargs):
        nonlocal call_count
        call_count += 1
        return [{"doc_id": "abc-123" if call_count == 1 else "def-456", "chunk_id": 0}]

    def rerank_side_effect(query, results, **kwargs):
        # First call: filter everything (low confidence)
        # Second call: pass chunks (rewritten query found better results)
        if query == "test query":
            return []
        return results

    async def hydrate_side_effect(refs, **kwargs):
        if refs[0].get("doc_id") == "abc-123":
            return candidates
        return rewritten_chunks

    with (
        patch.object(app, "_search_qdrant_sync", side_effect=search_side_effect),
        patch.object(app, "_hydrate", new_callable=AsyncMock, side_effect=hydrate_side_effect),
        patch.object(app, "_rerank_sync", side_effect=rerank_side_effect),
        patch.object(app, "_rewrite_query", new_callable=AsyncMock, return_value="rewritten test query"),
    ):
        ranked, _all_candidates, crag_meta = await app.retrieve_and_rerank("test query")

    assert len(ranked) == 1
    assert ranked[0]["doc_id"] == "def-456"
    assert crag_meta["retrieval_attempts"] == 2
    assert crag_meta["query_rewritten"] is True
    assert crag_meta["original_query"] == "test query"
    assert crag_meta["rewritten_query"] == "rewritten test query"


# ── test_crag_max_one_retry ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crag_max_one_retry():
    """CRAG should retry at most once — no infinite loops."""
    candidates = [_make_chunk(score=-10.0)]
    search_call_count = 0

    def search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        return [{"doc_id": f"doc-{search_call_count}", "chunk_id": 0}]

    with (
        patch.object(app, "_search_qdrant_sync", side_effect=search_side_effect),
        patch.object(app, "_hydrate", new_callable=AsyncMock, return_value=candidates),
        patch.object(app, "_rerank_sync", return_value=[]),  # Always filter
        patch.object(app, "_rewrite_query", new_callable=AsyncMock, return_value="rewritten query"),
    ):
        ranked, _all_candidates, crag_meta = await app.retrieve_and_rerank("test query")

    # Should have searched exactly 2 times (original + 1 retry), not more
    assert search_call_count == 2
    assert crag_meta["retrieval_attempts"] == 2
    assert len(ranked) == 0  # Both attempts failed


# ── test_crag_metadata_fields ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crag_metadata_fields():
    """CRAG metadata should include all required fields after rewrite."""
    candidates = [_make_chunk(score=-10.0)]

    with (
        patch.object(app, "_search_qdrant_sync", return_value=[{"doc_id": "abc-123", "chunk_id": 0}]),
        patch.object(app, "_hydrate", new_callable=AsyncMock, return_value=candidates),
        patch.object(app, "_rerank_sync", return_value=[]),
        patch.object(app, "_rewrite_query", new_callable=AsyncMock, return_value="improved query"),
    ):
        _, _, crag_meta = await app.retrieve_and_rerank("original question")

    # All required fields present
    assert "retrieval_attempts" in crag_meta
    assert "query_rewritten" in crag_meta
    assert "original_query" in crag_meta
    assert "rewritten_query" in crag_meta

    assert crag_meta["retrieval_attempts"] == 2
    assert crag_meta["query_rewritten"] is True
    assert crag_meta["original_query"] == "original question"
    assert crag_meta["rewritten_query"] == "improved query"


# ── test_crag_fallback_after_retry ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_crag_fallback_after_retry():
    """When CRAG retry also returns nothing, should fall back to empty results."""
    candidates = [_make_chunk(score=-10.0)]

    with (
        patch.object(app, "_search_qdrant_sync", return_value=[{"doc_id": "abc-123", "chunk_id": 0}]),
        patch.object(app, "_hydrate", new_callable=AsyncMock, return_value=candidates),
        patch.object(app, "_rerank_sync", return_value=[]),  # Always empty
        patch.object(app, "_rewrite_query", new_callable=AsyncMock, return_value="better question"),
    ):
        ranked, all_candidates, crag_meta = await app.retrieve_and_rerank("bad query")

    # Ranked is empty — CRAG retry also failed
    assert len(ranked) == 0
    # But candidates from both attempts should be in all_candidates
    assert len(all_candidates) >= 1
    # Metadata shows the retry happened
    assert crag_meta["retrieval_attempts"] == 2
    assert crag_meta["query_rewritten"] is True


# ── test_crag_no_rewrite_when_no_candidates ──────────────────────────────────


@pytest.mark.asyncio
async def test_crag_no_rewrite_when_no_candidates():
    """When Qdrant returns nothing at all, CRAG should not attempt rewrite."""
    with (
        patch.object(app, "_search_qdrant_sync", return_value=[]),
        patch.object(app, "_hydrate", new_callable=AsyncMock, return_value=[]),
        patch.object(app, "_rerank_sync", return_value=[]),
        patch.object(app, "_rewrite_query", new_callable=AsyncMock) as mock_rewrite,
    ):
        ranked, _candidates, crag_meta = await app.retrieve_and_rerank("query with no results")

    assert len(ranked) == 0
    assert crag_meta["retrieval_attempts"] == 1
    assert crag_meta["query_rewritten"] is False
    # Rewrite should NOT be called — no candidates means no point rewriting
    mock_rewrite.assert_not_awaited()


# ── test_crag_rewrite_query_function ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_crag_rewrite_query_function():
    """_rewrite_query should call the LLM and return the rewritten text."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "What patent law covers AI systems?"}}]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch.object(app, "_http_client", mock_client):
        result = await app._rewrite_query("AI patent requirements")

    assert result == "What patent law covers AI systems?"
    # Verify /nothink was included in the prompt
    call_args = mock_client.post.call_args
    messages = call_args.kwargs.get("json", call_args.args[1] if len(call_args.args) > 1 else {}).get("messages", [])
    assert any("/nothink" in msg.get("content", "") for msg in messages)


# ── test_rag_metadata_includes_crag_fields ──────────────────────────────────


def test_process_response_includes_crag_fields_when_rewritten():
    """process_response should include retrieval_attempts and query_rewritten
    in rag_metadata when CRAG rewrote the query (fixes #50)."""
    response_data = {
        "choices": [{"message": {"content": "Answer with no citations"}}],
    }
    ctx = {
        "user_query": "test query",
        "ranked": [],
        "retrieved_set": set(),
        "corpus_coverage": "none",
        "crag": {
            "retrieval_attempts": 2,
            "query_rewritten": True,
            "original_query": "test query",
            "rewritten_query": "improved query",
        },
    }

    result_data, _metadata = app.process_response(response_data, ctx)

    assert "rag_metadata" in result_data
    rag_meta = result_data["rag_metadata"]
    assert rag_meta["retrieval_attempts"] == 2
    assert rag_meta["query_rewritten"] is True
    assert rag_meta["original_query"] == "test query"
    assert rag_meta["rewritten_query"] == "improved query"


def test_process_response_includes_crag_fields_when_not_rewritten():
    """process_response should include retrieval_attempts=1 and
    query_rewritten=False when CRAG did not rewrite (fixes #50)."""
    response_data = {
        "choices": [{"message": {"content": "Answer with no citations"}}],
    }
    ctx = {
        "user_query": "test query",
        "ranked": [],
        "retrieved_set": set(),
        "corpus_coverage": "none",
        "crag": {
            "retrieval_attempts": 1,
            "query_rewritten": False,
        },
    }

    result_data, _metadata = app.process_response(response_data, ctx)

    assert "rag_metadata" in result_data
    rag_meta = result_data["rag_metadata"]
    assert rag_meta["retrieval_attempts"] == 1
    assert rag_meta["query_rewritten"] is False


def test_validate_streamed_response_includes_crag_fields():
    """_validate_streamed_response should include CRAG fields in metadata
    for streaming responses (fixes #50)."""
    ctx = {
        "user_query": "test query",
        "ranked": [],
        "retrieved_set": set(),
        "corpus_coverage": "none",
        "docstore": None,
        "crag": {
            "retrieval_attempts": 2,
            "query_rewritten": True,
            "original_query": "test query",
            "rewritten_query": "improved query",
        },
    }

    metadata = app._validate_streamed_response("Answer with no citations", ctx)

    assert metadata is not None
    assert metadata["retrieval_attempts"] == 2
    assert metadata["query_rewritten"] is True
    assert metadata["original_query"] == "test query"
    assert metadata["rewritten_query"] == "improved query"
