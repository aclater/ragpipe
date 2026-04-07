"""Live integration tests for ragpipe.

Tests run against the live ragpipe service (:8090).
Requires ragpipe to be running.

Run with:
    PYTHONPATH=. pytest tests/test_live.py -v --ragpipe-url=http://localhost:8090

Skip in CI (service not available):
    SKIP_LIVE_TESTS=1 pytest tests/test_live.py -v -m "not live"

Known issues:
- retrieval_attempts and query_rewritten are always null (issue #62) — CRAG fields bug
"""

import os

import httpx
import pytest
import requests

RAGPIPE_URL = os.environ.get("RAGPIPE_URL", "http://localhost:8090")
TIMEOUT = 60.0


def _is_ragpipe_available() -> bool:
    """Check if ragpipe is reachable."""
    try:
        httpx.get(f"{RAGPIPE_URL}/health", timeout=5)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SKIP_LIVE_TESTS") == "1" or not _is_ragpipe_available(),
        reason="ragpipe not available — set SKIP_LIVE_TESTS=1 to skip",
    ),
]


@pytest.fixture
def ragpipe_url():
    return RAGPIPE_URL


@pytest.fixture
def chat(ragpipe_url):
    def _chat(content, **kwargs):
        payload = {
            "model": kwargs.get("model", "default"),
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        r = requests.post(
            f"{ragpipe_url}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    return _chat


# ── Health and connectivity ────────────────────────────────────────────────────


def test_health_returns_200(ragpipe_url):
    resp = httpx.get(f"{ragpipe_url}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_metrics_returns_prometheus(ragpipe_url):
    resp = httpx.get(f"{ragpipe_url}/metrics", timeout=10)
    assert resp.status_code == 200
    assert "ragpipe_queries_total" in resp.text


def test_v1_models_returns_models(ragpipe_url):
    resp = httpx.get(f"{ragpipe_url}/v1/models", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) > 0


def test_admin_config_returns_routes(ragpipe_url):
    resp = httpx.get(
        f"{ragpipe_url}/admin/config",
        headers={"Authorization": "Bearer change-me"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "route_count" in data
    assert data["route_count"] >= 1


# ── Basic chat completions ─────────────────────────────────────────────────────


def test_simple_query_returns_choices(chat):
    data = chat("What is a patent?")
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]


def test_response_has_rag_metadata(chat):
    data = chat("What is a patent?")
    assert "rag_metadata" in data


def test_rag_metadata_has_required_fields(chat):
    data = chat("What is a patent?")
    meta = data.get("rag_metadata", {})
    assert "grounding" in meta
    assert "cited_chunks" in meta
    assert "corpus_coverage" in meta


def test_grounding_is_valid_value(chat):
    data = chat("What is a patent?")
    grounding = data.get("rag_metadata", {}).get("grounding")
    assert grounding in ("corpus", "general", "mixed"), f"Invalid grounding: {grounding}"


def test_corpus_query_returns_corpus_grounding(chat):
    data = chat("What is Adam Clater's job title?")
    grounding = data.get("rag_metadata", {}).get("grounding")
    assert grounding in ("corpus", "mixed"), f"Expected corpus/mixed, got {grounding}"


def test_off_topic_query_returns_general_grounding(chat):
    data = chat("What is 2+2?")
    grounding = data.get("rag_metadata", {}).get("grounding")
    assert grounding in ("general", "mixed"), f"Expected general/mixed, got {grounding}"


# ── CRAG fields ────────────────────────────────────────────────────────────────


@pytest.mark.xfail(reason="Bug: retrieval_attempts always null — issue #62", strict=False)
def test_retrieval_attempts_always_present(chat):
    data = chat("What is a patent?")
    attempts = data.get("rag_metadata", {}).get("retrieval_attempts")
    assert attempts is not None, "retrieval_attempts should not be null"
    assert isinstance(attempts, int)


@pytest.mark.xfail(reason="Bug: query_rewritten always null — issue #62", strict=False)
def test_query_rewritten_always_present(chat):
    data = chat("What is a patent?")
    rewritten = data.get("rag_metadata", {}).get("query_rewritten")
    assert rewritten is not None, "query_rewritten should not be null"
    assert isinstance(rewritten, bool)


@pytest.mark.xfail(reason="Bug: retrieval_attempts always null — issue #62", strict=False)
def test_retrieval_attempts_default_is_1(chat):
    data = chat("What is a patent?")
    attempts = data.get("rag_metadata", {}).get("retrieval_attempts")
    assert attempts == 1, f"Expected 1, got {attempts}"


@pytest.mark.xfail(reason="Bug: CRAG not working — issue #62", strict=False)
def test_crag_fires_on_vague_query(chat):
    data = chat("Tell me about patents")
    attempts = data.get("rag_metadata", {}).get("retrieval_attempts")
    assert attempts is not None and attempts >= 2, f"Expected >= 2 attempts on vague query, got {attempts}"


# ── Citation format ────────────────────────────────────────────────────────────


def test_citations_format_correct(chat):
    data = chat("What is a patent?")
    cited = data.get("rag_metadata", {}).get("cited_chunks", [])
    for chunk in cited:
        chunk_id = chunk.get("id", "")
        assert ":" in chunk_id, f"Expected doc_id:chunk_id format, got {chunk_id}"
        parts = chunk_id.split(":")
        assert len(parts) == 2, f"Expected doc_id:chunk_id format, got {chunk_id}"
        doc_id, cid = parts
        assert doc_id, f"Empty doc_id in {chunk_id}"
        assert cid.isdigit(), f"Invalid chunk_id (expected numeric) in {chunk_id}"


def test_cited_chunks_have_title(chat):
    data = chat("What is a patent?")
    cited = data.get("rag_metadata", {}).get("cited_chunks", [])
    if cited:
        for chunk in cited:
            assert "title" in chunk, f"Chunk missing title: {chunk}"
            assert chunk["title"], "Title should not be empty"


def test_cited_chunks_have_source(chat):
    data = chat("What is a patent?")
    cited = data.get("rag_metadata", {}).get("cited_chunks", [])
    if cited:
        for chunk in cited:
            assert "source" in chunk, f"Chunk missing source: {chunk}"


# ── Collection routing ─────────────────────────────────────────────────────────


def test_personnel_query_routes_to_personnel(chat):
    data = chat("What is Adam Clater's job title?")
    cited = data.get("rag_metadata", {}).get("cited_chunks", [])
    assert len(cited) > 0, "Personnel query should return chunks"


def test_mpep_query_routes_to_mpep(chat):
    data = chat("What are the requirements for a patent?")
    cited = data.get("rag_metadata", {}).get("cited_chunks", [])
    assert len(cited) > 0, "Patent query should return chunks"


def test_nato_query_routes_to_nato(chat):
    data = chat("What is NATO?")
    cited = data.get("rag_metadata", {}).get("cited_chunks", [])
    grounding = data.get("rag_metadata", {}).get("grounding")
    assert len(cited) > 0, f"NATO query should return chunks, got grounding={grounding}"


# ── Streaming ──────────────────────────────────────────────────────────────────


def test_stream_true_returns_sse(ragpipe_url):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = requests.post(
        f"{ragpipe_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()
    assert "text/event-stream" in resp.headers.get("content-type", "")
    lines = [line.decode("utf-8") if isinstance(line, bytes) else line for line in resp.iter_lines()]
    assert len(lines) > 0
    assert any("chatcmpl" in line for line in lines)


def test_stream_false_returns_json(ragpipe_url):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    resp = requests.post(
        f"{ragpipe_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    assert "application/json" in resp.headers.get("content-type", "")


# ── Admin endpoints ─────────────────────────────────────────────────────────────


def test_reload_prompt_returns_200(ragpipe_url):
    resp = httpx.post(
        f"{ragpipe_url}/admin/reload-prompt",
        headers={"Authorization": "Bearer change-me"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data or "reloaded" in str(data).lower()


def test_reload_routes_returns_200(ragpipe_url):
    resp = httpx.post(
        f"{ragpipe_url}/admin/reload-routes",
        headers={"Authorization": "Bearer change-me"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data or "reloaded" in str(data).lower()


# ── Error handling ─────────────────────────────────────────────────────────────


@pytest.mark.xfail(reason="Bug: empty messages cause LLM to hang instead of returning 400", strict=False)
def test_empty_messages_returns_400(ragpipe_url):
    payload = {"model": "default", "messages": [], "stream": False}
    resp = requests.post(
        f"{ragpipe_url}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    assert resp.status_code in (400, 422)


def test_missing_model_still_works(ragpipe_url):
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    resp = requests.post(
        f"{ragpipe_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200


def test_very_long_query_handled(ragpipe_url):
    long_query = "Explain " * 250
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": long_query}],
        "stream": False,
    }
    resp = requests.post(
        f"{ragpipe_url}/v1/chat/completions",
        json=payload,
        timeout=TIMEOUT,
    )
    assert resp.status_code in (200, 400, 422, 500)
