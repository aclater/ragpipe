"""Live integration tests for CRAG metadata in API responses (fixes #62).

These tests call the live ragpipe endpoint and assert that CRAG metadata
fields are always present in rag_metadata — never null. They require a
running ragpipe instance at RAGPIPE_URL (default http://localhost:8090).

Run:
    pytest tests/test_integration_crag.py -v --ragpipe-url=http://localhost:8090
"""

import json

import httpx
import pytest

pytestmark = pytest.mark.integration

RAGPIPE_URL_DEFAULT = "http://localhost:8090"
TIMEOUT = 120  # seconds — LLM inference can be slow


@pytest.fixture(scope="module")
def ragpipe_url(request):
    return request.config.getoption("--ragpipe-url", default=RAGPIPE_URL_DEFAULT)


@pytest.fixture(scope="module")
def ragpipe_healthy(ragpipe_url):
    """Skip all tests in this module if ragpipe is not reachable."""
    try:
        r = httpx.get(f"{ragpipe_url}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"ragpipe not healthy at {ragpipe_url}: {r.status_code}")
    except httpx.ConnectError:
        pytest.skip(f"ragpipe not reachable at {ragpipe_url}")
    return ragpipe_url


def _chat(url: str, content: str) -> dict:
    """Send a non-streaming chat completion request and return parsed JSON."""
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        },
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


class TestCragMetadataInResponse:
    """CRAG metadata fields must always be present in rag_metadata."""

    def test_crag_fields_always_present(self, ragpipe_healthy):
        """Every response must include retrieval_attempts and query_rewritten
        in rag_metadata — they must never be null (fixes #62)."""
        url = ragpipe_healthy
        data = _chat(url, "What does MPEP Chapter 2100 cover?")

        assert "rag_metadata" in data, "Response missing rag_metadata entirely"
        meta = data["rag_metadata"]

        assert "retrieval_attempts" in meta, "retrieval_attempts missing from rag_metadata"
        assert meta["retrieval_attempts"] is not None, "retrieval_attempts is null"
        assert isinstance(meta["retrieval_attempts"], int), (
            f"retrieval_attempts should be int, got {type(meta['retrieval_attempts'])}"
        )
        assert meta["retrieval_attempts"] >= 1, (
            f"retrieval_attempts should be >= 1, got {meta['retrieval_attempts']}"
        )

        assert "query_rewritten" in meta, "query_rewritten missing from rag_metadata"
        assert meta["query_rewritten"] is not None, "query_rewritten is null"
        assert isinstance(meta["query_rewritten"], bool), (
            f"query_rewritten should be bool, got {type(meta['query_rewritten'])}"
        )

    def test_crag_metadata_low_confidence_query(self, ragpipe_healthy):
        """A deliberately vague query should return retrieval_attempts >= 1
        and the field must not be null — even if CRAG doesn't trigger."""
        url = ragpipe_healthy
        data = _chat(url, "patent claims software")

        meta = data.get("rag_metadata", {})

        assert meta.get("retrieval_attempts") is not None, (
            "retrieval_attempts is null on low-confidence query"
        )
        assert meta["retrieval_attempts"] >= 1

        assert meta.get("query_rewritten") is not None, (
            "query_rewritten is null on low-confidence query"
        )

    def test_crag_rewrite_metadata_consistent(self, ragpipe_healthy):
        """When query_rewritten is true, original_query and rewritten_query
        must also be present. When false, retrieval_attempts must be 1."""
        url = ragpipe_healthy
        data = _chat(url, "who works there")

        meta = data.get("rag_metadata", {})

        assert "retrieval_attempts" in meta
        assert "query_rewritten" in meta

        if meta["query_rewritten"]:
            assert meta["retrieval_attempts"] >= 2, (
                "query_rewritten=true but retrieval_attempts < 2"
            )
            assert "original_query" in meta and meta["original_query"], (
                "query_rewritten=true but original_query missing"
            )
            assert "rewritten_query" in meta and meta["rewritten_query"], (
                "query_rewritten=true but rewritten_query missing"
            )
        else:
            assert meta["retrieval_attempts"] == 1, (
                f"query_rewritten=false but retrieval_attempts={meta['retrieval_attempts']}"
            )


class TestCragFieldsMultipleQueries:
    """Run several queries to verify CRAG fields are consistently present."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is Article 5 of the NATO treaty?",
            "patent claims software",
            "who works there",
            "article 5 thing",
            "What does the MPEP say about prior art?",
        ],
        ids=["specific", "vague-patent", "vague-personnel", "vague-nato", "specific-mpep"],
    )
    def test_crag_fields_present_across_queries(self, ragpipe_healthy, query):
        """retrieval_attempts and query_rewritten must be present and non-null
        for every query type — specific or vague."""
        url = ragpipe_healthy
        data = _chat(url, query)

        meta = data.get("rag_metadata", {})
        assert meta.get("retrieval_attempts") is not None, (
            f"retrieval_attempts is null for query: {query}"
        )
        assert meta.get("query_rewritten") is not None, (
            f"query_rewritten is null for query: {query}"
        )
