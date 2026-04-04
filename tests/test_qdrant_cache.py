"""Tests for Qdrant result caching and score threshold in app._search_qdrant_sync."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _reset_app_cache():
    """Clear the module-level Qdrant cache between tests."""
    import ragpipe.app as app_mod

    with app_mod._qdrant_cache_lock:
        app_mod._qdrant_cache.clear()


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure each test starts with a clean cache."""
    _reset_app_cache()
    yield
    _reset_app_cache()


def _make_point(doc_id: str, chunk_id: int, score: float = 0.9):
    """Create a mock Qdrant point with a payload."""
    return SimpleNamespace(
        payload={"doc_id": doc_id, "chunk_id": chunk_id, "source": "test.pdf"},
        score=score,
    )


def _make_qdrant_client(points):
    """Create a mock QdrantClient that returns the given points."""
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=points)
    return client


# ── Cache hit / miss ────────────────────────────────────────────────────────


def test_cache_miss_then_hit():
    """First call is a miss (calls Qdrant), second call is a hit (skips Qdrant)."""
    import ragpipe.app as app_mod

    points = [_make_point("doc-1", 0), _make_point("doc-1", 1)]
    mock_client = _make_qdrant_client(points)

    with (
        patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
        patch.object(app_mod, "_check_collection", return_value=True),
        patch.object(app_mod, "qdrant", mock_client),
    ):
        # First call — cache miss
        result1 = app_mod._search_qdrant_sync("hello world", collection="test_coll")
        assert len(result1) == 2
        assert mock_client.query_points.call_count == 1

        # Second call — cache hit, no additional Qdrant call
        result2 = app_mod._search_qdrant_sync("hello world", collection="test_coll")
        assert result2 == result1
        assert mock_client.query_points.call_count == 1


def test_different_queries_are_separate_cache_entries():
    """Different query text produces different cache keys."""
    import ragpipe.app as app_mod

    points = [_make_point("doc-1", 0)]
    mock_client = _make_qdrant_client(points)

    with (
        patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
        patch.object(app_mod, "_check_collection", return_value=True),
        patch.object(app_mod, "qdrant", mock_client),
    ):
        app_mod._search_qdrant_sync("query A", collection="coll")
        app_mod._search_qdrant_sync("query B", collection="coll")
        # Both are misses — two Qdrant calls
        assert mock_client.query_points.call_count == 2


def test_different_collections_are_separate_cache_entries():
    """Same query text but different collection names produce separate cache keys."""
    import ragpipe.app as app_mod

    points = [_make_point("doc-1", 0)]
    mock_client = _make_qdrant_client(points)

    with (
        patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
        patch.object(app_mod, "_check_collection", return_value=True),
        patch.object(app_mod, "qdrant", mock_client),
    ):
        app_mod._search_qdrant_sync("same query", collection="coll_a")
        app_mod._search_qdrant_sync("same query", collection="coll_b")
        assert mock_client.query_points.call_count == 2


def test_empty_results_not_cached():
    """Empty results (no points) should not be cached — collection may appear later."""
    import ragpipe.app as app_mod

    empty_client = _make_qdrant_client([])
    full_client = _make_qdrant_client([_make_point("doc-1", 0)])

    with (
        patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
        patch.object(app_mod, "_check_collection", return_value=True),
    ):
        # First call — empty result, should NOT be cached
        with patch.object(app_mod, "qdrant", empty_client):
            result1 = app_mod._search_qdrant_sync("query", collection="coll")
            assert result1 == []

        # Second call — should hit Qdrant again (not served from cache)
        with patch.object(app_mod, "qdrant", full_client):
            result2 = app_mod._search_qdrant_sync("query", collection="coll")
            assert len(result2) == 1
            assert full_client.query_points.call_count == 1


def test_cache_respects_max_size():
    """Cache evicts oldest entries when it exceeds QDRANT_CACHE_SIZE."""
    import ragpipe.app as app_mod

    points = [_make_point("doc-1", 0)]
    mock_client = _make_qdrant_client(points)

    original_size = app_mod.QDRANT_CACHE_SIZE
    try:
        app_mod.QDRANT_CACHE_SIZE = 2  # Small cache for testing

        with (
            patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
            patch.object(app_mod, "_check_collection", return_value=True),
            patch.object(app_mod, "qdrant", mock_client),
        ):
            # Fill cache with 3 entries (exceeds size=2)
            app_mod._search_qdrant_sync("query_1", collection="coll")
            app_mod._search_qdrant_sync("query_2", collection="coll")
            app_mod._search_qdrant_sync("query_3", collection="coll")

            assert mock_client.query_points.call_count == 3
            assert len(app_mod._qdrant_cache) == 2

            # query_1 should have been evicted — re-query triggers Qdrant call
            app_mod._search_qdrant_sync("query_1", collection="coll")
            assert mock_client.query_points.call_count == 4

            # query_3 should still be cached
            app_mod._search_qdrant_sync("query_3", collection="coll")
            assert mock_client.query_points.call_count == 4  # No new call
    finally:
        app_mod.QDRANT_CACHE_SIZE = original_size


# ── Score threshold ─────────────────────────────────────────────────────────


def test_score_threshold_passed_to_qdrant():
    """score_threshold parameter is forwarded to qc.query_points()."""
    import ragpipe.app as app_mod

    points = [_make_point("doc-1", 0)]
    mock_client = _make_qdrant_client(points)

    with (
        patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
        patch.object(app_mod, "_check_collection", return_value=True),
        patch.object(app_mod, "qdrant", mock_client),
    ):
        app_mod._search_qdrant_sync("query", collection="coll", score_threshold=0.5)
        call_kwargs = mock_client.query_points.call_args
        assert call_kwargs.kwargs.get("score_threshold") == 0.5


def test_score_threshold_uses_env_default():
    """When no score_threshold is passed, QDRANT_SCORE_THRESHOLD env var default is used."""
    import ragpipe.app as app_mod

    points = [_make_point("doc-1", 0)]
    mock_client = _make_qdrant_client(points)

    with (
        patch.object(app_mod, "_embed_query", return_value=tuple([0.1] * 384)),
        patch.object(app_mod, "_check_collection", return_value=True),
        patch.object(app_mod, "qdrant", mock_client),
    ):
        app_mod._search_qdrant_sync("query", collection="coll")
        call_kwargs = mock_client.query_points.call_args
        assert call_kwargs.kwargs.get("score_threshold") == app_mod.QDRANT_SCORE_THRESHOLD


# ── RouteConfig score threshold ─────────────────────────────────────────────


def test_route_config_qdrant_score_threshold_default():
    """RouteConfig.qdrant_score_threshold defaults to None (uses global env default)."""
    from ragpipe.router import RouteConfig

    rc = RouteConfig(name="test", examples=["hi"], model_url="http://localhost:8080")
    assert rc.qdrant_score_threshold is None


def test_route_config_qdrant_score_threshold_from_yaml(tmp_path):
    """qdrant_score_threshold is parsed from YAML route config."""
    import yaml

    from ragpipe.router import load_routes_config

    config = {
        "routes": {
            "precise": {
                "examples": ["exact match needed"],
                "model_url": "http://localhost:8080",
                "qdrant_score_threshold": 0.7,
            },
        },
    }
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.dump(config))
    routes, _, _ = load_routes_config(str(path))
    assert routes[0].qdrant_score_threshold == 0.7
