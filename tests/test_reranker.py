"""Tests for the reranker module."""

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _reset_module(monkeypatch):
    """Reset module-level state between tests."""
    import ragpipe.reranker as mod

    mod._model = None
    yield


# ── Helper to reimport with env vars ─────────────────────────────────────────


def _reload_reranker(**env_overrides):
    """Reimport reranker module with overridden env vars."""
    for k, v in env_overrides.items():
        os.environ[k] = v
    import ragpipe.reranker as mod

    importlib.reload(mod)
    for k in env_overrides:
        os.environ.pop(k, None)
    return mod


# ── Test data ────────────────────────────────────────────────────────────────


def _make_results(n=10):
    return [
        {
            "text": f"Document chunk {i} about topic {chr(65 + i)}",
            "source": f"doc{i}.md",
            "chunk_id": i,
            "chunk_total": n,
        }
        for i in range(n)
    ]


# ── Tests: enabled path ─────────────────────────────────────────────────────


def test_rerank_returns_top_n_results():
    mod = _reload_reranker(RERANKER_ENABLED="true", RERANKER_TOP_N="3", RERANKER_TOP_K="10")
    results = _make_results(10)
    ranked = mod.rerank("What is topic A?", results)
    assert len(ranked) == 3


def test_rerank_preserves_schema():
    mod = _reload_reranker(RERANKER_ENABLED="true", RERANKER_TOP_N="3")
    results = _make_results(5)
    ranked = mod.rerank("test query", results)
    for r in ranked:
        assert "text" in r
        assert "source" in r
        assert "reranker_score" in r


def test_rerank_scores_monotonically_decreasing():
    mod = _reload_reranker(RERANKER_ENABLED="true", RERANKER_TOP_N="10")
    results = _make_results(10)
    ranked = mod.rerank("test query", results)
    scores = [r["reranker_score"] for r in ranked]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Score at {i} ({scores[i]}) < score at {i + 1} ({scores[i + 1]})"


def test_rerank_empty_input():
    mod = _reload_reranker(RERANKER_ENABLED="true")
    assert mod.rerank("query", []) == []


def test_rerank_fewer_than_top_n():
    """If fewer results than top_n, return all of them."""
    mod = _reload_reranker(RERANKER_ENABLED="true", RERANKER_TOP_N="20")
    results = _make_results(3)
    ranked = mod.rerank("query", results)
    assert len(ranked) == 3


# ── Tests: disabled path ─────────────────────────────────────────────────────


def test_disabled_returns_top_n_passthrough():
    mod = _reload_reranker(RERANKER_ENABLED="false", RERANKER_TOP_N="3")
    results = _make_results(10)
    ranked = mod.rerank("query", results)
    assert len(ranked) == 3
    # Passthrough: same order, no reranker_score
    assert ranked == results[:3]


def test_disabled_preserves_schema():
    mod = _reload_reranker(RERANKER_ENABLED="false", RERANKER_TOP_N="5")
    results = _make_results(5)
    ranked = mod.rerank("query", results)
    for r in ranked:
        assert "text" in r
        assert "source" in r


def test_disabled_no_reranker_score():
    mod = _reload_reranker(RERANKER_ENABLED="false", RERANKER_TOP_N="5")
    results = _make_results(5)
    ranked = mod.rerank("query", results)
    for r in ranked:
        assert "reranker_score" not in r


# ── Tests: model swap ────────────────────────────────────────────────────────


def test_model_swap_via_env():
    """RERANKER_MODEL env var changes the model name without error."""
    mod = _reload_reranker(
        RERANKER_ENABLED="true",
        RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2",
        RERANKER_TOP_N="3",
    )
    results = _make_results(5)
    ranked = mod.rerank("What is topic A?", results)
    assert len(ranked) == 3
