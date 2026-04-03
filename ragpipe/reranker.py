"""Reranker stage — scores and reorders retrieved chunks.

Runs after Qdrant vector search, before results are passed to the LLM.
Uses a cross-encoder model via ONNX Runtime (CPU-only) to score
(query, document) pairs and returns the top_n highest-scoring results.

Default model: Xenova/ms-marco-MiniLM-L-6-v2 (ONNX, 22M, English).
"""

import logging
import os
import time
from typing import Any

log = logging.getLogger("ragpipe.reranker")

RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N = int(os.environ.get("RERANKER_TOP_N", "5"))
RERANKER_MIN_SCORE = float(os.environ.get("RERANKER_MIN_SCORE", "-5"))

_model = None


def _get_model():
    """Lazy-load the cross-encoder reranker."""
    global _model
    if _model is not None:
        return _model

    from ragpipe.models import Reranker

    _model = Reranker(repo_id=RERANKER_MODEL)
    _model.load()
    return _model


def warm_up():
    """Load the model eagerly at startup instead of on first request."""
    _get_model()


def rerank(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rerank retrieval results by cross-encoder relevance score.

    Args:
        query: The user's search query.
        results: List of dicts with at least a "text" key. Passed through
                 from Qdrant search output.

    Returns:
        The top_n results sorted by descending reranker score, in the same
        schema as the input. Each dict gets a "reranker_score" key added.
        If reranking is disabled, returns the input unchanged (up to top_n).
    """
    if not RERANKER_ENABLED:
        return results[:RERANKER_TOP_N]

    if not results:
        return results

    model = _get_model()

    documents = [r["text"] for r in results]

    start = time.monotonic()
    scores = model.score(query, documents)
    elapsed_ms = (time.monotonic() - start) * 1000

    log.debug("Reranked %d candidates in %.1f ms", len(results), elapsed_ms)

    for result, score in zip(results, scores, strict=False):
        result["reranker_score"] = score

    ranked = sorted(results, key=lambda r: r["reranker_score"], reverse=True)

    # Filter out chunks below the minimum confidence score.
    # When all chunks are filtered, the model gets empty context and
    # falls back to general knowledge with the ⚠️ prefix — this is
    # correct behavior for adversarial or off-topic queries.
    filtered = [r for r in ranked if r["reranker_score"] >= RERANKER_MIN_SCORE]
    if len(filtered) < len(ranked):
        log.info(
            "Filtered %d/%d chunks below min_score=%.1f (top=%.4f, cutoff=%.4f)",
            len(ranked) - len(filtered),
            len(ranked),
            RERANKER_MIN_SCORE,
            ranked[0]["reranker_score"],
            ranked[-1]["reranker_score"] if ranked else 0,
        )

    return filtered[:RERANKER_TOP_N]
