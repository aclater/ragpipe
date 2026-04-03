"""Reranker stage — scores and reorders retrieved chunks.

Runs after Qdrant vector search, before results are passed to the LLM.
Uses a cross-encoder model via fastembed (ONNX Runtime, CPU-only) to score
(query, document) pairs and returns the top_n highest-scoring results.

Default model: Xenova/ms-marco-MiniLM-L-6-v2 (ONNX, 22M, English).
No PyTorch dependency — fastembed uses ONNX Runtime for lightweight inference.
"""

import logging
import os
import time
from typing import Any

log = logging.getLogger("ragpipe.reranker")

RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N = int(os.environ.get("RERANKER_TOP_N", "5"))
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))

_model = None


def _get_model():
    """Lazy-load the cross-encoder reranker via fastembed."""
    global _model
    if _model is not None:
        return _model

    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log.info("Loading reranker model %s (fastembed/ONNX, threads=%d)", RERANKER_MODEL, ONNX_THREADS)
    _model = TextCrossEncoder(model_name=RERANKER_MODEL, threads=ONNX_THREADS)
    log.info("Reranker model loaded")
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
    scores = list(model.rerank(query, documents))
    elapsed_ms = (time.monotonic() - start) * 1000

    log.debug("Reranked %d candidates in %.1f ms", len(results), elapsed_ms)

    for result, score in zip(results, scores, strict=False):
        result["reranker_score"] = float(score)

    ranked = sorted(results, key=lambda r: r["reranker_score"], reverse=True)
    return ranked[:RERANKER_TOP_N]
