"""Tests for the ONNX Runtime model wrappers."""

import numpy as np

from ragpipe.models import Embedder, Reranker

# ── Embedder ─────────────────────────────────────────────────────────────────


def test_embed_one_returns_vector():
    emb = Embedder()
    vec = emb.embed_one("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] > 0


def test_embed_batch_shape():
    emb = Embedder()
    vecs = emb.embed(["hello", "world", "test"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 3
    assert vecs.shape[1] == emb.embedding_size


def test_embed_normalized():
    """Embeddings should be L2-normalized (unit vectors)."""
    emb = Embedder()
    vecs = emb.embed(["test sentence"])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5


def test_embed_different_texts_different_vectors():
    emb = Embedder()
    vecs = emb.embed(["cats are great", "quantum physics equations"])
    similarity = np.dot(vecs[0], vecs[1])
    assert similarity < 0.95  # different texts should not be identical


# ── Reranker ─────────────────────────────────────────────────────────────────


def test_reranker_score_returns_floats():
    rr = Reranker()
    scores = rr.score("what is python?", ["Python is a programming language", "The weather is nice"])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_reranker_relevant_scores_higher():
    rr = Reranker()
    scores = rr.score("what is python?", ["Python is a programming language", "The weather is nice today"])
    assert scores[0] > scores[1]


def test_reranker_empty_documents():
    rr = Reranker()
    assert rr.score("query", []) == []
