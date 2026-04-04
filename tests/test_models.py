"""Tests for the ONNX Runtime model wrappers."""

from unittest import mock

import numpy as np

from ragpipe.models import Embedder, Reranker, _get_providers

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


# ── Provider detection ──────────────────────────────────────────────────────


def test_get_providers_includes_cpu():
    """_get_providers() must always include CPUExecutionProvider."""
    providers = _get_providers()
    assert "CPUExecutionProvider" in providers


def _patch_providers(available, env_overrides=None):
    """Helper: patch ort.get_available_providers and RAGPIPE_DEVICE env var."""
    import os as _os

    if env_overrides is None:
        # Remove RAGPIPE_DEVICE if present so auto-detect runs cleanly.
        clean_env = {k: v for k, v in _os.environ.items() if k != "RAGPIPE_DEVICE"}
        env_ctx = mock.patch.dict("os.environ", clean_env, clear=True)
    else:
        env_ctx = mock.patch.dict("os.environ", env_overrides)

    provider_ctx = mock.patch("ragpipe.models.ort.get_available_providers", return_value=available)
    # Return a combined context manager via contextlib.ExitStack isn't needed;
    # callers use two separate `with` lines combined per ruff SIM117.
    return provider_ctx, env_ctx


def test_get_providers_cpu_only_when_no_gpu():
    """With only CPU available, _get_providers() returns CPU alone."""
    p_ctx, e_ctx = _patch_providers(["CPUExecutionProvider"])
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["CPUExecutionProvider"]


def test_get_providers_prefers_cuda():
    """When CUDA is available, it should be preferred over CPU."""
    p_ctx, e_ctx = _patch_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_prefers_migraphx():
    """When MIGraphX is available (but not CUDA), it should be preferred."""
    p_ctx, e_ctx = _patch_providers(["MIGraphXExecutionProvider", "CPUExecutionProvider"])
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["MIGraphXExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_cuda_over_migraphx():
    """When both CUDA and MIGraphX are available, CUDA should win."""
    p_ctx, e_ctx = _patch_providers(["CUDAExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider"])
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_forced_rocm_maps_to_rocm_ep():
    """RAGPIPE_DEVICE=rocm should map to ROCMExecutionProvider."""
    p_ctx, e_ctx = _patch_providers(
        ["ROCMExecutionProvider", "CPUExecutionProvider"],
        env_overrides={"RAGPIPE_DEVICE": "rocm"},
    )
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_prefers_rocm_over_cpu():
    """When ROCMExecutionProvider is available (but not CUDA or MIGraphX), it should be preferred."""
    p_ctx, e_ctx = _patch_providers(["ROCMExecutionProvider", "CPUExecutionProvider"])
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_migraphx_over_rocm():
    """When both MIGraphX and ROCM EPs are available, MIGraphX should win."""
    p_ctx, e_ctx = _patch_providers(["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"])
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["MIGraphXExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_forced_migraphx():
    """RAGPIPE_DEVICE=migraphx should select MIGraphXExecutionProvider."""
    p_ctx, e_ctx = _patch_providers(
        ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
        env_overrides={"RAGPIPE_DEVICE": "migraphx"},
    )
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["MIGraphXExecutionProvider", "CPUExecutionProvider"]


def test_get_providers_forced_cpu():
    """RAGPIPE_DEVICE=cpu should force CPU even when GPU is available."""
    p_ctx, e_ctx = _patch_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        env_overrides={"RAGPIPE_DEVICE": "cpu"},
    )
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["CPUExecutionProvider"]


def test_get_providers_forced_cuda_unavailable_falls_back():
    """RAGPIPE_DEVICE=cuda when CUDA is not available should fall back to CPU."""
    p_ctx, e_ctx = _patch_providers(
        ["CPUExecutionProvider"],
        env_overrides={"RAGPIPE_DEVICE": "cuda"},
    )
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["CPUExecutionProvider"]


def test_get_providers_invalid_device_falls_back():
    """RAGPIPE_DEVICE with an invalid value should fall back to auto-detect."""
    p_ctx, e_ctx = _patch_providers(
        ["CPUExecutionProvider"],
        env_overrides={"RAGPIPE_DEVICE": "tpu"},
    )
    with p_ctx, e_ctx:
        providers = _get_providers()
    assert providers == ["CPUExecutionProvider"]
