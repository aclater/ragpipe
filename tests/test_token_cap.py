"""Tests for default max_completion_tokens cap and thinking mode.

Verifies that process_chat_request():
- Injects DEFAULT_MAX_COMPLETION_TOKENS when client omits max_tokens
- Respects explicit client max_tokens without overriding
- Always sets enable_thinking=False via chat_template_kwargs
- Uses the RAGPIPE_MAX_COMPLETION_TOKENS env var when set

See: https://github.com/aclater/ragpipe/issues/60
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ragpipe.app as app


@pytest.fixture(autouse=True)
def _patch_globals(monkeypatch):
    """Patch module-level singletons so process_chat_request can run."""
    monkeypatch.setattr(app, "docstore", MagicMock())
    monkeypatch.setattr(app, "_http_client", MagicMock())
    monkeypatch.setattr(app, "MODEL_URL", "http://test:8080")


@pytest.fixture(autouse=True)
def _patch_retrieval():
    """Stub out retrieval pipeline — these tests only care about body mutation."""
    empty_crag = {"retrieval_attempts": 1, "query_rewritten": False}
    with patch.object(app, "retrieve_and_rerank", new_callable=AsyncMock, return_value=([], [], empty_crag)):
        yield


def _make_body(max_tokens=None, max_completion_tokens=None):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "What is RAG?"}],
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    return body


# ── Token cap tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_cap_injected_when_no_max_tokens():
    """When client sends no max_tokens, ragpipe injects DEFAULT_MAX_COMPLETION_TOKENS."""
    body = _make_body()
    result, _ = await app.process_chat_request(body)
    assert result["max_completion_tokens"] == app.DEFAULT_MAX_COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_default_cap_is_1024():
    """Default cap must be 1024 — not the old 4096 that caused runaway generation."""
    assert app.DEFAULT_MAX_COMPLETION_TOKENS == 1024


@pytest.mark.asyncio
async def test_explicit_max_tokens_not_overridden():
    """When client sets max_tokens, ragpipe must not inject max_completion_tokens."""
    body = _make_body(max_tokens=500)
    result, _ = await app.process_chat_request(body)
    assert "max_completion_tokens" not in result
    assert result["max_tokens"] == 500


@pytest.mark.asyncio
async def test_explicit_max_completion_tokens_not_overridden():
    """When client sets max_completion_tokens, ragpipe must not override it."""
    body = _make_body(max_completion_tokens=2048)
    result, _ = await app.process_chat_request(body)
    assert result["max_completion_tokens"] == 2048


@pytest.mark.asyncio
async def test_env_var_override(monkeypatch):
    """RAGPIPE_MAX_COMPLETION_TOKENS env var controls the default cap."""
    monkeypatch.setattr(app, "DEFAULT_MAX_COMPLETION_TOKENS", 512)
    body = _make_body()
    result, _ = await app.process_chat_request(body)
    assert result["max_completion_tokens"] == 512


# ── Thinking mode tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thinking_disabled_on_all_requests():
    """enable_thinking must be False for every request through process_chat_request."""
    body = _make_body()
    result, _ = await app.process_chat_request(body)
    assert result["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_thinking_disabled_even_with_existing_kwargs():
    """enable_thinking=False must be set even if chat_template_kwargs already exists."""
    body = _make_body()
    body["chat_template_kwargs"] = {"some_other_param": True}
    result, _ = await app.process_chat_request(body)
    assert result["chat_template_kwargs"]["enable_thinking"] is False
    assert result["chat_template_kwargs"]["some_other_param"] is True
