"""Tests for admin endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Create a test client with admin token configured."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Test prompt v1")
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT_FILE", str(prompt_file))
    # Force reimport to pick up env vars
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    return TestClient(ragpipe.app.app)


def test_reload_prompt_success(client, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Test prompt v2")
    resp = client.post("/admin/reload-prompt", headers={"Authorization": "Bearer test-secret"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reloaded"
    assert data["changed"] is True
    assert "hash" in data
    assert data["source"].startswith("file:")


def test_reload_prompt_unauthorized(client):
    resp = client.post("/admin/reload-prompt", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_reload_prompt_no_auth_header(client):
    resp = client.post("/admin/reload-prompt")
    assert resp.status_code == 401


def test_reload_prompt_disabled_without_token(monkeypatch):
    """Without RAGPIPE_ADMIN_TOKEN, the endpoint returns 403."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "")
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    tc = TestClient(ragpipe.app.app)
    resp = tc.post("/admin/reload-prompt")
    assert resp.status_code == 403
