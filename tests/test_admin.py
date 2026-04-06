"""Tests for admin endpoints."""

from unittest.mock import MagicMock, patch

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


AUTH = {"Authorization": "Bearer test-secret"}


def test_reload_prompt_success(client, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Test prompt v2")
    resp = client.post("/admin/reload-prompt", headers=AUTH)
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


def test_reload_prompt_auto_generates_token_when_empty(monkeypatch, tmp_path):
    """When RAGPIPE_ADMIN_TOKEN is empty and no token file exists,
    a token is auto-generated so admin endpoints are accessible."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "")
    token_file = tmp_path / "admin_token"
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN_FILE", str(token_file))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Test prompt")
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT_FILE", str(prompt_file))
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)

    # Token was auto-generated
    assert ragpipe.app.ADMIN_TOKEN != ""
    assert token_file.is_file()
    saved_token = token_file.read_text().strip()
    assert saved_token == ragpipe.app.ADMIN_TOKEN

    # Endpoint works with the auto-generated token
    tc = TestClient(ragpipe.app.app)
    resp = tc.post(
        "/admin/reload-prompt",
        headers={"Authorization": f"Bearer {ragpipe.app.ADMIN_TOKEN}"},
    )
    assert resp.status_code == 200

    # But unauthenticated access is still rejected
    resp = tc.post("/admin/reload-prompt")
    assert resp.status_code == 401


def test_get_config_requires_auth(client):
    """GET /admin/config requires admin auth."""
    resp = client.get("/admin/config")
    assert resp.status_code == 401

    resp = client.get("/admin/config", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "routes_file" in data
    assert "routes_hash" in data
    assert "route_count" in data
    assert "prompt_file" in data
    assert "prompt_hash" in data
    assert "prompt_source" in data


def test_reload_routes_auth(monkeypatch, tmp_path):
    """Reload-routes requires valid auth token."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    routes_file = tmp_path / "routes.yaml"
    routes_file.write_text(
        "routes:\n  test_route:\n    examples:\n      - hello world\n    model_url: http://localhost:8080\n"
    )
    monkeypatch.setenv("RAGPIPE_ROUTES_FILE", str(routes_file))

    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    tc = TestClient(ragpipe.app.app)

    # Without auth
    resp = tc.post("/admin/reload-routes")
    assert resp.status_code == 401

    # Wrong token
    resp = tc.post("/admin/reload-routes", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_reload_routes_no_routes_file(monkeypatch):
    """Without RAGPIPE_ROUTES_FILE, returns 404."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    monkeypatch.setenv("RAGPIPE_ROUTES_FILE", "")
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    tc = TestClient(ragpipe.app.app)
    resp = tc.post("/admin/reload-routes", headers=AUTH)
    assert resp.status_code == 404


def test_reload_routes_invalid_yaml(monkeypatch, tmp_path):
    """With malformed YAML, returns 400."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    routes_file = tmp_path / "routes.yaml"
    routes_file.write_text("invalid: yaml: content:")
    monkeypatch.setenv("RAGPIPE_ROUTES_FILE", str(routes_file))

    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    tc = TestClient(ragpipe.app.app)

    resp = tc.post("/admin/reload-routes", headers=AUTH)
    assert resp.status_code == 400
    assert "Failed to parse" in resp.json()["error"]


def test_reload_routes_no_change(monkeypatch, tmp_path):
    """When file hash matches, returns changed: false without re-embedding."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    routes_file = tmp_path / "routes.yaml"
    routes_content = (
        "routes:\n  test_route:\n    examples:\n      - hello world\n    model_url: http://localhost:8080\n"
    )
    routes_file.write_text(routes_content)
    monkeypatch.setenv("RAGPIPE_ROUTES_FILE", str(routes_file))

    import hashlib
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)

    # Pre-set the hash to match the current file — simulates a previous load
    ragpipe.app._routes_hash = hashlib.sha256(routes_file.read_bytes()).hexdigest()
    ragpipe.app._routes_count = 1

    tc = TestClient(ragpipe.app.app)
    resp = tc.post("/admin/reload-routes", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["changed"] is False
    assert data["route_count"] == 1


def test_reload_routes_detects_change(monkeypatch, tmp_path):
    """When file hash differs, triggers reload with new router."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    routes_file = tmp_path / "routes.yaml"
    routes_file.write_text(
        "routes:\n  test_route:\n    examples:\n      - hello world\n    model_url: http://localhost:8080\n"
    )
    monkeypatch.setenv("RAGPIPE_ROUTES_FILE", str(routes_file))

    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)

    # Set a stale hash so the reload detects a change
    ragpipe.app._routes_hash = "stale_hash"

    # Mock SemanticRouter so we don't need the real embedder
    mock_router = MagicMock()
    mock_router.close_all = MagicMock()
    with patch("ragpipe.router.SemanticRouter", return_value=mock_router):
        tc = TestClient(ragpipe.app.app)
        resp = tc.post("/admin/reload-routes", headers=AUTH)

    assert resp.status_code == 200
    data = resp.json()
    assert data["changed"] is True
    assert data["route_count"] == 1
    assert ragpipe.app._router is mock_router


def test_token_loaded_from_file(monkeypatch, tmp_path):
    """Token file is used when env var is empty."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "")
    token_file = tmp_path / "admin_token"
    token_file.write_text("file-secret-token\n")
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN_FILE", str(token_file))
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    assert ragpipe.app.ADMIN_TOKEN == "file-secret-token"  # noqa: S105


def test_env_var_takes_precedence_over_file(monkeypatch, tmp_path):
    """RAGPIPE_ADMIN_TOKEN env var takes precedence over token file."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "env-token")
    token_file = tmp_path / "admin_token"
    token_file.write_text("file-token\n")
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN_FILE", str(token_file))
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    assert ragpipe.app.ADMIN_TOKEN == "env-token"  # noqa: S105


def test_auto_generated_token_file_permissions(monkeypatch, tmp_path):
    """Auto-generated token file must have restricted permissions (0o600)."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "")
    token_file = tmp_path / "admin_token"
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN_FILE", str(token_file))
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    assert token_file.stat().st_mode & 0o777 == 0o600


def test_reload_system_prompt_alias(monkeypatch, tmp_path):
    """POST /admin/reload-system-prompt is an alias for reload-prompt."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Test prompt v1")
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT_FILE", str(prompt_file))

    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    tc = TestClient(ragpipe.app.app)

    prompt_file.write_text("Updated prompt v3")
    resp = tc.post("/admin/reload-system-prompt", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reloaded"
    assert data["changed"] is True
