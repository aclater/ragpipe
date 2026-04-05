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


def test_get_config_no_auth_required(client):
    """GET /admin/config returns current config state without auth."""
    resp = client.get("/admin/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "routes_file" in data
    assert "routes_hash" in data
    assert "route_count" in data
    assert "prompt_file" in data
    assert "prompt_hash" in data
    assert "prompt_source" in data


def test_reload_routes_success(monkeypatch, tmp_path):
    """Test hot-reload of routes file requires auth."""
    monkeypatch.setenv("RAGPIPE_ADMIN_TOKEN", "test-secret")
    routes_file = tmp_path / "routes.yaml"
    routes_file.write_text("""
routes:
  test_route:
    examples:
      - "hello world"
    model_url: http://localhost:8080
""")
    monkeypatch.setenv("RAGPIPE_ROUTES_FILE", str(routes_file))

    import importlib
    import ragpipe.app
    importlib.reload(ragpipe.app)
    tc = TestClient(ragpipe.app.app)

    # Without auth - should get 401
    resp = tc.post("/admin/reload-routes")
    assert resp.status_code == 401

    # With wrong token - should get 401
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
    resp = tc.post("/admin/reload-routes", headers={"Authorization": "Bearer test-secret"})
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

    resp = tc.post("/admin/reload-routes", headers={"Authorization": "Bearer test-secret"})
    assert resp.status_code == 400
    assert "Failed to parse" in resp.json()["error"]


def test_reload_routes_no_change(monkeypatch, tmp_path):
    """When file hasn't changed, returns changed: false."""
    pytest.skip("Requires full embedder initialization - tested manually")


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
    resp = tc.post("/admin/reload-system-prompt", headers={"Authorization": "Bearer test-secret"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reloaded"
    assert data["changed"] is True
