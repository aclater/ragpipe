"""Tests for the /metrics Prometheus endpoint."""

import pytest
from fastapi.testclient import TestClient

from ragpipe.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_metrics_endpoint_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_returns_prometheus_format(client):
    response = client.get("/metrics")
    text = response.text
    assert "ragpipe_queries_total" in text
    assert "ragpipe_query_latency_seconds" in text
    assert "ragpipe_chunks_retrieved_total" in text
    assert "ragpipe_chunks_passed_reranker_total" in text
    assert "ragpipe_invalid_citations_total" in text


def test_metrics_content_type_is_plain_text(client):
    response = client.get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_has_startup_timestamp(client):
    response = client.get("/metrics")
    text = response.text
    assert "ragpipe_startup_ready_timestamp" in text


def test_metrics_no_auth_required(client):
    response = client.get("/metrics")
    assert response.status_code == 200
