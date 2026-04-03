"""Tests for the semantic router — config parsing, classification, pipeline lifecycle."""

import pytest
import yaml

from ragpipe.router import RouteConfig, RoutePipeline, SemanticRouter, load_routes_config

# ── Config parsing ───────────────────────────────────────────────────────────


def test_load_valid_config(tmp_path):
    config = {
        "threshold": 0.4,
        "fallback_route": "general",
        "routes": {
            "topic_a": {
                "examples": ["What about topic A?", "Tell me about A"],
                "model_url": "http://localhost:8080",
                "qdrant_collection": "collection_a",
            },
            "general": {
                "examples": ["Hello", "What is the weather?"],
                "model_url": "http://localhost:8081",
                "rag_enabled": False,
            },
        },
    }
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.dump(config))
    routes, threshold, fallback = load_routes_config(str(path))
    assert len(routes) == 2
    assert threshold == 0.4
    assert fallback == "general"
    route_names = {r.name for r in routes}
    assert "topic_a" in route_names
    assert "general" in route_names
    topic_a = next(r for r in routes if r.name == "topic_a")
    general = next(r for r in routes if r.name == "general")
    assert topic_a.qdrant_collection == "collection_a"
    assert general.rag_enabled is False


def test_load_config_defaults(tmp_path):
    config = {
        "routes": {
            "only_route": {
                "examples": ["test query"],
                "model_url": "http://localhost:8080",
            },
        },
    }
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.dump(config))
    routes, threshold, fallback = load_routes_config(str(path))
    assert threshold == 0.3
    assert fallback is None
    r = routes[0]
    assert r.rag_enabled is True
    assert r.reranker_min_score == -5.0
    assert r.reranker_top_n == 5
    assert r.top_k == 20


def test_load_config_missing_examples(tmp_path):
    config = {"routes": {"bad": {"model_url": "http://localhost:8080"}}}
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.dump(config))
    with pytest.raises(ValueError, match="at least one example"):
        load_routes_config(str(path))


def test_load_config_missing_model_url(tmp_path):
    config = {"routes": {"bad": {"examples": ["test"]}}}
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.dump(config))
    with pytest.raises(ValueError, match="model_url"):
        load_routes_config(str(path))


def test_load_config_invalid_fallback(tmp_path):
    config = {
        "fallback_route": "nonexistent",
        "routes": {"real": {"examples": ["test"], "model_url": "http://localhost:8080"}},
    }
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.dump(config))
    with pytest.raises(ValueError, match="not found in routes"):
        load_routes_config(str(path))


# ── Classification ───────────────────────────────────────────────────────────


@pytest.fixture
def router():
    """Create a router with a real embedder for classification tests."""
    from ragpipe.models import Embedder

    emb = Embedder()
    emb.load()

    routes = [
        RouteConfig(
            name="defense",
            examples=[
                "What does the defense authorization act say?",
                "Summarize military procurement provisions",
                "What is the defense budget?",
            ],
            model_url="http://localhost:8080",
        ),
        RouteConfig(
            name="science",
            examples=[
                "Explain quantum entanglement",
                "How does photosynthesis work?",
                "What is the theory of relativity?",
            ],
            model_url="http://localhost:8081",
            rag_enabled=False,
        ),
    ]
    return SemanticRouter(routes, emb, threshold=0.3, fallback_route="science")


def test_classify_returns_best_route(router):
    from ragpipe.models import Embedder

    emb = Embedder()
    vec = emb.embed_one("What are the military spending provisions?")
    route, score = router.classify(vec)
    assert route == "defense"
    assert score > 0.3


def test_classify_science_route(router):
    from ragpipe.models import Embedder

    emb = Embedder()
    vec = emb.embed_one("How do black holes form?")
    route, _score = router.classify(vec)
    assert route == "science"


def test_classify_fallback_below_threshold():
    """Queries below threshold should fall back."""
    from ragpipe.models import Embedder

    emb = Embedder()
    emb.load()
    routes = [
        RouteConfig(
            name="very_specific",
            examples=["Exact match only for this very specific topic about widget calibration"],
            model_url="http://localhost:8080",
        ),
        RouteConfig(
            name="fallback",
            examples=["general question"],
            model_url="http://localhost:8081",
        ),
    ]
    r = SemanticRouter(routes, emb, threshold=0.99, fallback_route="fallback")
    vec = emb.embed_one("completely unrelated query about cooking")
    route, _score = r.classify(vec)
    assert route == "fallback"


def test_all_scores(router):
    from ragpipe.models import Embedder

    emb = Embedder()
    vec = emb.embed_one("defense budget")
    scores = router.all_scores(vec)
    assert "defense" in scores
    assert "science" in scores
    assert scores["defense"] > scores["science"]


# ── Pipeline lifecycle ───────────────────────────────────────────────────────


def test_pipeline_config():
    config = RouteConfig(
        name="test",
        examples=["test"],
        model_url="http://localhost:8080",
        qdrant_collection="test_collection",
        rag_enabled=True,
    )
    pipeline = RoutePipeline(config)
    assert pipeline.config.name == "test"
    assert pipeline.config.qdrant_collection == "test_collection"


def test_pipeline_system_prompt_from_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Custom route prompt")
    config = RouteConfig(
        name="test",
        examples=["test"],
        model_url="http://localhost:8080",
        system_prompt_file=str(prompt_file),
    )
    pipeline = RoutePipeline(config)
    assert pipeline.system_prompt == "Custom route prompt"


def test_pipeline_system_prompt_none_without_file():
    config = RouteConfig(name="test", examples=["test"], model_url="http://localhost:8080")
    pipeline = RoutePipeline(config)
    assert pipeline.system_prompt is None


def test_rag_disabled_route():
    config = RouteConfig(name="general", examples=["test"], model_url="http://localhost:8080", rag_enabled=False)
    assert config.rag_enabled is False


# ── Backward compatibility ───────────────────────────────────────────────────


def test_no_routes_file(monkeypatch):
    """Without RAGPIPE_ROUTES_FILE, the app should work in single-pipeline mode."""
    monkeypatch.delenv("RAGPIPE_ROUTES_FILE", raising=False)
    import importlib

    import ragpipe.app

    importlib.reload(ragpipe.app)
    assert ragpipe.app.ROUTES_FILE is None
