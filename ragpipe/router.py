"""Semantic router — classifies queries and dispatches to per-route pipelines.

Routes are defined in a YAML config file (RAGPIPE_ROUTES_FILE). Each route
specifies example utterances, a target LLM, and optional per-route RAG
backend configuration. The router embeds all examples at startup and
classifies incoming queries by cosine similarity (dot product on
L2-normalized embeddings).
"""

import dataclasses
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from qdrant_client import QdrantClient

from ragpipe.docstore import create_docstore
from ragpipe.models import Embedder

log = logging.getLogger("ragpipe.router")


@dataclasses.dataclass(frozen=True)
class RouteConfig:
    """Per-route configuration parsed from YAML."""

    name: str
    examples: list[str]
    model_url: str
    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    docstore_url: str | None = None
    docstore_backend: str = "postgres"
    system_prompt_file: str | None = None
    reranker_min_score: float = -5.0
    reranker_top_n: int = 5
    top_k: int = 20
    rag_enabled: bool = True


class RoutePipeline:
    """Lazily-initialized per-route resources.

    Owns: Qdrant client, docstore, system prompt.
    Does NOT own: embedder, reranker model, httpx client, thread pool (shared).
    """

    def __init__(self, config: RouteConfig):
        self.config = config
        self._qdrant: QdrantClient | None = None
        self._docstore = None
        self._system_prompt: str | None = None

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            url = self.config.qdrant_url or os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
            self._qdrant = QdrantClient(url=url, timeout=10)
            log.info("Route '%s': connected to Qdrant at %s", self.config.name, url)
        return self._qdrant

    @property
    def docstore(self):
        if self._docstore is None and self.config.rag_enabled:
            self._docstore = create_docstore(
                backend=self.config.docstore_backend,
                url=self.config.docstore_url,
            )
            log.info("Route '%s': docstore initialized (%s)", self.config.name, self.config.docstore_backend)
        return self._docstore

    @property
    def system_prompt(self) -> str | None:
        if self._system_prompt is None and self.config.system_prompt_file:
            path = Path(self.config.system_prompt_file)
            if path.exists():
                self._system_prompt = path.read_text().strip()
                log.info("Route '%s': loaded system prompt from %s", self.config.name, path)
            else:
                log.warning("Route '%s': prompt file not found: %s", self.config.name, path)
        return self._system_prompt

    async def close(self):
        if self._qdrant:
            self._qdrant.close()
            self._qdrant = None


class SemanticRouter:
    """Embedding-based query classifier.

    Pre-computes embeddings for all route example utterances at init time.
    At query time, computes cosine similarity (dot product on L2-normalized
    vectors) and selects the highest-scoring route above the threshold.
    """

    def __init__(
        self,
        routes: list[RouteConfig],
        embedder: Embedder,
        *,
        threshold: float = 0.3,
        fallback_route: str | None = None,
    ):
        self._routes = {r.name: r for r in routes}
        self._embedder = embedder
        self._threshold = threshold
        self._fallback_route = fallback_route
        self._pipelines: dict[str, RoutePipeline] = {}

        # Pre-compute example embeddings
        self._route_names: list[str] = []
        all_examples: list[str] = []
        for route in routes:
            for example in route.examples:
                self._route_names.append(route.name)
                all_examples.append(example)

        if all_examples:
            self._embeddings = embedder.embed(all_examples)
            log.info(
                "Router: embedded %d examples across %d routes (threshold=%.2f)",
                len(all_examples),
                len(routes),
                threshold,
            )
        else:
            self._embeddings = np.zeros((0, embedder.embedding_size))

        # Create pipelines lazily
        for route in routes:
            self._pipelines[route.name] = RoutePipeline(route)

    def classify(self, query_embedding: np.ndarray) -> tuple[str, float]:
        """Classify a query and return (route_name, score).

        Returns the fallback route if no route exceeds the threshold,
        or the first configured route if no fallback is defined.
        """
        if len(self._embeddings) == 0:
            fallback = self._fallback_route or next(iter(self._routes))
            return fallback, 0.0

        # Cosine similarity via dot product (embeddings are L2-normalized)
        scores = self._embeddings @ query_embedding

        # Group by route — take max score per route
        route_scores: dict[str, float] = {}
        for i, route_name in enumerate(self._route_names):
            score = float(scores[i])
            if route_name not in route_scores or score > route_scores[route_name]:
                route_scores[route_name] = score

        best_route = max(route_scores, key=route_scores.get)
        best_score = route_scores[best_route]

        if best_score < self._threshold:
            fallback = self._fallback_route or best_route
            return fallback, best_score

        return best_route, best_score

    def all_scores(self, query_embedding: np.ndarray) -> dict[str, float]:
        """Return max similarity score for each route. Used by /admin/classify."""
        if len(self._embeddings) == 0:
            return {}

        scores = self._embeddings @ query_embedding
        route_scores: dict[str, float] = {}
        for i, route_name in enumerate(self._route_names):
            score = float(scores[i])
            if route_name not in route_scores or score > route_scores[route_name]:
                route_scores[route_name] = score
        return route_scores

    def get_pipeline(self, route_name: str) -> RoutePipeline:
        """Get the pipeline for a route. Creates it lazily if needed."""
        return self._pipelines[route_name]

    async def close_all(self):
        """Close all per-route resources."""
        for pipeline in self._pipelines.values():
            await pipeline.close()


def load_routes_config(path: str) -> tuple[list[RouteConfig], float, str | None]:
    """Load routes configuration from a YAML file.

    Returns (routes, threshold, fallback_route).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "routes" not in raw:
        raise ValueError(f"Routes config must be a YAML dict with a 'routes' key: {path}")

    threshold = float(raw.get("threshold", 0.3))
    fallback_route = raw.get("fallback_route")

    routes = []
    for name, cfg in raw["routes"].items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Route '{name}' must be a dict")
        examples = cfg.get("examples", [])
        if not examples:
            raise ValueError(f"Route '{name}' must have at least one example utterance")
        model_url = cfg.get("model_url")
        if not model_url:
            raise ValueError(f"Route '{name}' must specify model_url")

        routes.append(
            RouteConfig(
                name=name,
                examples=examples,
                model_url=model_url,
                qdrant_url=cfg.get("qdrant_url"),
                qdrant_collection=cfg.get("qdrant_collection"),
                docstore_url=cfg.get("docstore_url"),
                docstore_backend=cfg.get("docstore_backend", "postgres"),
                system_prompt_file=cfg.get("system_prompt_file"),
                reranker_min_score=float(cfg.get("reranker_min_score", -5)),
                reranker_top_n=int(cfg.get("reranker_top_n", 5)),
                top_k=int(cfg.get("top_k", 20)),
                rag_enabled=cfg.get("rag_enabled", True),
            )
        )

    if fallback_route and fallback_route not in {r.name for r in routes}:
        raise ValueError(f"fallback_route '{fallback_route}' not found in routes")

    log.info("Loaded %d routes from %s (threshold=%.2f, fallback=%s)", len(routes), path, threshold, fallback_route)
    return routes, threshold, fallback_route
