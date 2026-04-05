from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

ragpipe_queries_total = Counter(
    "ragpipe_queries_total",
    "Total number of ragpipe queries",
    ["grounding", "route"],
    registry=REGISTRY,
)

ragpipe_query_latency_seconds = Histogram(
    "ragpipe_query_latency_seconds",
    "Query latency in seconds",
    ["route"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

ragpipe_chunks_retrieved_total = Counter(
    "ragpipe_chunks_retrieved_total",
    "Total chunks retrieved from docstore per query",
    ["route"],
    registry=REGISTRY,
)

ragpipe_chunks_passed_reranker_total = Counter(
    "ragpipe_chunks_passed_reranker_total",
    "Total chunks passed to reranker per query",
    ["route"],
    registry=REGISTRY,
)

ragpipe_invalid_citations_total = Counter(
    "ragpipe_invalid_citations_total",
    "Total invalid citations detected",
    ["type"],
    registry=REGISTRY,
)

ragpipe_embed_cache_hits_total = Counter(
    "ragpipe_embed_cache_hits_total",
    "Total embedding cache hits",
    registry=REGISTRY,
)

ragpipe_embed_cache_misses_total = Counter(
    "ragpipe_embed_cache_misses_total",
    "Total embedding cache misses",
    registry=REGISTRY,
)

ragpipe_chunk_cache_hits_total = Counter(
    "ragpipe_chunk_cache_hits_total",
    "Total chunk cache hits (docstore LRU)",
    registry=REGISTRY,
)

ragpipe_chunk_cache_misses_total = Counter(
    "ragpipe_chunk_cache_misses_total",
    "Total chunk cache misses (docstore LRU)",
    registry=REGISTRY,
)

ragpipe_startup_ready_timestamp = Gauge(
    "ragpipe_startup_ready_timestamp",
    "Unix timestamp when ragpipe became ready",
    registry=REGISTRY,
)

ragpipe_onnx_sessions_active = Gauge(
    "ragpipe_onnx_sessions_active",
    "Number of active ONNX sessions (embedder + reranker)",
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    return generate_latest(REGISTRY)
