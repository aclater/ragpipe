# ragpipe

RAG proxy with semantic routing, corpus-preferring grounding, and citation validation. Sits between your client and multiple LLMs, intercepts OpenAI-compatible chat completions, and enriches queries with retrieval-augmented context from Qdrant vector databases backed by Postgres document stores.

What makes it different: ragpipe classifies queries semantically and routes them to the right LLM and RAG backend — keeping data sources isolated across routing domains. It validates the model's citations against what was actually retrieved, strips hallucinated references from non-streaming responses, and classifies the grounding mode (corpus/general/mixed).

### Single-pipeline request flow
![Request flow](architecture.svg)

### Multi-host semantic routing
![Routing](architecture-routing.svg)

### Package components
![Components](architecture-components.svg)

## How it works

1. **Classify** the query semantically (cosine similarity, <1ms) and select a route
2. **Embed** the query (ONNX Runtime, gte-modernbert-base, LRU cached)
3. **Search** the route's Qdrant collection for top-K candidate vectors
4. **Hydrate** chunk text from the route's Postgres document store (async, cached)
5. **Rerank** with cross-encoder (ONNX Runtime, MiniLM-L-6-v2), filter below min score
6. **Inject** the route's system prompt + context with `[doc_id:chunk_id]` labels
7. **Forward** to the route's LLM (streaming or non-streaming)
8. **Post-process**: parse citations, validate, classify grounding, attach `rag_metadata`, emit audit log

Without a routes config (`RAGPIPE_ROUTES_FILE` unset), ragpipe operates as a single-pipeline proxy — fully backward compatible.

## Quick start (container)

```bash
podman build -t ragpipe .
podman run --rm -p 8090:8090 \
    -e MODEL_URL=http://host.containers.internal:8080 \
    -e QDRANT_URL=http://host.containers.internal:6333 \
    -e DOCSTORE_URL=postgresql://user:pass@host.containers.internal:5432/db \
    ragpipe
```

Or pull the published image:

```bash
podman pull ghcr.io/aclater/ragpipe:main
```

## Quick start (pip)

```bash
pip install git+https://github.com/aclater/ragpipe
ragpipe
```

Or:

```bash
python -m ragpipe
```

## Configuration

All configuration is via environment variables. The 5 core variables you need to get started:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | `http://127.0.0.1:8080` | LLM endpoint to forward to |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant vector search endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `RAG_PROXY_PORT` | `8090` | Port to listen on |
| `RAG_TOP_K` | `20` | Qdrant candidate count before reranking |

For the full reference covering embedding, reranking, grounding, routing, and admin, see [docs/configuration.md](docs/configuration.md).

## Documentation

- [Configuration](docs/configuration.md) — Full environment variable reference
- [API](docs/api.md) — Endpoints, `rag_metadata`, streaming behavior, admin endpoints
- [Architecture](docs/architecture.md) — Performance benchmarks and pipeline details
- [Routing](docs/routing.md) — Semantic routing configuration and debugging

## Known issues

- **Streaming citation stripping**: Streaming responses are audited and validated post-hoc (dual-path accumulation), but invalid citations cannot be stripped because the text has already been delivered to the client. Invalid citations are logged as errors. Non-streaming requests strip invalid citations before delivery.
- **LLM phrasing variance**: The negative finding classifier depends on the model using recognizable negation patterns ("no evidence", "not mentioned", etc.) before the `⚠️` marker. When the model phrases its negative finding differently, the response may be classified as `mixed` instead of `general`.
- **Passthrough model list**: `/v1/models` passthrough always returns the global upstream's model list, not the routed model's list.
- **No upstream failover**: If a route's upstream LLM is down, ragpipe returns 502. No automatic fallback to another route.

## Development

```bash
pip install '.[dev]'
python -m pytest tests/ -v
ruff check && ruff format --check
```

## License

AGPL-3.0-or-later
