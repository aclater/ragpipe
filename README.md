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
4. **Hydrate** chunk text from the route's Postgres document store (async, cached), including title extraction per source
5. **Rerank** with cross-encoder (ONNX Runtime, MiniLM-L-6-v2), filter below min score
6. **Inject** the route's system prompt + context with `[doc_id:chunk_id]` labels
7. **Forward** to the route's LLM (streaming or non-streaming)
8. **Post-process**: parse citations, validate, classify grounding, attach `rag_metadata`, emit audit log

Without a routes config (`RAGPIPE_ROUTES_FILE` unset), ragpipe operates as a single-pipeline proxy — fully backward compatible.

## Multi-collection routing

When `RAGPIPE_ROUTES_FILE` is configured, ragpipe routes queries to different Qdrant collections based on semantic classification. Each route specifies its own collection, LLM, docstore, and system prompt. Current live collections: `personnel`, `nato`, `mpep`, `documents`.

```
Query → SemanticClassifier → Route: personnel → Qdrant:personnel → Postgres chunks + titles
                                              → Route: nato → Qdrant:nato → Postgres chunks + titles
                                              → Route: mpep → Qdrant:mpep → Postgres chunks + titles
```

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
# CPU variant
podman pull ghcr.io/aclater/ragpipe:main

# ROCm variant (AMD GPU, MIGraphX)
podman pull ghcr.io/aclater/ragpipe:main-rocm
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

All configuration is via environment variables.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | `http://127.0.0.1:8080` | LLM endpoint to forward to |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant vector search endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name (used in single-pipeline mode) |
| `RAG_PROXY_PORT` | `8090` | Port to listen on |
| `RAG_TOP_K` | `20` | Qdrant candidate count before reranking (must be ≤ `MIGRAPHX_BATCH_SIZE` if using MIGraphX) |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `Alibaba-NLP/gte-modernbert-base` | HuggingFace repo for ONNX embedding model (quantized, 768d) |
| `ONNX_PAD_LENGTH` | `128` | Fixed padding length for tokenizer — prevents MIGraphX recompilation per input shape |
| `EMBED_CACHE_SIZE` | `256` | LRU cache size for query embeddings |
| `ONNX_THREADS` | `4` | ONNX Runtime intra-op thread count per model |
| `RAGPIPE_MODEL_CACHE` | `~/.cache/ragpipe` | Local directory for downloaded ONNX models |

### Reranking

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | HuggingFace repo for ONNX cross-encoder model |
| `RERANKER_TOP_N` | `5` | Max results to keep after reranking |
| `RERANKER_MIN_SCORE` | `-5` | Minimum reranker score — chunks below this are filtered out |

### Document store

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCSTORE_BACKEND` | `postgres` | `postgres` or `sqlite` |
| `DOCSTORE_URL` | *(required for postgres)* | Postgres connection string |
| `DOCSTORE_SQLITE_PATH` | `/tmp/docstore.db` | SQLite file path |
| `CHUNK_CACHE_SIZE` | `2048` | LRU cache entries for hydrated chunk text |

### Grounding

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_SYSTEM_PROMPT_FILE` | — | Path to custom system prompt file (hot-reloadable) |
| `RAGPIPE_SYSTEM_PROMPT` | — | Inline custom system prompt text |
| `THINKING_BUDGET` | `1024` | Token budget for model reasoning |

### Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_ROUTES_FILE` | — | Path to YAML routes config. When unset, single-pipeline mode |
| `RAGPIPE_ADMIN_TOKEN` | — | Bearer token for admin endpoints. Admin endpoints are disabled when unset |

For the full reference covering routing configuration and debugging, see [docs/routing.md](docs/routing.md).

## GPU acceleration (gfx1151: CPU; other AMD: MIGraphX)

**On gfx1151 (Strix Halo), both embedder and reranker use CPU.** MIGraphX is skipped automatically via `_is_gfx1151()` detection. This is because MIGraphX tensors land in GTT (system RAM) on UMA APUs, not VRAM — ROCm VMM is not supported on gfx1151 by design. Benchmarks confirm CPU outperforms MIGraphX-on-GTT for models this small.

**On other AMD GPUs**, MIGraphXExecutionProvider is used normally. ROCMExecutionProvider is ABI-incompatible with ROCm 7.x and silently falls back to CPU.

**MXR pre-compilation cache — 39x startup improvement:**

On first startup, ragpipe compiles ONNX models to `.mxr` format via `ORT_MIGRAPHX_MODEL_CACHE_PATH`. Subsequent startups load pre-compiled `.mxr` files directly (non-gfx1151 AMD GPUs with MIGraphX):

| Startup type | Time | Mechanism |
|---|---|---|
| Cold (first ever) | ~3:53 | JIT compilation |
| Warm (.mxr cached) | ~6 seconds | Load from `ORT_MIGRAPHX_MODEL_CACHE_PATH` |

Cached files are ~149 MB per model. The cache persists across restarts — do not restart ragpipe casually; the warm path is fast enough for development but the cold path still takes ~4 minutes.

```
Environment for MXR cache (non-gfx1151 MIGraphX):
- ORT_MIGRAPHX_MODEL_CACHE_PATH=/home/default/.cache/ragpipe/mxr
```

On gfx1151, the embedder and reranker use CPU instead of MIGraphX, so MXR caching does not apply. The cold start still takes ~3:53 on first boot for ONNX model JIT compilation.

**⚠️ Cold start: ~3:53 on first query.** Warm start (MXR cached, non-gfx1151): ~6 seconds. Plan restarts accordingly.

## Prometheus metrics

ragpipe exposes a `/metrics` endpoint for Prometheus scraping:

```
# HELP ragpipe_queries_total Total number of RAG queries processed
# TYPE ragpipe_queries_total counter
ragpipe_queries_total 1234

# HELP ragpipe_chunks_retrieved_total Total chunks retrieved from docstore
# TYPE ragpipe_chunks_retrieved_total counter
ragpipe_chunks_retrieved_total 5678

# HELP ragpipe_invalid_citations_total Citations that could not be validated
# TYPE ragpipe_invalid_citations_total counter
ragpipe_invalid_citations_total 12

# HELP ragpipe_embed_cache_hits Embedding cache hits
# TYPE ragpipe_embed_cache_hits counter
ragpipe_embed_cache_hits 890

# HELP ragpipe_embed_cache_misses Embedding cache misses
# TYPE ragpipe_embed_cache_misses counter
ragpipe_embed_cache_misses 344

# HELP ragpipe_request_duration_seconds Request latency histogram
# TYPE ragpipe_request_duration_seconds histogram
ragpipe_request_duration_seconds_bucket{le="0.1"} 100
ragpipe_request_duration_seconds_bucket{le="0.5"} 500
...
```

## Admin API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/admin/config` | Returns current routing and system prompt configuration |
| `GET` | `/admin/mxr-status` | Returns MXR cache status (cached models, file sizes) |
| `POST` | `/admin/reload-prompt` | Hot-reload system prompt from file/env (requires `RAGPIPE_ADMIN_TOKEN`) |
| `POST` | `/admin/reload-routes` | Hot-reload routes from `RAGPIPE_ROUTES_FILE` (requires `RAGPIPE_ADMIN_TOKEN`) |
| `POST` | `/admin/compile-mxr` | Trigger MXR pre-compilation (non-blocking, returns 202; requires `RAGPIPE_ADMIN_TOKEN`) |

```bash
# Reload system prompt (hot-reload, no restart needed)
curl -X POST http://localhost:8090/admin/reload-prompt \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"

# Reload routes (hot-reload, no restart needed)
curl -X POST http://localhost:8090/admin/reload-routes \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"

# Get current config
curl http://localhost:8090/admin/config \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"

# Get MXR cache status
curl http://localhost:8090/admin/mxr-status \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"

# Trigger MXR pre-compilation (non-blocking)
curl -X POST http://localhost:8090/admin/compile-mxr \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"
```

## rag_metadata

Non-streaming responses include a `rag_metadata` field:

```json
{
    "grounding": "corpus",
    "cited_chunks": [
        {"id": "abc-123:0", "title": "Q3 Red Hat Strategy", "source": "gdrive://filename.pdf"},
        {"id": "abc-123:1", "title": "Q3 Red Hat Strategy", "source": "gdrive://filename.pdf"}
    ],
    "corpus_coverage": "full"
}
```

**`cited_chunks` format (v3+):** Each entry is an object with `id` (doc_id:chunk_index), `title` (extracted title for the source), and `source` (document URI). The flat string format from v2 is no longer used.

```python
# Extract chunk IDs for validation
chunk_ids = [c["id"] for c in response.rag_metadata["cited_chunks"]]
titles = [c["title"] for c in response.rag_metadata["cited_chunks"]]
```

## Documentation

- [Configuration](docs/configuration.md) — Full environment variable reference
- [API](docs/api.md) — Endpoints, `rag_metadata`, streaming behavior, admin endpoints
- [Architecture](docs/architecture.md) — Performance benchmarks and pipeline details
- [Routing](docs/routing.md) — Semantic routing configuration and debugging

## Known issues

- **⚠️ Cold start (~3:53):** First query after ragpipe startup takes ~3:53 while ONNX models are compiled. Warm start (MXR cached): ~6 seconds. Do not restart ragpipe in production unless critical.
- **Streaming citation stripping**: Streaming responses are audited and validated post-hoc (dual-path accumulation), but invalid citations cannot be stripped because the text has already been delivered to the client. Invalid citations are logged as errors. Non-streaming requests strip invalid citations before delivery.
- **LLM phrasing variance**: The negative finding classifier depends on the model using recognizable negation patterns ("no evidence", "not mentioned", etc.) before the `⚠️` marker. When the model phrases its negative finding differently, the response may be classified as `mixed` instead of `general`.
- **Passthrough model list**: `/v1/models` passthrough always returns the global upstream's model list, not the routed model's list.
- **No upstream failover**: If a route's upstream LLM is down, ragpipe returns 502. No automatic fallback to another route.

## Development

```bash
pip install '.[dev]'
python -m pytest tests/ -v    # 164 tests
ruff check && ruff format --check
```

## License

AGPL-3.0-or-later
