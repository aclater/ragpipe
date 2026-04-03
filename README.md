# ragpipe

RAG proxy with semantic routing, corpus-preferring grounding, and citation validation. Sits between your client and multiple LLMs, intercepts OpenAI-compatible chat completions, and enriches queries with retrieval-augmented context from Qdrant vector databases backed by Postgres document stores.

What makes it different: ragpipe classifies queries semantically and routes them to the right LLM and RAG backend — keeping data classifications separate across security domains. It validates the model's citations against what was actually retrieved, strips hallucinated references, classifies the grounding mode (corpus/general/mixed), and emits a text-free audit log for observability.

### Single-pipeline request flow
![Request flow](architecture.svg)

### Multi-host semantic routing
![Routing](architecture-routing.svg)

### Package components
![Components](architecture-components.svg)

## How it works

1. **Classify** the query semantically (cosine similarity, <1ms) and select a route
2. **Embed** the query (ONNX Runtime, bge-base-en-v1.5, LRU cached)
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

All configuration is via environment variables with sensible defaults.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | `http://127.0.0.1:8080` | LLM endpoint to forward to |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant vector search endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `RAG_PROXY_PORT` | `8090` | Port to listen on |
| `RAG_TOP_K` | `20` | Qdrant candidate count before reranking |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `qdrant/bge-base-en-v1.5-onnx-q` | HuggingFace repo for ONNX embedding model (pre-quantized) |
| `EMBED_CACHE_SIZE` | `256` | LRU cache size for query embeddings |
| `ONNX_THREADS` | `4` | ONNX Runtime intra-op thread count per model |
| `RAGPIPE_MODEL_CACHE` | `~/.cache/ragpipe` | Local directory for downloaded ONNX models |

### Reranker

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | HuggingFace repo for ONNX cross-encoder model |
| `RERANKER_TOP_N` | `5` | Max results to keep after reranking |
| `RERANKER_MIN_SCORE` | `-5` | Minimum reranker score — chunks below this are filtered out. Set to `-999` to disable |

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
| `RAGPIPE_SYSTEM_PROMPT_FILE` | — | Path to custom system prompt file (takes precedence) |
| `RAGPIPE_SYSTEM_PROMPT` | — | Inline custom system prompt text |
| `THINKING_BUDGET` | `1024` | Token budget for model reasoning |

If neither prompt variable is set, ragpipe uses a built-in corpus-preferring grounding prompt that instructs the model to cite documents as `[doc_id:chunk_id]` and prefix general knowledge with `⚠️ Not in corpus:`.

### Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_ROUTES_FILE` | — | Path to YAML routes config. When unset, single-pipeline mode |

See [`examples/routes-multi-host.yaml`](examples/routes-multi-host.yaml) for a multi-host routing config.

### Admin

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_ADMIN_TOKEN` | — | Bearer token for admin endpoints. Admin endpoints are disabled when unset |

## Performance

Benchmarked against the legacy fastembed-based implementation with identical queries on the same corpus (4,831 documents):

| Metric | Legacy (fastembed + psycopg2) | ragpipe |
|--------|------|-------|
| Memory (RSS) | 4,100 MB | 708 MB |
| Startup time | ~2s | ~370ms |
| Embed latency | ~10ms | ~9ms |
| Rerank (20 docs) | ~6ms | ~6ms |
| Threads | 130 | 70 |
| Repeated query (e2e) | ~9.3s | ~4.2s (cache hit) |
| Cold query avg (16 queries) | 26.5s | 22.1s |

### Optimization history

| Change | Impact |
|--------|--------|
| Drop fastembed, use ONNX Runtime directly | 83% memory reduction (4.1 GB → 708 MB), 5x faster startup, ~17% faster avg query |
| asyncpg connection pool for hydration | Native async, frees thread pool worker per request |
| LRU chunk cache (2,048 entries) | 55% faster repeated queries (eliminates Postgres round-trip on cache hit) |
| Persistent httpx client | Saves 1-5ms/request TCP handshake overhead |
| Reranker min score threshold (-5) | Filters irrelevant chunks on adversarial/off-topic queries, saves prompt tokens |
| Dual-path streaming audit | Streaming responses audited + validated post-hoc with zero latency impact |
| Negative finding classifier | Citations supporting "X is not mentioned" correctly classified as general, not mixed |
| Semantic router | <1ms query classification, per-route LLM/Qdrant/docstore/prompt, cross-host forwarding |

## API

Ragpipe is fully OpenAI-compatible. It intercepts `/v1/chat/completions` and passes through everything else unchanged.

**Added to non-streaming responses:** A `rag_metadata` field:

```json
{
    "grounding": "corpus",
    "cited_chunks": ["abc-123:0", "abc-123:1"],
    "corpus_coverage": "full"
}
```

**Streaming responses** include a performance summary block before `[DONE]` with token counts, generation speed, and RAG source info.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | RAG-augmented chat (streaming and non-streaming) |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings via the loaded model |
| `GET` | `/health` | Health check |
| `POST` | `/admin/reload-prompt` | Hot-reload system prompt from file/env (requires `RAGPIPE_ADMIN_TOKEN`) |
| `POST` | `/admin/classify` | Test route classification without sending a chat completion (requires `RAGPIPE_ADMIN_TOKEN`) |
| `*` | `/{path}` | Passthrough to model |

### Admin: reload prompt

Hot-reload the system prompt without restarting. Preserves chunk cache, connection pool, and ONNX model state.

```bash
curl -X POST http://localhost:8090/admin/reload-prompt \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"
```

Response:

```json
{
    "status": "reloaded",
    "changed": true,
    "hash": "a1b2c3d4...",
    "source": "file:/path/to/prompt.txt"
}
```

### Admin: classify query

Test which route a query would be sent to without sending a real chat completion. Essential for tuning route examples.

```bash
curl -X POST http://localhost:8090/admin/classify \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the NDAA say about AI?"}'
```

Response:

```json
{
    "route": "lookup",
    "score": 0.8189,
    "all_routes": {
        "analysis": 0.6245,
        "personnel": 0.4881,
        "lookup": 0.8189,
        "general": 0.4268
    }
}
```

## Known issues

- **Streaming citation stripping**: Streaming responses are audited and validated post-hoc (dual-path accumulation), but invalid citations cannot be stripped because the text has already been delivered to the client. Invalid citations are logged as errors. Non-streaming requests strip invalid citations before delivery.
- **LLM phrasing variance**: The negative finding classifier depends on the model using recognizable negation patterns ("no evidence", "not mentioned", etc.) before the `⚠️` marker. When the model phrases its negative finding differently, the response may be classified as `mixed` instead of `general`. The adversarial tuning agent will address prompt compliance consistency.
- **Passthrough model list**: `/v1/models` passthrough always returns the global upstream's model list, not the routed model's list.
- **No upstream failover**: If a route's upstream LLM is down, ragpipe returns 502. No automatic fallback to another route.

## Development

```bash
pip install '.[dev]'
python -m pytest tests/ -v    # 97 tests
ruff check && ruff format --check
```

## License

AGPL-3.0-or-later
