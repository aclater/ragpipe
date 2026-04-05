# Architecture

## Request pipeline

When a request arrives at `/v1/chat/completions`, ragpipe executes these stages:

1. **Classify** the query semantically (cosine similarity, <1ms) and select a route
2. **Embed** the query (ONNX Runtime, gte-modernbert-base, LRU cached)
3. **Search** the route's Qdrant collection for top-K candidate vectors
4. **Hydrate** chunk text from the route's Postgres document store (async, cached)
5. **Rerank** with cross-encoder (ONNX Runtime, MiniLM-L-6-v2), filter below min score
6. **Inject** the route's system prompt + context with `[doc_id:chunk_id]` labels
7. **Forward** to the route's LLM (streaming or non-streaming)
8. **Post-process**: parse citations, validate, classify grounding, attach `rag_metadata`, emit audit log

Without a routes config (`RAGPIPE_ROUTES_FILE` unset), ragpipe operates as a single-pipeline proxy — fully backward compatible.

### Single-pipeline request flow
![Request flow](../architecture.svg)

### Multi-host semantic routing
![Routing](../architecture-routing.svg)

### Package components
![Components](../architecture-components.svg)

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
