# ragpipe

RAG proxy with semantic routing, corpus-preferring grounding, and citation validation. Routes queries to different LLMs and RAG backends based on semantic classification, keeping data classifications separate.

## Architecture
```
POST /v1/chat/completions
  → classify query semantically (cosine similarity, <1ms)
  → select route → per-route LLM, Qdrant, docstore, system prompt
  → embed query (ONNX Runtime, bge-base-en-v1.5, LRU cached)
  → search route's Qdrant (top-K vectors, reference payloads only)
  → hydrate chunk text from Postgres docstore (asyncpg pool, LRU cached)
  → rerank with cross-encoder (ONNX Runtime, MiniLM-L-6-v2)
  → filter chunks below RERANKER_MIN_SCORE (-5 default)
  → inject system prompt + context with [doc_id:chunk_id] labels
  → forward to LLM via persistent httpx client (stream or non-stream)
  → parse citations from response
  → validate against retrieved set + docstore
  → strip invalid citations
  → classify grounding (corpus / general / mixed, with negative finding detection)
  → attach rag_metadata, emit audit log
```

## Package structure
```
ragpipe/
  __init__.py      — public API + __version__
  __main__.py      — python -m ragpipe entry point
  app.py           — FastAPI app, request pipeline, endpoints, admin API
  router.py        — SemanticRouter, RouteConfig, RoutePipeline, YAML config loader
  models.py        — ONNX Runtime Embedder + Reranker wrappers
  grounding.py     — system prompt (hot-reloadable), citation parsing/validation, audit
  reranker.py      — reranker stage with min score filtering
  docstore.py      — Postgres (asyncpg) / SQLite backends + CachedDocstore LRU wrapper
tests/
  test_admin.py      — 4 tests (reload endpoint auth)
  test_router.py     — 14 tests (config parsing, classification, pipeline lifecycle)
  test_grounding.py  — 36 tests (prompt, citations, grounding, negative findings, audit, reload)
  test_docstore.py   — 25 tests (backends + cache layer)
  test_reranker.py   — 12 tests (enabled/disabled/threshold/model swap)
  test_models.py     — 7 tests (embedder + reranker ONNX wrappers)
examples/
  routes-multi-host.yaml — cross-host routing config example
```

## Key design decisions
- Semantic router: cosine similarity on pre-embedded examples, <1ms classification, per-route LLM/Qdrant/docstore/prompt
- ONNX Runtime directly (no fastembed) — 708 MB RSS vs 4.1 GB, 370ms startup
- asyncpg connection pool (2-8 conns) for async hydration, psycopg2 retained for sync ingestion
- LRU chunk cache (2,048 entries) — 55% faster repeated queries, invalidated on upsert/delete
- Persistent httpx client — reuses TCP connections to model, 1-5ms saved per request
- Reranker min score threshold (-5) — filters irrelevant chunks, saves prompt tokens on adversarial queries
- Qdrant stores vectors + reference payloads only — no text
- Full chunk text lives in Postgres (or SQLite for dev)
- Citations are parsed and validated by code, not by the LLM
- Audit log captures grounding decisions without logging text content
- System prompt hot-reloadable via POST /admin/reload-prompt (secured with RAGPIPE_ADMIN_TOKEN)
- Hydration runs as native async (no thread pool hop), embedding/reranking in thread pool
- ONNX Runtime threads capped at 4, CPU memory arenas disabled
- Models downloaded from HuggingFace Hub, cached in RAGPIPE_MODEL_CACHE
- Default embedding model (qdrant/bge-base-en-v1.5-onnx-q) is pre-quantized

## Known issues
- Streaming responses are audited post-hoc (dual-path accumulation) but invalid citations cannot be stripped in-flight — logged as errors instead
- LLM phrasing variance: negative finding classifier depends on recognizable negation patterns before the ⚠️ marker — when the model phrases differently, classification may vary between runs
- /v1/models passthrough returns global upstream's model list, not routed model's
- No upstream failover — route's LLM down → 502, no automatic fallback

## Performance history
| Change | Impact |
|--------|--------|
| Drop fastembed → raw ONNX Runtime | 83% memory reduction (4.1 GB → 708 MB), 5x faster startup, ~17% faster avg query |
| asyncpg connection pool | Native async hydration, frees thread pool worker per request |
| LRU chunk cache (2,048 entries) | 55% faster repeated queries (eliminates Postgres round-trip on cache hit) |
| Persistent httpx client | Saves 1-5ms/request TCP handshake overhead |
| Reranker min score threshold (-5) | Filters irrelevant chunks, adversarial queries get clean empty context |
| Dual-path streaming audit | Streaming responses now audited + validated post-hoc, zero latency impact |
| Negative finding classifier | Citations supporting "X is not mentioned" classified as general, not mixed |
| Semantic router | <1ms query classification, cross-host routing verified |

## Running tests
```bash
pip install '.[dev]'
python -m pytest tests/ -v    # 97 tests
ruff check && ruff format --check
```

## Container image
```bash
podman build -t ragpipe .
# Or pull published: ghcr.io/aclater/ragpipe:main
```

Image is UBI9 Python 3.11, pinned to digest. Models pre-downloaded at build time.
Runs as non-root (USER 1001).
