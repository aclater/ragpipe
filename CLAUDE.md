# ragpipe

RAG proxy with corpus-preferring grounding and citation validation. Intercepts OpenAI-compatible chat completions, performs retrieval + reranking + context injection, then post-processes with citation validation and grounding classification.

## Architecture
```
POST /v1/chat/completions
  → embed query (ONNX Runtime, bge-base-en-v1.5, LRU cached)
  → search Qdrant (top-K vectors, reference payloads only)
  → hydrate chunk text from Postgres docstore (asyncpg pool, LRU cached)
  → rerank with cross-encoder (ONNX Runtime, MiniLM-L-6-v2)
  → filter chunks below RERANKER_MIN_SCORE (-5 default)
  → inject system prompt + context with [doc_id:chunk_id] labels
  → forward to LLM via persistent httpx client (stream or non-stream)
  → parse citations from response
  → validate against retrieved set + docstore
  → strip invalid citations
  → classify grounding (corpus / general / mixed)
  → attach rag_metadata, emit audit log
```

## Package structure
```
ragpipe/
  __init__.py      — public API + __version__
  __main__.py      — python -m ragpipe entry point
  app.py           — FastAPI app, request pipeline, endpoints, admin API
  models.py        — ONNX Runtime Embedder + Reranker wrappers
  grounding.py     — system prompt (hot-reloadable), citation parsing/validation, audit
  reranker.py      — reranker stage with min score filtering
  docstore.py      — Postgres (asyncpg) / SQLite backends + CachedDocstore LRU wrapper
tests/
  test_admin.py      — 4 tests (reload endpoint auth)
  test_grounding.py  — 33 tests (prompt, citations, grounding, audit, reload)
  test_docstore.py   — 25 tests (backends + cache layer)
  test_reranker.py   — 12 tests (enabled/disabled/threshold/model swap)
  test_models.py     — 7 tests (embedder + reranker ONNX wrappers)
```

## Key design decisions
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
- Streaming responses (Open WebUI) bypass citation validation and audit logging
- Adversarial queries can produce "mixed" grounding when model cites docs to support a negative finding — prompt tuning issue
- sovereign_ai query regressed with min score threshold — only 2 chunks pass at -3.02, insufficient for grounding

## Performance history
| Change | Impact |
|--------|--------|
| Drop fastembed → raw ONNX Runtime | 83% memory reduction (4.1 GB → 708 MB), 5x faster startup, ~17% faster avg query |
| asyncpg connection pool | Native async hydration, frees thread pool worker per request |
| LRU chunk cache (2,048 entries) | 55% faster repeated queries (eliminates Postgres round-trip on cache hit) |
| Persistent httpx client | Saves 1-5ms/request TCP handshake overhead |
| Reranker min score threshold (-5) | Filters irrelevant chunks, adversarial queries get clean empty context |

## Running tests
```bash
pip install '.[dev]'
python -m pytest tests/ -v    # 80 tests
ruff check && ruff format --check
```

## Container image
```bash
podman build -t ragpipe .
# Or pull published: ghcr.io/aclater/ragpipe:main
```

Image is UBI9 Python 3.11, pinned to digest. Models pre-downloaded at build time.
Runs as non-root (USER 1001).
