# ragpipe

RAG proxy with corpus-preferring grounding and citation validation. Intercepts OpenAI-compatible chat completions, performs retrieval + reranking + context injection, then post-processes with citation validation and grounding classification.

## Architecture
```
POST /v1/chat/completions
  → embed query (ONNX Runtime, bge-base-en-v1.5, LRU cached)
  → search Qdrant (top-K vectors, reference payloads only)
  → hydrate chunk text from Postgres docstore
  → rerank with cross-encoder (ONNX Runtime, MiniLM-L-6-v2)
  → inject system prompt + context with [doc_id:chunk_id] labels
  → forward to LLM (stream or non-stream)
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
  app.py           — FastAPI app, request pipeline, endpoints
  models.py        — ONNX Runtime Embedder + Reranker (replaced fastembed)
  grounding.py     — system prompt, citation parsing/validation, audit
  reranker.py      — reranker stage wrapping models.Reranker
  docstore.py      — Postgres/SQLite document store
tests/
  test_grounding.py  — 30 tests
  test_docstore.py   — 19 tests
  test_reranker.py   — 9 tests
  test_models.py     — 7 tests (embedder + reranker ONNX wrappers)
```

## Key design decisions
- ONNX Runtime directly (no fastembed) — 708 MB RSS vs 4.1 GB, 370ms startup
- Qdrant stores vectors + reference payloads only — no text
- Full chunk text lives in Postgres (or SQLite for dev)
- Citations are parsed and validated by code, not by the LLM
- Audit log captures grounding decisions without logging text content
- System prompt is overridable via RAGPIPE_SYSTEM_PROMPT_FILE or RAGPIPE_SYSTEM_PROMPT
- All blocking I/O runs in a 4-worker thread pool
- ONNX Runtime threads capped at 4, CPU memory arenas disabled
- Models downloaded from HuggingFace Hub, cached in RAGPIPE_MODEL_CACHE

## Running tests
```bash
pip install '.[dev]'
python -m pytest tests/ -v
ruff check && ruff format --check
```

## Container image
```bash
podman build -t ragpipe .
# Or pull published: ghcr.io/aclater/ragpipe:main
```

Image is UBI9 Python 3.11, pinned to digest. Models pre-downloaded at build time.
Runs as non-root (USER 1001).
