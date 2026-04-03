# ragpipe

RAG proxy with corpus-preferring grounding and citation validation. Intercepts OpenAI-compatible chat completions, performs retrieval + reranking + context injection, then post-processes with citation validation and grounding classification.

## Architecture
```
POST /v1/chat/completions
  → embed query (fastembed/ONNX, CPU)
  → search Qdrant (top-K vectors, reference payloads only)
  → hydrate chunk text from Postgres docstore
  → rerank with cross-encoder (fastembed/ONNX, CPU)
  → inject system prompt + context with [doc_id:chunk_id] labels
  → forward to model
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
  grounding.py     — system prompt, citation parsing/validation, audit
  reranker.py      — fastembed cross-encoder wrapper
  docstore.py      — Postgres/SQLite document store
tests/
  test_grounding.py  — 30 tests
  test_docstore.py   — 19 tests
  test_reranker.py   — 9 tests
```

## Key design decisions
- Qdrant stores vectors + reference payloads only — no text
- Full chunk text lives in Postgres (or SQLite for dev)
- Citations are parsed and validated by code, not by the LLM
- Audit log captures grounding decisions without logging text content
- System prompt is overridable via RAGPIPE_SYSTEM_PROMPT_FILE or RAGPIPE_SYSTEM_PROMPT
- All blocking I/O runs in a 4-worker thread pool
- ONNX Runtime threads capped at 4 to prevent memory bloat

## Running tests
```bash
pip install '.[dev]'
python -m pytest tests/ -v
ruff check && ruff format --check
```
