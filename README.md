# ragpipe

RAG proxy with corpus-preferring grounding and citation validation. Sits between your client and LLM, intercepts OpenAI-compatible chat completions, and enriches every query with retrieval-augmented context from a Qdrant vector database backed by a Postgres document store.

What makes it different: ragpipe doesn't just inject context — it validates the model's citations against what was actually retrieved, strips hallucinated references, classifies the grounding mode (corpus/general/mixed), and emits a text-free audit log for observability.

![Architecture](architecture.svg)

## How it works

1. **Embed** the user's query (ONNX Runtime, bge-base-en-v1.5, LRU cached)
2. **Search** Qdrant for top-K candidate vectors (reference payloads only)
3. **Hydrate** chunk text from Postgres document store
4. **Rerank** with cross-encoder (ONNX Runtime, MiniLM-L-6-v2)
5. **Inject** system prompt + context with `[doc_id:chunk_id]` labels
6. **Forward** to LLM (streaming or non-streaming)
7. **Post-process**: parse citations, validate against retrieved set + docstore, strip invalid, classify grounding, attach `rag_metadata`, emit audit log

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
| `EMBED_MODEL` | `qdrant/bge-base-en-v1.5-onnx-q` | HuggingFace repo for ONNX embedding model |
| `EMBED_CACHE_SIZE` | `256` | LRU cache size for query embeddings |
| `ONNX_THREADS` | `4` | ONNX Runtime intra-op thread count per model |
| `RAGPIPE_MODEL_CACHE` | `~/.cache/ragpipe` | Local directory for downloaded ONNX models |

### Reranker

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | HuggingFace repo for ONNX cross-encoder model |
| `RERANKER_TOP_N` | `5` | Results to keep after reranking |

### Document store

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCSTORE_BACKEND` | `postgres` | `postgres` or `sqlite` |
| `DOCSTORE_URL` | *(required for postgres)* | Postgres connection string |
| `DOCSTORE_SQLITE_PATH` | `/tmp/docstore.db` | SQLite file path |

### Grounding

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_SYSTEM_PROMPT_FILE` | — | Path to custom system prompt file (takes precedence) |
| `RAGPIPE_SYSTEM_PROMPT` | — | Inline custom system prompt text |
| `THINKING_BUDGET` | `1024` | Token budget for model reasoning |

If neither prompt variable is set, ragpipe uses a built-in corpus-preferring grounding prompt that instructs the model to cite documents as `[doc_id:chunk_id]` and prefix general knowledge with `⚠️ Not in corpus:`.

## Performance

Benchmarked against the legacy fastembed-based implementation with identical queries on the same corpus (4,831 documents):

| Metric | Legacy (fastembed + psycopg2) | ragpipe (ONNX RT + asyncpg + cache) |
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
| Drop fastembed → raw ONNX Runtime | 83% memory reduction (4.1 GB → 708 MB), 5x faster startup, ~17% faster avg query |
| asyncpg connection pool | Native async hydration, frees thread pool worker per request |
| LRU chunk cache (2,048 entries) | 55% faster repeated queries (eliminates Postgres round-trip on cache hit) |

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

**Endpoints:**
- `POST /v1/chat/completions` — RAG-augmented chat (streaming and non-streaming)
- `POST /v1/embeddings` — OpenAI-compatible embeddings via the loaded model
- `GET /health` — health check
- `* /{path}` — passthrough to model

## Development

```bash
pip install '.[dev]'
python -m pytest tests/ -v    # 64 tests
ruff check && ruff format --check
```

## License

AGPL-3.0-or-later
