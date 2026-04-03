# ragpipe

RAG proxy with corpus-preferring grounding and citation validation. Sits between your client and LLM, intercepts OpenAI-compatible chat completions, and enriches every query with retrieval-augmented context from a Qdrant vector database backed by a Postgres document store.

What makes it different: ragpipe doesn't just inject context — it validates the model's citations against what was actually retrieved, strips hallucinated references, classifies the grounding mode (corpus/general/mixed), and emits a text-free audit log for observability.

## How it works

```
Client → ragpipe (:8090) → LLM (:8080)
              │
         1. Embed query (fastembed/ONNX)
         2. Search Qdrant (top-K candidates)
         3. Hydrate chunk text from Postgres
         4. Rerank with cross-encoder
         5. Inject system prompt + context
         6. Forward to model
         7. Parse [doc_id:chunk_id] citations
         8. Validate against retrieved set + docstore
         9. Classify grounding, emit audit log
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
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model (fastembed/ONNX) |
| `EMBED_CACHE_SIZE` | `256` | LRU cache size for query embeddings |
| `ONNX_THREADS` | `4` | ONNX Runtime thread count per model |

### Reranker

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder model (fastembed/ONNX) |
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

## API

Ragpipe is fully OpenAI-compatible. It intercepts `/v1/chat/completions` and passes through everything else unchanged.

**Added to responses:** A `rag_metadata` field with:

```json
{
    "grounding": "corpus",
    "cited_chunks": ["abc-123:0", "abc-123:1"],
    "corpus_coverage": "full"
}
```

**Endpoints:**
- `POST /v1/chat/completions` — RAG-augmented chat (streaming and non-streaming)
- `POST /v1/embeddings` — OpenAI-compatible embeddings via the loaded model
- `GET /health` — health check
- `* /{path}` — passthrough to model

## License

AGPL-3.0-or-later
