# Configuration

All configuration is via environment variables with sensible defaults.

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_URL` | `http://127.0.0.1:8080` | LLM endpoint to forward to |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant vector search endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `RAG_PROXY_PORT` | `8090` | Port to listen on |
| `RAG_TOP_K` | `20` | Qdrant candidate count before reranking |

## Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `Alibaba-NLP/gte-modernbert-base` | HuggingFace repo for ONNX embedding model (quantized, 768d) |
| `ONNX_PAD_LENGTH` | `128` | Fixed padding length for tokenizer — prevents MIGraphX recompilation per input shape |
| `EMBED_CACHE_SIZE` | `256` | LRU cache size for query embeddings |
| `ONNX_THREADS` | `4` | ONNX Runtime intra-op thread count per model |
| `RAGPIPE_MODEL_CACHE` | `~/.cache/ragpipe` | Local directory for downloaded ONNX models |

## Reranker

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable cross-encoder reranking |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | HuggingFace repo for ONNX cross-encoder model |
| `RERANKER_TOP_N` | `5` | Max results to keep after reranking |
| `RERANKER_MIN_SCORE` | `-5` | Minimum reranker score — chunks below this are filtered out. Set to `-999` to disable |

## Document store

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCSTORE_BACKEND` | `postgres` | `postgres` or `sqlite` |
| `DOCSTORE_URL` | *(required for postgres)* | Postgres connection string |
| `DOCSTORE_SQLITE_PATH` | `/tmp/docstore.db` | SQLite file path |
| `CHUNK_CACHE_SIZE` | `2048` | LRU cache entries for hydrated chunk text |

## Grounding

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_SYSTEM_PROMPT_FILE` | — | Path to custom system prompt file (takes precedence) |
| `RAGPIPE_SYSTEM_PROMPT` | — | Inline custom system prompt text |
| `THINKING_BUDGET` | `1024` | Token budget for model reasoning |

If neither prompt variable is set, ragpipe uses a built-in corpus-preferring grounding prompt that instructs the model to cite documents as `[doc_id:chunk_id]` and prefix general knowledge with `⚠️ Not in corpus:`.

## Routing

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_ROUTES_FILE` | — | Path to YAML routes config. When unset, single-pipeline mode |

See [Routing](routing.md) for details and the [multi-host example config](../examples/routes-multi-host.yaml).

## Admin

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGPIPE_ADMIN_TOKEN` | — | Bearer token for admin endpoints. Admin endpoints are disabled when unset |
