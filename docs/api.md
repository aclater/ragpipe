# API

Ragpipe is fully OpenAI-compatible. It intercepts `/v1/chat/completions` and passes through everything else unchanged.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | RAG-augmented chat (streaming and non-streaming) |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings via the loaded model |
| `GET` | `/health` | Health check |
| `POST` | `/admin/reload-prompt` | Hot-reload system prompt from file/env (requires `RAGPIPE_ADMIN_TOKEN`) |
| `POST` | `/admin/classify` | Test route classification without sending a chat completion (requires `RAGPIPE_ADMIN_TOKEN`) |
| `*` | `/{path}` | Passthrough to model |

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

> **⚠️ Breaking change (v3):** `cited_chunks` changed from a flat list of `"doc_id:chunk_id"` strings to a list of objects with `id`, `title`, and `source` fields.
>
> **Migration required:** Update consumers to parse `cited_chunks` as objects.
> ```python
> # Before: cited_chunks = ["abc-123:0", "abc-123:1"]
> # After:  cited_chunks = [{"id": "abc-123:0", "title": "...", "source": "..."}, ...]
> chunk_ids = [c["id"] for c in cited_chunks]
> ```

## Streaming behavior

Streaming responses include a performance summary block before `[DONE]` with token counts, generation speed, and RAG source info. Invalid citations in streaming responses cannot be stripped (the text has already been delivered) but are logged as errors. See [Known issues](../README.md#known-issues) for details.

## Admin: reload prompt

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

## Admin: classify query

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
