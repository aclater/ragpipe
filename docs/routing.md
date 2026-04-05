# Semantic Routing

## What is routing

ragpipe classifies incoming queries semantically using cosine similarity against pre-embedded examples. Each query is matched to the best-fitting route, which determines:

- Which LLM to forward to (`MODEL_URL` per route)
- Which Qdrant collection to search
- Which document store to hydrate from
- Which system prompt to inject

This keeps data sources isolated across routing domains — a personnel query never sees the same context as a legal lookup query.

## Configuration

Set `RAGPIPE_ROUTES_FILE` to a YAML config file path:

```bash
export RAGPIPE_ROUTES_FILE=/path/to/routes.yaml
```

When unset, ragpipe operates as a single-pipeline proxy (no routing).

See the [multi-host example config](../examples/routes-multi-host.yaml) for a complete routing configuration.

## Debugging routes

Use the `/admin/classify` endpoint to test which route a query would match without sending a real chat completion:

```bash
curl -X POST http://localhost:8090/admin/classify \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the NDAA say about AI?"}'
```

This returns the matched route, its score, and scores for all routes — essential for tuning your route examples. See the [API docs](api.md#admin-classify-query) for the full response format.
