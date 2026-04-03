"""Ragpipe — RAG proxy with corpus-preferring grounding and citation validation.

Intercepts OpenAI-compatible chat completions, searches Qdrant for relevant
document chunks, hydrates text from the document store, reranks with a
cross-encoder, injects grounded context, and post-processes responses with
citation validation and grounding classification.
"""

import asyncio
import functools
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from ragpipe.grounding import (
    build_metadata,
    build_system_message,
    determine_corpus_coverage,
    format_context,
    log_audit,
    parse_citations,
    query_hash,
    strip_invalid_citations,
    validate_citations,
)
from ragpipe.reranker import rerank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ragpipe")

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:8080")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "documents")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
TOP_K = int(os.environ.get("RAG_TOP_K", "20"))
PROXY_PORT = int(os.environ.get("RAG_PROXY_PORT", "8090"))

# ONNX Runtime thread count — limits per-session thread arenas.
# Default 0 = all logical cores, which creates excessive per-thread
# arena overhead. 4 threads is sufficient for a 4-worker thread pool.
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))

# Thinking budget — allows the model to reason across retrieved chunks
# and general knowledge without unconstrained latency
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "1024"))

# ── Globals initialized at startup ───────────────────────────────────────────

qdrant: QdrantClient = None
embedder: TextEmbedding = None
docstore = None
_collection_exists: bool = False

# Thread pool for blocking I/O (embedder.encode, docstore, reranker)
_executor = ThreadPoolExecutor(max_workers=4)

# Embedding cache — avoids re-encoding repeated queries
EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "256"))

app = FastAPI()


@app.on_event("startup")
def startup():
    global qdrant, embedder, docstore, _collection_exists
    log.info("Connecting to Qdrant at %s", QDRANT_URL)
    qdrant = QdrantClient(url=QDRANT_URL, timeout=10)

    collections = [c.name for c in qdrant.get_collections().collections]
    _collection_exists = COLLECTION_NAME in collections
    if not _collection_exists:
        log.warning(
            "Collection '%s' not found in Qdrant — RAG context will be empty until documents are ingested",
            COLLECTION_NAME,
        )

    # fastembed uses ONNX Runtime (CPU-only) — no PyTorch, no GPU segfault risk
    log.info("Loading embedding model: %s (fastembed/ONNX, threads=%d)", EMBED_MODEL, ONNX_THREADS)
    embedder = TextEmbedding(EMBED_MODEL, threads=ONNX_THREADS)

    from ragpipe.docstore import create_docstore

    docstore = create_docstore()

    # Warm up the reranker so first request isn't slow
    try:
        from ragpipe.reranker import warm_up

        warm_up()
    except Exception:
        log.warning("Reranker warm-up failed — will load on first request")

    log.info(
        "Ragpipe ready — forwarding to %s (thinking_budget=%d, embed_cache=%d)",
        MODEL_URL,
        THINKING_BUDGET,
        EMBED_CACHE_SIZE,
    )


@app.on_event("shutdown")
def shutdown():
    if qdrant:
        qdrant.close()
    log.info("Ragpipe shut down")


# ── Retrieval pipeline ───────────────────────────────────────────────────────


@functools.lru_cache(maxsize=EMBED_CACHE_SIZE)
def _embed_query(query: str) -> tuple:
    """Embed a query string, caching the result. Returns tuple for hashability."""
    vectors = list(embedder.embed([query]))
    return tuple(vectors[0].tolist())


def _check_collection() -> bool:
    """Check if the Qdrant collection exists, with caching."""
    global _collection_exists
    if _collection_exists:
        return True
    # Re-check — ingestion may have created it since startup
    collections = [c.name for c in qdrant.get_collections().collections]
    _collection_exists = COLLECTION_NAME in collections
    return _collection_exists


def _search_qdrant_sync(query: str) -> list[dict]:
    """Synchronous Qdrant search — runs in thread pool."""
    try:
        if not _check_collection():
            return []

        query_vector = list(_embed_query(query))
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
        )

        if not results.points:
            return []

        return [point.payload for point in results.points if point.payload.get("doc_id")]
    except Exception:
        log.exception("Qdrant search failed")
        return []


def _hydrate_sync(refs: list[dict]) -> list[dict]:
    """Synchronous docstore hydration — runs in thread pool."""
    if not refs:
        return []

    lookup_keys = [(r["doc_id"], r["chunk_id"]) for r in refs]
    texts = docstore.get_chunks(lookup_keys)

    hydrated = []
    for ref in refs:
        key = (ref["doc_id"], ref["chunk_id"])
        text = texts.get(key)
        if text is None:
            log.warning(
                "Orphaned vector: doc_id=%s chunk_id=%d — excluding from results",
                ref["doc_id"],
                ref["chunk_id"],
            )
            continue
        hydrated.append(
            {
                "text": text,
                "source": ref.get("source", "unknown"),
                "doc_id": ref["doc_id"],
                "chunk_id": ref["chunk_id"],
            }
        )

    return hydrated


def _rerank_sync(query: str, candidates: list[dict]) -> list[dict]:
    """Synchronous reranking — runs in thread pool."""
    return rerank(query, candidates)


async def retrieve_and_rerank(user_query: str) -> tuple[list[dict], list[dict]]:
    """Full async retrieval pipeline: Qdrant → hydrate → rerank.

    All blocking operations (embedding, Qdrant I/O, docstore SQL, reranker
    inference) run in a thread pool to avoid blocking the FastAPI event loop.

    Returns (ranked_chunks, all_retrieved_refs) where all_retrieved_refs
    is the full set of hydrated results before reranking, needed for
    citation validation.
    """
    loop = asyncio.get_event_loop()

    refs = await loop.run_in_executor(_executor, _search_qdrant_sync, user_query)
    candidates = await loop.run_in_executor(_executor, _hydrate_sync, refs)
    ranked = await loop.run_in_executor(_executor, _rerank_sync, user_query, candidates)
    return ranked, candidates


# ── Request processing ───────────────────────────────────────────────────────


async def process_chat_request(body: dict) -> tuple[dict, dict]:
    """Process a chat completion request with grounding.

    Returns (modified_body, retrieval_context) where retrieval_context
    contains everything needed for post-response citation validation.
    """
    messages = body.get("messages", [])
    if not messages:
        return body, {"ranked": [], "retrieved_set": set(), "corpus_coverage": "none", "user_query": ""}

    # Find the last user message for the query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        return body, {"ranked": [], "retrieved_set": set(), "corpus_coverage": "none", "user_query": ""}

    # Retrieve and rerank (async — blocking I/O runs in thread pool)
    ranked, all_candidates = await retrieve_and_rerank(user_query)
    corpus_coverage = determine_corpus_coverage(ranked)

    # Build the retrieved set for citation validation — includes all
    # candidates that were retrieved, not just the reranked top-N,
    # because the model sees only the top-N but we validate against
    # the full retrieved set to catch hallucinated citations
    retrieved_set = {(c["doc_id"], c["chunk_id"]) for c in all_candidates}

    # Format context with citation-friendly labels
    context = format_context(ranked, docstore=docstore)
    system_content = build_system_message(context)

    if corpus_coverage == "none":
        # Log empty retrieval — useful signal for corpus gap analysis
        q_hash = query_hash(user_query)
        log.info("Empty retrieval for query_hash=%s — proceeding with general knowledge", q_hash)

    # Inject system message into the conversation
    # Insert after any existing system messages but before user messages
    context_msg = {"role": "system", "content": system_content}
    new_messages = []
    inserted = False
    for msg in messages:
        if not inserted and msg.get("role") != "system":
            new_messages.append(context_msg)
            inserted = True
        new_messages.append(msg)
    if not inserted:
        new_messages.append(context_msg)

    body["messages"] = new_messages

    # Disable thinking — reasoning_content is not handled by Open WebUI
    # streaming and wastes tokens. The RAG context provides the grounding
    # the model needs without chain-of-thought.
    body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if "max_tokens" not in body and "max_completion_tokens" not in body:
        body["max_completion_tokens"] = 4096

    return body, {
        "ranked": ranked,
        "retrieved_set": retrieved_set,
        "corpus_coverage": corpus_coverage,
        "user_query": user_query,
    }


def process_response(response_data: dict, ctx: dict) -> dict:
    """Post-process the LLM response: validate citations, add metadata, audit.

    This is where we parse the model's output, check that cited chunks are
    real and were in the retrieved set, classify the grounding mode, and
    emit the audit log. Crucially, this parsing is done by code — we never
    ask the LLM to generate the metadata.
    """
    choices = response_data.get("choices", [])
    if not choices:
        return response_data

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return response_data

    user_query = ctx.get("user_query", "")
    ranked = ctx.get("ranked", [])
    retrieved_set = ctx.get("retrieved_set", set())
    corpus_coverage = ctx.get("corpus_coverage", "none")

    # Parse citations from the model's response
    citations = parse_citations(content)

    # Validate each citation against the retrieved set and docstore
    valid_citations, validation_errors = validate_citations(citations, retrieved_set, docstore)

    # Determine citation validation status
    if not citations:
        citation_status = "pass"  # No citations to validate
    elif validation_errors:
        citation_status = "stripped"
        # Log each invalid citation as an error with the query hash
        q_hash = query_hash(user_query)
        for err in validation_errors:
            log.error(
                "Invalid citation: query_hash=%s doc_id=%s chunk_id=%d reason=%s",
                q_hash,
                err["doc_id"],
                err["chunk_id"],
                err["reason"],
            )
        # Strip invalid citations from the response — preserve the rest
        content = strip_invalid_citations(content, validation_errors)
        response_data["choices"][0]["message"]["content"] = content
    else:
        citation_status = "pass"

    # Build metadata — populated by parsing, not by the LLM
    metadata = build_metadata(content, valid_citations, corpus_coverage)

    # Attach metadata to the response
    response_data["rag_metadata"] = metadata

    # Audit log — never logs text content
    log_audit(
        q_hash=query_hash(user_query),
        retrieved_chunks=ranked,
        ranked_chunks=ranked,
        corpus_coverage=corpus_coverage,
        grounding=metadata["grounding"],
        valid_citations=valid_citations,
        citation_validation=citation_status,
    )

    return response_data


def _format_perf_summary(timings: dict | None, usage: dict | None, ctx: dict) -> str:
    """Format a performance summary block appended to streaming responses."""
    parts = ["\n\n---\n📊 **Performance**\n"]
    if usage:
        parts.append(f"- Prompt: {usage.get('prompt_tokens', '?')} tokens")
        parts.append(f"- Completion: {usage.get('completion_tokens', '?')} tokens")
        parts.append(f"- Total: {usage.get('total_tokens', '?')} tokens")
    if timings:
        prompt_ms = timings.get("prompt_ms", 0)
        predicted_ms = timings.get("predicted_ms", 0)
        predicted_tps = timings.get("predicted_per_second", 0)
        prompt_tps = timings.get("prompt_per_second", 0)
        parts.append(f"- Prompt: {prompt_ms / 1000:.1f}s ({prompt_tps:.0f} t/s)")
        parts.append(f"- Generation: {predicted_ms / 1000:.1f}s ({predicted_tps:.0f} t/s)")
    ranked = ctx.get("ranked", [])
    if ranked:
        sources = sorted({r.get("source", "?") for r in ranked})
        parts.append(f"- RAG: {len(ranked)} chunks from {len(sources)} source(s)")
    return "\n".join(parts) + "\n"


# ── HTTP endpoints ───────────────────────────────────────────────────────────


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """Intercept chat completions, inject grounded RAG context, forward to model."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    body, retrieval_ctx = await process_chat_request(body)

    stream = body.get("stream", False)

    if stream:
        # Stream chunks through, capture the final chunk's timings/usage,
        # and append a performance summary before [DONE].
        async def stream_response():
            timings = None
            usage = None
            last_chunk_id = None
            async with (
                httpx.AsyncClient(timeout=300) as client,
                client.stream(
                    "POST",
                    f"{MODEL_URL}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as resp,
            ):
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data: "):
                        yield raw + "\n"
                        continue
                    payload = raw[6:]
                    if payload.strip() == "[DONE]":
                        # Inject performance summary before [DONE]
                        if timings or usage:
                            summary = _format_perf_summary(timings, usage, retrieval_ctx)
                            summary_chunk = {
                                "choices": [{"finish_reason": None, "index": 0, "delta": {"content": summary}}],
                                "id": last_chunk_id or "",
                                "object": "chat.completion.chunk",
                            }
                            yield f"data: {json.dumps(summary_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        continue
                    try:
                        chunk_data = json.loads(payload)
                        if "timings" in chunk_data:
                            timings = chunk_data["timings"]
                        if "usage" in chunk_data:
                            usage = chunk_data["usage"]
                        last_chunk_id = chunk_data.get("id", last_chunk_id)
                    except json.JSONDecodeError:
                        pass
                    yield raw + "\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{MODEL_URL}/v1/chat/completions",
                json=body,
            )
        try:
            resp.raise_for_status()
            response_data = resp.json()
        except httpx.HTTPStatusError as e:
            log.error("Model returned %d: %s", e.response.status_code, e.response.text[:200])
            return JSONResponse({"error": str(e)}, status_code=e.response.status_code)
        # Post-process: validate citations, classify grounding, audit
        response_data = process_response(response_data, retrieval_ctx)
        return JSONResponse(content=response_data)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.api_route("/v1/embeddings", methods=["POST"])
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    texts = body.get("input", [])
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return JSONResponse({"error": "No input provided"}, status_code=400)

    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(_executor, lambda: [v.tolist() for v in embedder.embed(texts)])

    return JSONResponse(
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
            "model": EMBED_MODEL,
        }
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests to the model unchanged."""
    async with httpx.AsyncClient(timeout=300) as client:
        body = await request.body()
        resp = await client.request(
            method=request.method,
            url=f"{MODEL_URL}/{path}",
            content=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        )
        try:
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            log.error("Passthrough %s returned %d", path, e.response.status_code)
            return JSONResponse({"error": str(e)}, status_code=e.response.status_code)


def main():
    """Entry point for the ragpipe CLI command."""
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)


if __name__ == "__main__":
    main()
