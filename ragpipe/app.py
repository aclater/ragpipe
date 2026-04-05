"""Ragpipe — RAG proxy with corpus-preferring grounding and citation validation.

Intercepts OpenAI-compatible chat completions, searches Qdrant for relevant
document chunks, hydrates text from the document store, reranks with a
cross-encoder, injects grounded context, and post-processes responses with
citation validation and grounding classification.
"""

import asyncio
import collections
import functools
import hashlib
import json
import logging
import os
import sys
import threading
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
from ragpipe.models import Embedder
from ragpipe.reranker import rerank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ragpipe")

# ── Configuration ────────────────────────────────────────────────────────────

ADMIN_TOKEN = os.environ.get("RAGPIPE_ADMIN_TOKEN", "")
ROUTES_FILE = os.environ.get("RAGPIPE_ROUTES_FILE")
MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:8080")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "documents")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base")
TOP_K = int(os.environ.get("RAG_TOP_K", "20"))
QDRANT_SCORE_THRESHOLD = float(os.environ.get("QDRANT_SCORE_THRESHOLD", "0.3"))
PROXY_PORT = int(os.environ.get("RAG_PROXY_PORT", "8090"))

# Thinking budget — allows the model to reason across retrieved chunks
# and general knowledge without unconstrained latency
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "1024"))

# ── Globals initialized at startup ───────────────────────────────────────────

qdrant: QdrantClient = None
embedder: Embedder = None
docstore = None
_collection_exists: bool = False
_ready: bool = False

# Separate thread pools for embedding and reranking so they don't
# compete under concurrent requests. Embedding feeds Qdrant search;
# reranking runs after hydration — keeping them independent prevents
# reranking from blocking the next request's embedding.
_embed_executor = ThreadPoolExecutor(max_workers=2)
_rerank_executor = ThreadPoolExecutor(max_workers=2)

# Persistent httpx client — reuses TCP connections across requests
_http_client: httpx.AsyncClient = None

# Semantic router — initialized from RAGPIPE_ROUTES_FILE if set
_router = None

# Embedding cache — avoids re-encoding repeated queries
EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "256"))

# Qdrant result cache — skips the HTTP round-trip for repeated queries
QDRANT_CACHE_SIZE = int(os.environ.get("QDRANT_CACHE_SIZE", "512"))

# Qdrant result cache — OrderedDict-based LRU keyed by (query_hash, collection).
# Thread-safe via a lock since _search_qdrant_sync runs in the thread pool.
_qdrant_cache: collections.OrderedDict[tuple[str, str], list[dict]] = collections.OrderedDict()
_qdrant_cache_lock = threading.Lock()

# Query-log writer — asyncpg pool for direct writes to query_log.
# Uses the same DOCSTORE_URL as the docstore since they share the DB.
_query_log_pool = None
_query_log_init_lock = threading.Lock()


async def _get_query_log_pool():
    """Lazy-init asyncpg pool for query_log writes. Fire-and-forget."""
    global _query_log_pool
    if _query_log_pool is not None:
        return _query_log_pool
    with _query_log_init_lock:
        if _query_log_pool is not None:
            return _query_log_pool
        from ragpipe.docstore import DOCSTORE_URL
        if not DOCSTORE_URL:
            return None
        import asyncpg
        _query_log_pool = await asyncpg.create_pool(DOCSTORE_URL, min_size=1, max_size=2)
        return _query_log_pool


async def _write_query_log(
    *,
    query_text: str,
    query_hash: str,
    grounding: str,
    cited_chunks: list[str],
    total_chunks: int,
    latency_ms: int | None,
    model: str | None,
    route: str | None,
    collection_id: str | None,
) -> None:
    """Write a completed request to query_log. Fire-and-forget, never surfaces errors."""
    try:
        pool = await _get_query_log_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO query_log
                    (collection_id, query_text, query_hash, grounding, cited_chunks,
                     total_chunks, latency_ms, model, route)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                collection_id,
                query_text,
                query_hash,
                grounding,
                cited_chunks,
                total_chunks,
                latency_ms,
                model,
                route,
            )
    except Exception:
        log.warning("Failed to write query_log entry — continuing", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global qdrant, embedder, docstore, _collection_exists, _http_client, _router, _ready
    # Qdrant — deferred; RAG context unavailable until Qdrant is reachable
    log.info("Connecting to Qdrant at %s", QDRANT_URL)
    try:
        qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
        collections = [c.name for c in qdrant.get_collections().collections]
        _collection_exists = COLLECTION_NAME in collections
        if not _collection_exists:
            log.warning(
                "Collection '%s' not found in Qdrant — RAG context will be empty until documents are ingested",
                COLLECTION_NAME,
            )
    except Exception:
        log.warning("Qdrant unavailable at %s — will retry on first request", QDRANT_URL)
        qdrant = None

    # Guard: RAG_TOP_K must not exceed MIGRAPHX_BATCH_SIZE or MIGraphX
    # will encounter a shape mismatch on the reranker's first real call.
    from ragpipe.models import MIGRAPHX_BATCH_SIZE

    if TOP_K > MIGRAPHX_BATCH_SIZE:
        raise RuntimeError(
            f"RAG_TOP_K={TOP_K} exceeds MIGRAPHX_BATCH_SIZE={MIGRAPHX_BATCH_SIZE}. "
            f"MIGraphX compiles static graphs — all batch dims must fit within "
            f"MIGRAPHX_BATCH_SIZE. Increase MIGRAPHX_BATCH_SIZE or reduce RAG_TOP_K."
        )

    log.info("Loading embedding model: %s (ONNX Runtime)", EMBED_MODEL)
    embedder = Embedder(repo_id=EMBED_MODEL)
    embedder.load()

    # Docstore — deferred; schema creation retried on first use if Postgres is down
    from ragpipe.docstore import create_docstore

    try:
        docstore = create_docstore()
    except Exception:
        log.warning("Docstore unavailable — will retry on first request")
        docstore = None

    # Persistent httpx client — reuses TCP connections to the model
    _http_client = httpx.AsyncClient(
        timeout=300,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )

    # Initialize semantic router if routes config is provided.
    # This triggers the FIRST MIGraphX compile (embedder, batch=64).
    if ROUTES_FILE:
        from ragpipe.router import SemanticRouter, load_routes_config

        configs, threshold, fallback = load_routes_config(ROUTES_FILE)
        _router = SemanticRouter(configs, embedder, threshold=threshold, fallback_route=fallback)
        log.info("Semantic router initialized with %d routes from %s", len(configs), ROUTES_FILE)
    else:
        log.info("No RAGPIPE_ROUTES_FILE — single-pipeline mode")

    # Warm up the reranker AFTER the embedder compile finishes.
    # MIGraphX cannot compile two graphs concurrently — sequential
    # warmup avoids crashes from concurrent rocMLIR compilations.
    # This triggers the SECOND MIGraphX compile (reranker, batch=64).
    try:
        from ragpipe.reranker import warm_up

        warm_up()
    except Exception:
        log.warning("Reranker warm-up failed — will load on first request")

    log.info(
        "Ragpipe ready — forwarding to %s (thinking_budget=%d, embed_cache=%d, qdrant_cache=%d, score_threshold=%.2f)",
        MODEL_URL,
        THINKING_BUDGET,
        EMBED_CACHE_SIZE,
        QDRANT_CACHE_SIZE,
        QDRANT_SCORE_THRESHOLD,
    )

    _ready = True

    yield

    # Shutdown
    if _router:
        await _router.close_all()
    if _http_client:
        await _http_client.aclose()
    if docstore:
        docstore.close()
    if qdrant:
        qdrant.close()
    _embed_executor.shutdown(wait=False)
    _rerank_executor.shutdown(wait=False)
    log.info("Ragpipe shut down")


app = FastAPI(lifespan=lifespan)


# ── Retrieval pipeline ───────────────────────────────────────────────────────


@functools.lru_cache(maxsize=EMBED_CACHE_SIZE)
def _embed_query(query: str) -> tuple:
    """Embed a query string, caching the result. Returns tuple for hashability.

    The key is normalized (strip + lower) so trivial whitespace/case
    differences don't waste cache slots or re-compute embeddings.
    """
    vec = embedder.embed_one(query)
    return tuple(vec.tolist())


def _embed_query_normalized(query: str) -> tuple:
    """Normalize query before cache lookup — strip whitespace, lowercase."""
    return _embed_query(query.strip().lower())


_qdrant_init_lock = threading.Lock()


def _check_collection() -> bool:
    """Check if the Qdrant collection exists, with caching."""
    global _collection_exists, qdrant
    if _collection_exists:
        return True
    if qdrant is None:
        with _qdrant_init_lock:
            if qdrant is None:
                try:
                    qdrant = QdrantClient(url=QDRANT_URL, timeout=10)
                except Exception:
                    return False
    # Re-check — ingestion may have created it since startup
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
    except Exception:
        return False
    _collection_exists = COLLECTION_NAME in collections
    return _collection_exists


def _search_qdrant_sync(
    query: str,
    *,
    qdrant_client=None,
    collection: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """Synchronous Qdrant search — runs in thread pool.

    Results are cached per (query_hash, collection) to skip the Qdrant HTTP
    round-trip on repeated queries. Empty results are NOT cached because the
    collection may not exist yet (ingestion in progress).
    """
    qc = qdrant_client or qdrant
    coll = collection or COLLECTION_NAME
    k = top_k or TOP_K
    threshold = score_threshold if score_threshold is not None else QDRANT_SCORE_THRESHOLD
    try:
        # Check collection exists (only for the global client)
        if qc is qdrant and not _check_collection():
            return []

        # Check the LRU cache before hitting Qdrant
        cache_key = (hashlib.sha256(query.encode()).hexdigest(), coll)
        with _qdrant_cache_lock:
            if cache_key in _qdrant_cache:
                _qdrant_cache.move_to_end(cache_key)
                log.debug("Qdrant cache hit: collection=%s", coll)
                return _qdrant_cache[cache_key]

        query_vector = list(_embed_query_normalized(query))
        results = qc.query_points(
            collection_name=coll,
            query=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=threshold,
        )

        if not results.points:
            return []

        payloads = [point.payload for point in results.points if point.payload.get("doc_id")]

        # Cache non-empty results — empty results are not cached because
        # the collection may have been created after the last check
        if payloads:
            with _qdrant_cache_lock:
                _qdrant_cache[cache_key] = payloads
                _qdrant_cache.move_to_end(cache_key)
                while len(_qdrant_cache) > QDRANT_CACHE_SIZE:
                    _qdrant_cache.popitem(last=False)

        return payloads
    except Exception:
        log.exception("Qdrant search failed")
        return []


async def _hydrate(refs: list[dict], *, ds=None) -> list[dict]:
    """Async docstore hydration — uses asyncpg pool, no thread pool needed."""
    if not refs:
        return []

    global docstore
    effective_ds = ds or docstore
    if effective_ds is None:
        from ragpipe.docstore import create_docstore

        try:
            docstore = create_docstore()
            effective_ds = docstore
        except Exception:
            log.warning("Docstore still unavailable — returning results without chunk text")
            return []
    lookup_keys = [(r["doc_id"], r["chunk_id"]) for r in refs]
    chunks = await effective_ds.get_chunks_async(lookup_keys)

    hydrated = []
    for ref in refs:
        key = (ref["doc_id"], ref["chunk_id"])
        chunk_data = chunks.get(key)
        if chunk_data is None:
            log.warning(
                "Orphaned vector: doc_id=%s chunk_id=%d — excluding from results",
                ref["doc_id"],
                ref["chunk_id"],
            )
            continue
        text = chunk_data.get("text", "") if isinstance(chunk_data, dict) else (chunk_data or "")
        title = chunk_data.get("title", "") if isinstance(chunk_data, dict) else ""
        source = chunk_data.get("source", ref.get("source", "unknown")) if isinstance(chunk_data, dict) else ref.get("source", "unknown")
        hydrated.append(
            {
                "text": text,
                "title": title,
                "source": source,
                "doc_id": ref["doc_id"],
                "chunk_id": ref["chunk_id"],
            }
        )

    return hydrated


def _rerank_sync(
    query: str,
    candidates: list[dict],
    *,
    min_score: float | None = None,
    top_n: int | None = None,
) -> list[dict]:
    """Synchronous reranking — runs in thread pool."""
    return rerank(query, candidates, min_score=min_score, top_n=top_n)


async def retrieve_and_rerank(
    user_query: str,
    *,
    pipeline=None,
) -> tuple[list[dict], list[dict]]:
    """Full async retrieval pipeline: Qdrant → hydrate → rerank.

    When pipeline is provided, uses per-route resources. Otherwise
    falls back to module-level singletons (backward compatible).

    Returns (ranked_chunks, all_retrieved_refs) where all_retrieved_refs
    is the full set of hydrated results before reranking, needed for
    citation validation.
    """
    loop = asyncio.get_running_loop()

    if pipeline is not None:
        search_kwargs = {
            "qdrant_client": pipeline.qdrant,
            "collection": pipeline.config.qdrant_collection or COLLECTION_NAME,
            "top_k": pipeline.config.top_k,
            "score_threshold": pipeline.config.qdrant_score_threshold,
        }
        rerank_kwargs = {
            "min_score": pipeline.config.reranker_min_score,
            "top_n": pipeline.config.reranker_top_n,
        }
        ds = pipeline.docstore
    else:
        search_kwargs = {}
        rerank_kwargs = {}
        ds = None

    refs = await loop.run_in_executor(_embed_executor, lambda: _search_qdrant_sync(user_query, **search_kwargs))
    candidates = await _hydrate(refs, ds=ds)
    ranked = await loop.run_in_executor(_rerank_executor, lambda: _rerank_sync(user_query, candidates, **rerank_kwargs))
    return ranked, candidates


# ── Request processing ───────────────────────────────────────────────────────


async def process_chat_request(body: dict, *, pipeline=None) -> tuple[dict, dict]:
    """Process a chat completion request with grounding.

    When pipeline is provided, uses per-route resources. Otherwise
    falls back to module-level singletons (backward compatible).

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

    # Check if this route has RAG disabled
    rag_enabled = pipeline.config.rag_enabled if pipeline else True

    if rag_enabled:
        ranked, all_candidates = await retrieve_and_rerank(user_query, pipeline=pipeline)
    else:
        ranked, all_candidates = [], []

    corpus_coverage = determine_corpus_coverage(ranked)

    retrieved_set = {(c["doc_id"], c["chunk_id"]) for c in all_candidates}

    # Format context with citation-friendly labels
    effective_ds = pipeline.docstore if pipeline else docstore
    context = format_context(ranked, docstore=effective_ds)
    effective_prompt = pipeline.system_prompt if pipeline else None
    system_content = build_system_message(context, system_prompt=effective_prompt)

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
        "docstore": effective_ds,
    }


def process_response(response_data: dict, ctx: dict) -> tuple[dict, dict]:
    """Post-process the LLM response: validate citations, add metadata, audit.

    This is where we parse the model's output, check that cited chunks are
    real and were in the retrieved set, classify the grounding mode, and
    emit the audit log. Crucially, this parsing is done by code — we never
    ask the LLM to generate the metadata.

    Returns (response_data, metadata) — metadata is empty dict if response
    has no choices or no content, signalling the caller to skip query_log.
    """
    choices = response_data.get("choices", [])
    if not choices:
        return response_data, {}

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return response_data, {}

    user_query = ctx.get("user_query", "")
    ranked = ctx.get("ranked", [])
    retrieved_set = ctx.get("retrieved_set", set())
    corpus_coverage = ctx.get("corpus_coverage", "none")

    # Parse citations from the model's response
    citations = parse_citations(content)

    # Validate each citation against the retrieved set and docstore
    effective_ds = ctx.get("docstore", docstore)
    valid_citations, validation_errors = validate_citations(citations, retrieved_set, effective_ds)

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
    metadata = build_metadata(content, valid_citations, corpus_coverage, docstore=effective_ds)

    # Resolve titles for audit log and query_log
    cited_chunk_titles = {}
    if effective_ds and valid_citations:
        try:
            cited_chunk_titles = effective_ds.get_chunks(valid_citations)
        except Exception:
            pass

    # Attach metadata to the response
    response_data["rag_metadata"] = metadata

    # Audit log — never logs text content
    log_audit(
        q_hash=query_hash(user_query),
        ranked_chunks=ranked,
        corpus_coverage=corpus_coverage,
        grounding=metadata["grounding"],
        valid_citations=valid_citations,
        citation_validation=citation_status,
        cited_chunk_titles=cited_chunk_titles,
    )

    return response_data, metadata


def _validate_streamed_response(content: str, ctx: dict) -> dict:
    """Post-hoc validation for streamed responses.

    Runs the same citation validation, grounding classification, and
    audit logging as process_response, but without modifying the
    response (which has already been sent to the client).

    Invalid citations are logged as errors but cannot be stripped
    from the already-delivered stream. The audit log captures the
    grounding decision for observability and tuning.

    Returns metadata dict (empty if no content) for query_log write.
    """
    user_query = ctx.get("user_query", "")
    ranked = ctx.get("ranked", [])
    retrieved_set = ctx.get("retrieved_set", set())
    corpus_coverage = ctx.get("corpus_coverage", "none")

    citations = parse_citations(content)
    effective_ds = ctx.get("docstore", docstore)
    valid_citations, validation_errors = validate_citations(citations, retrieved_set, effective_ds)

    if not citations:
        citation_status = "pass"
    elif validation_errors:
        citation_status = "invalid"
        q_hash = query_hash(user_query)
        for err in validation_errors:
            log.error(
                "Invalid citation (streamed): query_hash=%s doc_id=%s chunk_id=%d reason=%s",
                q_hash,
                err["doc_id"],
                err["chunk_id"],
                err["reason"],
            )
    else:
        citation_status = "pass"

    metadata = build_metadata(content, valid_citations, corpus_coverage, docstore=effective_ds)

    cited_chunk_titles = {}
    if effective_ds and valid_citations:
        try:
            cited_chunk_titles = effective_ds.get_chunks(valid_citations)
        except Exception:
            pass

    log_audit(
        q_hash=query_hash(user_query),
        ranked_chunks=ranked,
        corpus_coverage=corpus_coverage,
        grounding=metadata["grounding"],
        valid_citations=valid_citations,
        citation_validation=citation_status,
        cited_chunk_titles=cited_chunk_titles,
    )

    return metadata


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

    # Route selection — when a router is configured, classify the query
    # and select a per-route pipeline. Otherwise use module-level globals.
    pipeline = None
    if _router is not None:
        user_msg = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        if user_msg:
            import numpy as np

            query_vec = np.array(_embed_query_normalized(user_msg))
            route_name, route_score = _router.classify(query_vec)
            pipeline = _router.get_pipeline(route_name)
            retrieval_ctx_extra = {"route_name": route_name, "route_score": route_score}
            log.info("Routed to '%s' (score=%.4f)", route_name, route_score)
        else:
            retrieval_ctx_extra = {}
    else:
        retrieval_ctx_extra = {}

    body, retrieval_ctx = await process_chat_request(body, pipeline=pipeline)
    retrieval_ctx.update(retrieval_ctx_extra)

    # Determine which model URL to forward to
    target_url = pipeline.config.model_url if pipeline else MODEL_URL

    stream = body.get("stream", False)

    if stream:
        # Dual-path streaming: forward chunks to the client immediately
        # while accumulating the full response text in parallel. After
        # the stream completes, run citation validation + grounding
        # classification + audit logging on the accumulated text.
        # Zero latency impact — the user sees tokens as they arrive.
        accumulated_content = []

        async def stream_response():
            timings = None
            usage = None
            last_chunk_id = None
            async with _http_client.stream(
                "POST",
                f"{target_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
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
                        # Accumulate content deltas for post-stream validation
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            accumulated_content.append(text)
                    except json.JSONDecodeError:
                        pass
                    yield raw + "\n\n"

        async def validate_after_stream(response: StreamingResponse) -> StreamingResponse:
            """Wrap the streaming response to trigger post-hoc validation."""
            async for chunk in response.body_iterator:
                yield chunk
            # Stream complete — run citation validation on accumulated text
            full_content = "".join(accumulated_content)
            if full_content:
                metadata = _validate_streamed_response(full_content, retrieval_ctx)
                if metadata:
                    model = body.get("model")
                    route_name = retrieval_ctx.get("route_name")
                    asyncio.create_task(
                        _write_query_log(
                            query_text=retrieval_ctx.get("user_query", ""),
                            query_hash=query_hash(retrieval_ctx.get("user_query", "")),
                            grounding=metadata["grounding"],
                            cited_chunks=[c["id"] for c in metadata.get("cited_chunks", [])],
                            total_chunks=len(retrieval_ctx.get("ranked", [])),
                            latency_ms=None,
                            model=model,
                            route=route_name,
                            collection_id=None,
                        ),
                        name="query_log_write",
                    )

        return StreamingResponse(
            validate_after_stream(StreamingResponse(stream_response(), media_type="text/event-stream")),
            media_type="text/event-stream",
        )
    else:
        resp = await _http_client.post(
            f"{target_url}/v1/chat/completions",
            json=body,
        )
        try:
            resp.raise_for_status()
            response_data = resp.json()
        except httpx.HTTPStatusError as e:
            log.error("Model returned %d: %s", e.response.status_code, e.response.text[:200])
            return JSONResponse({"error": str(e)}, status_code=e.response.status_code)
        # Post-process: validate citations, classify grounding, audit
        response_data, metadata = process_response(response_data, retrieval_ctx)
        if metadata:
            model = body.get("model")
            route_name = retrieval_ctx.get("route_name")
            asyncio.create_task(
                _write_query_log(
                    query_text=retrieval_ctx.get("user_query", ""),
                    query_hash=query_hash(retrieval_ctx.get("user_query", "")),
                    grounding=metadata["grounding"],
                    cited_chunks=[c["id"] for c in metadata.get("cited_chunks", [])],
                    total_chunks=len(retrieval_ctx.get("ranked", [])),
                    latency_ms=None,
                    model=model,
                    route=route_name,
                    collection_id=None,
                ),
                name="query_log_write",
            )
        return JSONResponse(content=response_data)


@app.get("/health")
async def health():
    if not _ready:
        return JSONResponse({"status": "starting"}, status_code=503)
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

    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(_embed_executor, lambda: embedder.embed(texts).tolist())

    return JSONResponse(
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
            "model": EMBED_MODEL,
        }
    )


@app.post("/admin/reload-prompt")
async def reload_prompt(request: Request):
    """Hot-reload the system prompt from file/env/default.

    Requires RAGPIPE_ADMIN_TOKEN to be set and passed as a Bearer token.
    The adversarial tuning agent writes a new prompt file, then calls
    this endpoint to apply it without restarting ragpipe.
    """
    if not ADMIN_TOKEN:
        return JSONResponse(
            {"error": "RAGPIPE_ADMIN_TOKEN not configured — admin endpoints disabled"},
            status_code=403,
        )
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {ADMIN_TOKEN}":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    from ragpipe.grounding import reload_system_prompt

    result = reload_system_prompt()
    return JSONResponse(result)


@app.post("/admin/classify")
async def classify_query(request: Request):
    """Classify a query against the semantic router without sending a chat completion.

    Requires RAGPIPE_ADMIN_TOKEN. Returns the selected route, score, and
    all route scores for tuning centroid quality.
    """
    if not ADMIN_TOKEN:
        return JSONResponse(
            {"error": "RAGPIPE_ADMIN_TOKEN not configured — admin endpoints disabled"},
            status_code=403,
        )
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {ADMIN_TOKEN}":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if _router is None:
        return JSONResponse(
            {"error": "No routes configured — set RAGPIPE_ROUTES_FILE"},
            status_code=404,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    query = body.get("query", "")
    if not query:
        return JSONResponse({"error": "Missing 'query' field"}, status_code=400)

    import numpy as np

    query_vec = np.array(_embed_query_normalized(query))
    route_name, route_score = _router.classify(query_vec)
    all_scores = _router.all_scores(query_vec)

    return JSONResponse(
        {
            "route": route_name,
            "score": round(route_score, 4),
            "all_routes": {name: round(score, 4) for name, score in all_scores.items()},
        }
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests to the model unchanged."""
    body = await request.body()
    resp = await _http_client.request(
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
