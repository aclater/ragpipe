"""Corpus-preferring grounding — citation parsing, validation, and audit logging.

The model is instructed to cite retrieved documents as [doc_id:chunk_id].
This module parses those citations from the response, validates them against
the docstore and the retrieved set, and produces structured metadata about
how the response was grounded (corpus, general, or mixed).

Audit logs capture retrieval and grounding metadata without logging any
query text, response text, or document content.
"""

import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime

log = logging.getLogger("ragpipe.grounding")
audit_log = logging.getLogger("ragpipe.audit")

# ── System prompt ────────────────────────────────────────────────────────────
# Default corpus-preferring grounding prompt. Override via:
#   RAGPIPE_SYSTEM_PROMPT_FILE — path to a text file (preferred for production)
#   RAGPIPE_SYSTEM_PROMPT — inline text (useful for dev/testing)

DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant with access to a curated document corpus. Use the following rules when answering:

1. If the retrieved documents contain relevant information, use them as your primary source. Cite every claim drawn from the documents using [doc_id:chunk_id]. Answer directly from the document content — do not second-guess what the document is. If the text says it is a particular law, act, or report, treat it as that document.

2. Only use the "⚠️ Not in corpus:" prefix when NONE of the retrieved documents are relevant to the question. If you cite even one document, do NOT start with or include "⚠️ Not in corpus:" — the answer IS from the corpus. Partial coverage is still corpus coverage.

3. If the retrieved documents partially answer the question, answer the covered portion from documents with citations first. You may then add supplementary context from general knowledge, clearly separated and prefixed with "⚠️ Not in corpus:" for that portion only.

4. Never present general knowledge as if it came from the retrieved documents. Never fabricate citations.

5. If you are uncertain whether your general knowledge is accurate, say so explicitly rather than stating it with false confidence."""

DEFAULT_SYSTEM_PROMPT_HASH = hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode()).hexdigest()


def _load_system_prompt() -> str:
    """Load system prompt from file, env var, or hardcoded default."""
    path = os.environ.get("RAGPIPE_SYSTEM_PROMPT_FILE")
    if path:
        with open(path) as f:
            return f.read().strip()
    override = os.environ.get("RAGPIPE_SYSTEM_PROMPT")
    if override:
        return override
    return DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = _load_system_prompt()
SYSTEM_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def reload_system_prompt() -> dict:
    """Reload the system prompt from file/env/default. Returns status dict."""
    global SYSTEM_PROMPT, SYSTEM_PROMPT_HASH
    old_hash = SYSTEM_PROMPT_HASH
    SYSTEM_PROMPT = _load_system_prompt()
    SYSTEM_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()

    path = os.environ.get("RAGPIPE_SYSTEM_PROMPT_FILE")
    if path:
        source = f"file:{path}"
    elif os.environ.get("RAGPIPE_SYSTEM_PROMPT"):
        source = "env:RAGPIPE_SYSTEM_PROMPT"
    else:
        source = "default"

    changed = old_hash != SYSTEM_PROMPT_HASH
    if changed:
        log.info("System prompt reloaded (source=%s, hash=%s)", source, SYSTEM_PROMPT_HASH[:16])
    else:
        log.info("System prompt unchanged (source=%s, hash=%s)", source, SYSTEM_PROMPT_HASH[:16])

    return {
        "status": "reloaded",
        "changed": changed,
        "hash": SYSTEM_PROMPT_HASH,
        "source": source,
    }


def format_context(ranked_chunks: list[dict], docstore=None) -> str:
    """Format reranked chunks as context with doc_id:chunk_id references.

    Each chunk is labeled with its doc_id:chunk_id so the model can cite
    it using the [doc_id:chunk_id] format specified in the system prompt.

    When a docstore is provided, chunk 0 of each referenced document is
    fetched and prepended as a document header so the model can identify
    what the document is (e.g. its title or short title).

    NOTE: Output order is deterministic (preserves reranker sort order) and
    contains no timestamps, random values, or variable whitespace. This is
    intentional — identical retrieval results produce byte-identical context,
    maximizing llama-server KV cache prefix reuse.
    """
    if not ranked_chunks:
        return ""

    # Collect unique doc_ids and fetch their header chunks (chunk 0)
    doc_headers: dict[str, dict] = {}
    if docstore:
        seen_doc_ids = {r["doc_id"] for r in ranked_chunks if r.get("doc_id")}
        # Only fetch headers for docs where chunk 0 isn't already in the results
        result_keys = {(r.get("doc_id"), r.get("chunk_id")) for r in ranked_chunks}
        need_headers = [(did, 0) for did in seen_doc_ids if (did, 0) not in result_keys]
        if need_headers:
            headers = docstore.get_chunks(need_headers)
            for (did, _), chunk_data in headers.items():
                text = chunk_data.get("text", "") if isinstance(chunk_data, dict) else (chunk_data or "")
                title = chunk_data.get("title", "") if isinstance(chunk_data, dict) else ""
                source = chunk_data.get("source", "") if isinstance(chunk_data, dict) else ""
                # Truncate header to first 500 chars — just need the title
                doc_headers[did] = {"text": text[:500], "title": title, "source": source}

    parts = []
    header_emitted: set[str] = set()
    for r in ranked_chunks:
        doc_id = r.get("doc_id", "unknown")
        chunk_id = r.get("chunk_id", 0)
        source = r.get("source", "unknown")
        title = r.get("title", "")
        text = r.get("text", "")

        # Emit document header once before the first chunk from each document
        if doc_id not in header_emitted and doc_id in doc_headers:
            hdr = doc_headers[doc_id]
            hdr_title = hdr.get("title", "")
            hdr_source = hdr.get("source", source)
            hdr_text = hdr.get("text", "")
            title_part = f'(from: "{hdr_title}") ' if hdr_title else ""
            parts.append(f"[{doc_id}:0] {title_part}(Source: {hdr_source}) — DOCUMENT HEADER:\n{hdr_text}")
            header_emitted.add(doc_id)

        title_part = f'(from: "{title}") ' if title else ""
        parts.append(f"[{doc_id}:{chunk_id}] {title_part}(Source: {source})\n{text}")

    return "\n\n".join(parts)


def build_system_message(context: str, *, system_prompt: str | None = None) -> str:
    """Build the full system message combining grounding rules and context.

    Args:
        context: Formatted chunk text with citation labels.
        system_prompt: Override the global SYSTEM_PROMPT for this request.

    NOTE: The system prompt is cached at module level (not re-read per request)
    and this template uses no timestamps, random values, or variable whitespace.
    Identical context input produces byte-identical output, maximizing
    llama-server KV cache prefix reuse.
    """
    prompt = system_prompt or SYSTEM_PROMPT
    if context:
        return f"{prompt}\n\n--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---"
    else:
        return f"{prompt}\n\nNo relevant documents were retrieved for this query."


# ── Citation parsing ─────────────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(r"\[([a-fA-F0-9-]+):(\d+)\]")


def parse_citations(response_text: str) -> list[tuple[str, int]]:
    """Extract [doc_id:chunk_id] citations from response text."""
    return [(m.group(1), int(m.group(2))) for m in _CITATION_PATTERN.finditer(response_text)]


def validate_citations(
    citations: list[tuple[str, int]],
    retrieved_set: set[tuple[str, int]],
    docstore,
) -> tuple[list[tuple[str, int]], list[dict]]:
    """Validate citations against the retrieved set and docstore.

    Returns:
        (valid_citations, validation_errors)

    Each validation error is a dict with doc_id, chunk_id, and reason.
    Invalid citations are logged as errors but do not discard the response —
    the model may have also used general knowledge legitimately.
    """
    valid = []
    errors = []

    # Batch lookup all cited chunks from docstore
    existing = docstore.get_chunks(citations) if citations else {}

    for doc_id, chunk_id in citations:
        if (doc_id, chunk_id) not in retrieved_set:
            # Citation references a chunk that wasn't in the retrieved set
            errors.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "reason": "not_in_retrieved_set",
                }
            )
        elif (doc_id, chunk_id) not in existing:
            # Citation references a chunk that doesn't exist in docstore
            errors.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "reason": "not_in_docstore",
                }
            )
        else:
            valid.append((doc_id, chunk_id))

    return valid, errors


def strip_invalid_citations(response_text: str, invalid: list[dict]) -> str:
    """Remove invalid citations from the response text.

    Invalid citations are stripped to prevent misleading the caller,
    but the rest of the response is preserved since it may contain
    legitimate general knowledge content.

    Uses regex with word boundaries to avoid corrupting citations that
    are substrings of other valid citations (e.g. [abc:1] vs [abc:10]).
    """
    for inv in invalid:
        # Escape the doc_id (contains hyphens) and use exact bracket matching
        escaped = re.escape(f"[{inv['doc_id']}:{inv['chunk_id']}]")
        response_text = re.sub(escaped, "", response_text)
    return response_text


# ── Grounding classification ─────────────────────────────────────────────────

NOT_IN_CORPUS_MARKER = "⚠️ Not in corpus:"

# Phrases indicating the model found the answer is NOT in the corpus.
# When these appear alongside citations, the citations are providing
# background context for a negative finding, not a positive answer.
_NEGATIVE_FINDING_PATTERNS = [
    "no evidence",
    "no mention",
    "no record",
    "no information",
    "not mentioned",
    "not found",
    "not referenced",
    "not discussed",
    "not included",
    "not addressed",
    "does not mention",
    "does not contain",
    "does not include",
    "does not reference",
    "does not discuss",
    "does not address",
    "do not mention",
    "do not contain",
    "do not include",
]


def _is_negative_finding(response_text: str) -> bool:
    """Detect if the response is a negative finding — the model is saying
    something is NOT in the corpus, using citations only for context.

    Only checks text BEFORE the ⚠️ marker. Negative language inside the
    warning section is the model explaining why it's using general knowledge,
    which is normal for genuine mixed responses.
    """
    # Only examine the corpus-grounded portion (before the warning marker)
    marker_pos = response_text.find(NOT_IN_CORPUS_MARKER)
    corpus_portion = response_text[:marker_pos].lower() if marker_pos >= 0 else response_text.lower()
    return any(pattern in corpus_portion for pattern in _NEGATIVE_FINDING_PATTERNS)


def classify_grounding(
    response_text: str,
    valid_citations: list[tuple[str, int]],
    corpus_coverage: str,
) -> str:
    """Classify the response grounding mode.

    Returns "corpus", "general", or "mixed" based on whether the response
    uses citations and/or the general knowledge marker.

    Special case: when the model cites documents to support a negative
    finding ("X is not mentioned in the corpus"), the citations are
    contextual background, not a positive answer. This is classified
    as "general" rather than "mixed" because the actual answer to the
    question came from the absence of information, not from the corpus.
    """
    has_citations = len(valid_citations) > 0
    has_general = NOT_IN_CORPUS_MARKER in response_text

    if corpus_coverage == "none":
        return "general"
    if has_citations and has_general:
        # Check if this is a negative finding — model cited docs for context
        # but the actual answer is "not found in corpus"
        if _is_negative_finding(response_text):
            return "general"
        return "mixed"
    if has_citations:
        return "corpus"
    return "general"


def determine_corpus_coverage(ranked_chunks: list[dict]) -> str:
    """Determine corpus coverage based on what retrieval returned.

    - "full": reranked chunks are available (normal retrieval)
    - "none": no chunks passed retrieval or reranking
    """
    if not ranked_chunks:
        return "none"
    return "full"


def build_metadata(
    response_text: str,
    valid_citations: list[tuple[str, int]],
    corpus_coverage: str,
    *,
    docstore=None,
) -> dict:
    """Build the structured metadata block for the response.

    Breaking change: cited_chunks is now a list of dicts with id, title,
    and source fields instead of a flat list of "doc_id:chunk_id" strings.
    """
    grounding = classify_grounding(response_text, valid_citations, corpus_coverage)

    # Resolve titles from docstore
    title_lookup: dict[tuple[str, int], dict] = {}
    if docstore and valid_citations:
        try:
            title_lookup = docstore.get_chunks(valid_citations)
        except Exception:
            log.debug("Failed to resolve citation titles from docstore", exc_info=True)

    cited_chunks = []
    for d, c in valid_citations:
        chunk_data = title_lookup.get((d, c), {})
        cited_chunks.append(
            {
                "id": f"{d}:{c}",
                "title": chunk_data.get("title", "") if isinstance(chunk_data, dict) else "",
                "source": chunk_data.get("source", "") if isinstance(chunk_data, dict) else "",
            }
        )

    return {
        "grounding": grounding,
        "cited_chunks": cited_chunks,
        "corpus_coverage": corpus_coverage,
    }


# ── Audit logging ────────────────────────────────────────────────────────────
# Never logs query text, response text, or document content.
# Only logs hashes, IDs, scores, and classification metadata.


def query_hash(query_text: str) -> str:
    """SHA-256 hash of the raw query text for audit correlation."""
    return hashlib.sha256(query_text.encode()).hexdigest()


def log_audit(
    q_hash: str,
    ranked_chunks: list[dict],
    corpus_coverage: str,
    grounding: str,
    valid_citations: list[tuple[str, int]],
    citation_validation: str,
    *,
    route_name: str | None = None,
    route_score: float | None = None,
    cited_chunk_titles: dict[tuple[str, int], dict] | None = None,
) -> None:
    """Write a structured audit log entry.

    This is the primary observability signal for the grounding pipeline.
    It enables corpus gap analysis (many none coverages = corpus needs
    expansion) and citation quality monitoring without exposing content.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query_hash": q_hash,
        "retrieved_chunks": [
            {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "reranker_score": c.get("reranker_score")}
            for c in ranked_chunks
        ],
        "chunks_passed_threshold": len(ranked_chunks),
        "grounding": grounding,
        "corpus_coverage": corpus_coverage,
        "cited_chunks": [
            {
                "doc_id": d,
                "chunk_id": c,
                "title": cited_chunk_titles.get((d, c), {}).get("title", "") if cited_chunk_titles else "",
                "source": cited_chunk_titles.get((d, c), {}).get("source", "") if cited_chunk_titles else "",
            }
            for d, c in valid_citations
        ],
        "citation_validation": citation_validation,
        "response_type": "answered",
    }
    if route_name is not None:
        entry["route"] = route_name
    if route_score is not None:
        entry["route_score"] = round(route_score, 4)
    audit_log.info(json.dumps(entry))
