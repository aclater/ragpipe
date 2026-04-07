"""Tests for the grounding module — citation parsing, validation, classification, audit."""

import hashlib
import json
import os
from unittest.mock import MagicMock


def _reload():
    import importlib

    import ragpipe.grounding

    importlib.reload(ragpipe.grounding)
    return ragpipe.grounding


# ── System prompt ────────────────────────────────────────────────────────────


def test_default_system_prompt_hash_is_stable():
    """Default system prompt must not change without updating the hash."""
    mod = _reload()
    expected = hashlib.sha256(mod.DEFAULT_SYSTEM_PROMPT.encode()).hexdigest()
    assert expected == mod.DEFAULT_SYSTEM_PROMPT_HASH


def test_system_prompt_is_default_when_no_override():
    """Without env vars, SYSTEM_PROMPT equals DEFAULT_SYSTEM_PROMPT."""
    os.environ.pop("RAGPIPE_SYSTEM_PROMPT", None)
    os.environ.pop("RAGPIPE_SYSTEM_PROMPT_FILE", None)
    mod = _reload()
    assert mod.SYSTEM_PROMPT == mod.DEFAULT_SYSTEM_PROMPT


def test_system_prompt_env_override(monkeypatch):
    """RAGPIPE_SYSTEM_PROMPT overrides the default."""
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT", "Custom prompt for testing")
    mod = _reload()
    assert mod.SYSTEM_PROMPT == "Custom prompt for testing"
    assert hashlib.sha256(b"Custom prompt for testing").hexdigest() == mod.SYSTEM_PROMPT_HASH


def test_system_prompt_file_override(tmp_path, monkeypatch):
    """RAGPIPE_SYSTEM_PROMPT_FILE takes precedence over RAGPIPE_SYSTEM_PROMPT."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Prompt from file")
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT_FILE", str(prompt_file))
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT", "This should be ignored")
    mod = _reload()
    assert mod.SYSTEM_PROMPT == "Prompt from file"


def test_system_prompt_contains_citation_format():
    mod = _reload()
    assert "[doc_id:chunk_id]" in mod.DEFAULT_SYSTEM_PROMPT


def test_system_prompt_contains_not_in_corpus_marker():
    mod = _reload()
    assert "⚠️ Not in corpus:" in mod.DEFAULT_SYSTEM_PROMPT


def test_system_prompt_contains_concrete_citation_example():
    """System prompt must include a concrete citation example to prevent verbose format."""
    mod = _reload()
    assert "[133abba5-9e3f-4b1a-8c7d-2f6e8a0b3d4c:2]" in mod.DEFAULT_SYSTEM_PROMPT
    assert "Do NOT use verbose formats" in mod.DEFAULT_SYSTEM_PROMPT


def test_system_prompt_contains_few_shot_example():
    """System prompt must include a few-shot Q&A showing citation usage."""
    mod = _reload()
    assert "## Example" in mod.DEFAULT_SYSTEM_PROMPT
    assert "Assistant:" in mod.DEFAULT_SYSTEM_PROMPT
    # The example response must contain at least one citation in the expected format
    from ragpipe.grounding import _CITATION_PATTERN

    example_start = mod.DEFAULT_SYSTEM_PROMPT.index("## Example")
    example_text = mod.DEFAULT_SYSTEM_PROMPT[example_start:]
    assert _CITATION_PATTERN.search(example_text), "Few-shot example must contain valid [doc_id:chunk_id] citations"


# ── Context formatting ───────────────────────────────────────────────────────


def test_format_context_includes_citation_labels():
    mod = _reload()
    chunks = [
        {"doc_id": "abc-123", "chunk_id": 0, "source": "test.md", "text": "hello"},
        {"doc_id": "abc-123", "chunk_id": 1, "source": "test.md", "text": "world"},
    ]
    ctx = mod.format_context(chunks)
    assert "[abc-123:0]" in ctx
    assert "[abc-123:1]" in ctx
    assert "hello" in ctx
    assert "world" in ctx


def test_format_context_empty():
    mod = _reload()
    assert mod.format_context([]) == ""


def test_build_system_message_with_context():
    mod = _reload()
    msg = mod.build_system_message("some context")
    assert "DOCUMENT CONTEXT" in msg
    assert "some context" in msg
    assert mod.SYSTEM_PROMPT in msg


def test_build_system_message_has_citation_reminder_after_context():
    """Citation format reminder must appear after the context block to reinforce
    the instruction for models with limited attention over long contexts."""
    mod = _reload()
    msg = mod.build_system_message("some context")
    end_ctx_pos = msg.index("--- END CONTEXT ---")
    reminder_pos = msg.index("REMINDER:")
    assert reminder_pos > end_ctx_pos, "Citation reminder must appear after END CONTEXT"
    assert "[doc_id:chunk_id]" in msg[end_ctx_pos:]


def test_build_system_message_no_reminder_without_context():
    """When there's no context, there should be no citation reminder."""
    mod = _reload()
    msg = mod.build_system_message("")
    assert "REMINDER:" not in msg


def test_build_system_message_empty_retrieval():
    mod = _reload()
    msg = mod.build_system_message("")
    assert "No relevant documents were retrieved" in msg
    assert mod.SYSTEM_PROMPT in msg


# ── Citation parsing ─────────────────────────────────────────────────────────


def test_parse_citations_valid():
    mod = _reload()
    text = "According to [abc-def-123:5] and [999-888:0], the answer is clear."
    cites = mod.parse_citations(text)
    assert ("abc-def-123", 5) in cites
    assert ("999-888", 0) in cites


def test_parse_citations_none():
    mod = _reload()
    assert mod.parse_citations("No citations here.") == []


def test_parse_citations_ignores_malformed():
    mod = _reload()
    text = "[not-a-citation] and [abc:notanumber] but [abc-123:5] is valid"
    cites = mod.parse_citations(text)
    assert len(cites) == 1
    assert cites[0] == ("abc-123", 5)


def test_parse_citations_rejects_verbose_format():
    """Verbose format [doc_id:...:chunk_id:N] must not match — only [hash:N] is valid."""
    mod = _reload()
    verbose = "[doc_id:133abba5-9e3f-4b1a-8c7d-2f6e8a0b3d4c:chunk_id:2]"
    cites = mod.parse_citations(verbose)
    assert cites == [], f"Verbose citation should be rejected, got {cites}"


def test_parse_citations_accepts_correct_format_alongside_verbose():
    """Correct [hash:N] citations must still parse when verbose ones are also present."""
    mod = _reload()
    text = "See [doc_id:133abba5:chunk_id:2] for details. Also [133abba5-9e3f-4b1a-8c7d-2f6e8a0b3d4c:2] confirms this."
    cites = mod.parse_citations(text)
    assert len(cites) == 1
    assert cites[0] == ("133abba5-9e3f-4b1a-8c7d-2f6e8a0b3d4c", 2)


# ── Citation validation ──────────────────────────────────────────────────────


def test_validate_citations_all_valid():
    mod = _reload()
    citations = [("doc1", 0), ("doc1", 1)]
    retrieved = {("doc1", 0), ("doc1", 1)}
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("doc1", 0): {"text": "text0", "title": "", "source": ""},
        ("doc1", 1): {"text": "text1", "title": "", "source": ""},
    }

    valid, errors = mod.validate_citations(citations, retrieved, mock_ds)
    assert len(valid) == 2
    assert len(errors) == 0


def test_validate_citations_not_in_retrieved():
    mod = _reload()
    citations = [("doc1", 0), ("doc2", 5)]
    retrieved = {("doc1", 0)}
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("doc1", 0): {"text": "text0", "title": "", "source": ""},
        ("doc2", 5): {"text": "text5", "title": "", "source": ""},
    }

    valid, errors = mod.validate_citations(citations, retrieved, mock_ds)
    assert len(valid) == 1
    assert len(errors) == 1
    assert errors[0]["reason"] == "not_in_retrieved_set"


def test_validate_citations_not_in_docstore():
    mod = _reload()
    citations = [("doc1", 0)]
    retrieved = {("doc1", 0)}
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {}

    valid, errors = mod.validate_citations(citations, retrieved, mock_ds)
    assert len(valid) == 0
    assert len(errors) == 1
    assert errors[0]["reason"] == "not_in_docstore"


def test_strip_invalid_citations():
    mod = _reload()
    text = "Claim [abc:0] is valid but [xyz:5] is not."
    invalid = [{"doc_id": "xyz", "chunk_id": 5, "reason": "not_in_retrieved_set"}]
    stripped = mod.strip_invalid_citations(text, invalid)
    assert "[abc:0]" in stripped
    assert "[xyz:5]" not in stripped


def test_strip_invalid_citations_substring_safe():
    """Stripping [abc:1] must not corrupt [abc:10]."""
    mod = _reload()
    text = "See [abc-123:1] and [abc-123:10] for details."
    invalid = [{"doc_id": "abc-123", "chunk_id": 1, "reason": "not_in_retrieved_set"}]
    stripped = mod.strip_invalid_citations(text, invalid)
    assert "[abc-123:1]" not in stripped
    assert "[abc-123:10]" in stripped


def test_invalid_citation_does_not_discard_response():
    """Invalid citations are stripped but the response is preserved."""
    mod = _reload()
    text = "⚠️ Not in corpus: General knowledge here. Also [bad:99]."
    invalid = [{"doc_id": "bad", "chunk_id": 99, "reason": "not_in_docstore"}]
    stripped = mod.strip_invalid_citations(text, invalid)
    assert "General knowledge here" in stripped
    assert "[bad:99]" not in stripped


# ── Grounding classification ─────────────────────────────────────────────────


def test_classify_corpus():
    mod = _reload()
    assert mod.classify_grounding("Based on [a:0]", [("a", 0)], "full") == "corpus"


def test_classify_general():
    mod = _reload()
    assert mod.classify_grounding("⚠️ Not in corpus: answer", [], "none") == "general"


def test_classify_mixed():
    mod = _reload()
    assert mod.classify_grounding("From [a:0]. ⚠️ Not in corpus: also this.", [("a", 0)], "full") == "mixed"


def test_classify_no_citations_with_context_is_general():
    """If the model had context but didn't cite anything, treat as general."""
    mod = _reload()
    assert mod.classify_grounding("Just an answer.", [], "full") == "general"


def test_classify_negative_finding_is_general():
    """Citations used to support 'X is not mentioned' should be general, not mixed."""
    mod = _reload()
    text = (
        "Based on the provided documents [a:0], Adam Clater is a colonel. "
        "There is no evidence of involvement with Palantir. "
        "⚠️ Not in corpus: Palantir is a data analytics company."
    )
    assert mod.classify_grounding(text, [("a", 0)], "full") == "general"


def test_classify_negative_finding_various_patterns():
    """All negative finding patterns should trigger the general classification."""
    mod = _reload()
    patterns = [
        "The documents do not mention any connection",
        "There is no record of this in the corpus",
        "This topic is not discussed in the retrieved documents",
        "No information about this was found",
        "The NDAA does not contain any reference to this",
    ]
    for pattern in patterns:
        text = f"From [a:0]: context. {pattern}. ⚠️ Not in corpus: general answer."
        result = mod.classify_grounding(text, [("a", 0)], "full")
        assert result == "general", f"Pattern '{pattern}' classified as '{result}' instead of 'general'"


def test_classify_positive_mixed_not_affected():
    """Genuine mixed responses (positive claims from both sources) stay mixed."""
    mod = _reload()
    text = "According to [a:0], the budget is $500M. ⚠️ Not in corpus: This represents a 10% increase over last year."
    assert mod.classify_grounding(text, [("a", 0)], "full") == "mixed"


def test_corpus_coverage_none():
    mod = _reload()
    assert mod.determine_corpus_coverage([]) == "none"


def test_corpus_coverage_full():
    mod = _reload()
    assert mod.determine_corpus_coverage([{"doc_id": "a", "chunk_id": 0}]) == "full"


# ── Metadata ─────────────────────────────────────────────────────────────────


def test_metadata_corpus():
    mod = _reload()
    meta = mod.build_metadata("Answer [a:0] and [b:1]", [("a", 0), ("b", 1)], "full")
    assert meta["grounding"] == "corpus"
    assert len(meta["cited_chunks"]) == 2
    assert meta["cited_chunks"][0] == {"id": "a:0", "title": "", "source": ""}
    assert meta["cited_chunks"][1] == {"id": "b:1", "title": "", "source": ""}
    assert meta["corpus_coverage"] == "full"


def test_metadata_general():
    mod = _reload()
    meta = mod.build_metadata("⚠️ Not in corpus: answer", [], "none")
    assert meta["grounding"] == "general"
    assert meta["cited_chunks"] == []
    assert meta["corpus_coverage"] == "none"


def test_metadata_deduplicates_cited_chunks():
    """Duplicate citations from streaming must produce unique cited_chunks."""
    mod = _reload()
    # Simulate model citing the same chunk 6 times
    duplicated = [("a", 2)] * 6 + [("a", 4)] * 3
    meta = mod.build_metadata("Answer [a:2] [a:2] [a:4] [a:2]", duplicated, "full")
    ids = [c["id"] for c in meta["cited_chunks"]]
    assert ids == ["a:2", "a:4"], f"Expected deduplicated list, got {ids}"


def test_metadata_dedup_preserves_insertion_order():
    """Deduplication must preserve the order of first occurrence."""
    mod = _reload()
    citations = [("b", 1), ("a", 0), ("b", 1), ("a", 0), ("c", 3)]
    meta = mod.build_metadata("text", citations, "full")
    ids = [c["id"] for c in meta["cited_chunks"]]
    assert ids == ["b:1", "a:0", "c:3"]


# ── Footnote formatting ──────────────────────────────────────────────────────


def test_format_footnotes_basic():
    """Footnotes replace raw UUIDs with [N] and append references."""
    mod = _reload()
    content = "Adam is an architect [abc-123:2]. He works at Red Hat [def-456:0]."
    valid = [("abc-123", 2), ("def-456", 0)]
    cited_chunks = [
        {"id": "abc-123:2", "title": "Resume", "source": "gdrive://resume.pdf"},
        {"id": "def-456:0", "title": "Org Chart", "source": "gdrive://org.pdf"},
    ]
    rewritten, footnotes = mod.format_footnotes(content, valid, cited_chunks)

    assert "[1]" in rewritten
    assert "[2]" in rewritten
    assert "[abc-123:2]" not in rewritten
    assert "---" in rewritten
    assert "**References**" in rewritten
    assert "1. Resume (gdrive://resume.pdf)" in rewritten
    assert "2. Org Chart (gdrive://org.pdf)" in rewritten
    assert len(footnotes) == 2
    assert footnotes[0]["number"] == 1
    assert footnotes[0]["doc_id"] == "abc-123"
    assert footnotes[1]["number"] == 2


def test_format_footnotes_deduplicates():
    """Same doc_id:chunk_id appearing twice gets the same footnote number."""
    mod = _reload()
    content = "Fact A [abc:1]. Fact B [def:0]. Fact C [abc:1]."
    valid = [("abc", 1), ("def", 0), ("abc", 1)]
    cited_chunks = [
        {"id": "abc:1", "title": "Doc A", "source": "s3://a"},
        {"id": "def:0", "title": "Doc D", "source": "s3://d"},
    ]
    rewritten, footnotes = mod.format_footnotes(content, valid, cited_chunks)

    # Two unique footnotes, not three
    assert len(footnotes) == 2
    # Both [abc:1] should become [1]
    assert rewritten.count("[1]") == 2
    assert rewritten.count("[2]") == 1


def test_format_footnotes_empty():
    """No citations returns content unchanged and empty footnotes."""
    mod = _reload()
    content = "No citations here."
    rewritten, footnotes = mod.format_footnotes(content, [], [])
    assert rewritten == content
    assert footnotes == []


def test_format_footnotes_preserves_unvalidated():
    """Citations not in valid_citations are left as-is."""
    mod = _reload()
    content = "Valid [abc:1]. Invalid [xyz:9]."
    valid = [("abc", 1)]
    cited_chunks = [{"id": "abc:1", "title": "Doc", "source": "src"}]
    rewritten, footnotes = mod.format_footnotes(content, valid, cited_chunks)

    assert "[1]" in rewritten
    assert "[xyz:9]" in rewritten  # Not in valid_citations, left alone
    assert len(footnotes) == 1


def test_format_references_section():
    """format_references_section builds the references block."""
    mod = _reload()
    footnotes = [
        {"number": 1, "doc_id": "a", "chunk_id": 0, "title": "Alpha", "source": "s://a"},
        {"number": 2, "doc_id": "b", "chunk_id": 1, "title": "Beta", "source": "s://b"},
    ]
    refs = mod.format_references_section(footnotes)
    assert "---" in refs
    assert "**References**" in refs
    assert "1. Alpha (s://a)" in refs
    assert "2. Beta (s://b)" in refs


def test_format_references_section_empty():
    """No footnotes returns empty string."""
    mod = _reload()
    assert mod.format_references_section([]) == ""


# ── Audit logging ────────────────────────────────────────────────────────────


def test_audit_log_no_text_content(caplog):
    """Audit log must never contain query text, response text, or document content."""
    mod = _reload()
    import logging

    with caplog.at_level(logging.INFO, logger="ragpipe.audit"):
        mod.log_audit(
            q_hash="abc123",
            ranked_chunks=[{"doc_id": "d1", "chunk_id": 0, "reranker_score": 0.95}],
            corpus_coverage="full",
            grounding="corpus",
            valid_citations=[("d1", 0)],
            citation_validation="pass",
        )
    for record in caplog.records:
        assert "SECRET TEXT" not in record.getMessage()


def test_audit_log_structure(caplog):
    mod = _reload()
    import logging

    with caplog.at_level(logging.INFO, logger="ragpipe.audit"):
        mod.log_audit(
            q_hash="hashval",
            ranked_chunks=[],
            corpus_coverage="none",
            grounding="general",
            valid_citations=[],
            citation_validation="pass",
        )
    entry = json.loads(caplog.records[-1].getMessage())
    assert entry["query_hash"] == "hashval"
    assert entry["grounding"] == "general"
    assert entry["corpus_coverage"] == "none"
    assert "timestamp" in entry


# ── Empty retrieval ──────────────────────────────────────────────────────────


def test_empty_retrieval_proceeds_to_llm():
    """Empty retrieval should not hard-stop. System message should note it."""
    mod = _reload()
    msg = mod.build_system_message("")
    assert "No relevant documents were retrieved" in msg
    # Coverage should be none
    assert mod.determine_corpus_coverage([]) == "none"


# ── Prompt reload ────────────────────────────────────────────────────────────


def test_reload_from_file(tmp_path, monkeypatch):
    """reload_system_prompt() re-reads the file and updates the module state."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Initial prompt")
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT_FILE", str(prompt_file))
    mod = _reload()
    assert mod.SYSTEM_PROMPT == "Initial prompt"

    # Write a new prompt and reload without reimporting
    prompt_file.write_text("Updated prompt")
    result = mod.reload_system_prompt()
    assert result["changed"] is True
    assert result["source"] == f"file:{prompt_file}"
    assert mod.SYSTEM_PROMPT == "Updated prompt"


def test_reload_unchanged(monkeypatch):
    """reload_system_prompt() reports changed=False when prompt hasn't changed."""
    monkeypatch.delenv("RAGPIPE_SYSTEM_PROMPT_FILE", raising=False)
    monkeypatch.delenv("RAGPIPE_SYSTEM_PROMPT", raising=False)
    mod = _reload()
    result = mod.reload_system_prompt()
    assert result["changed"] is False
    assert result["source"] == "default"


def test_reload_returns_hash(monkeypatch):
    """reload_system_prompt() returns the SHA-256 hash of the active prompt."""
    monkeypatch.setenv("RAGPIPE_SYSTEM_PROMPT", "Hash me")
    mod = _reload()
    result = mod.reload_system_prompt()
    assert result["hash"] == hashlib.sha256(b"Hash me").hexdigest()


# ── Title in context injection ───────────────────────────────────────────────


def test_context_injection_includes_title():
    mod = _reload()
    chunks = [
        {"doc_id": "abc-123", "chunk_id": 0, "source": "test.md", "title": "My Doc", "text": "hello"},
    ]
    ctx = mod.format_context(chunks)
    assert '(from: "My Doc")' in ctx
    assert "[abc-123:0]" in ctx


def test_context_injection_omits_empty_title():
    mod = _reload()
    chunks = [
        {"doc_id": "abc-123", "chunk_id": 0, "source": "test.md", "title": "", "text": "hello"},
    ]
    ctx = mod.format_context(chunks)
    assert "(from:" not in ctx
    assert "[abc-123:0]" in ctx


# ── Header chunk injection and retrieved_set (fixes #38) ────────────────────


def test_format_context_returns_injected_headers_with_docstore():
    """format_context with docstore returns (context, injected_headers) tuple."""
    mod = _reload()
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("doc1", 0): {"text": "header text", "title": "Doc Title", "source": "test.md"},
    }
    # Ranked chunks have chunk_id=2 only — chunk 0 must be fetched as header
    ranked = [
        {"doc_id": "doc1", "chunk_id": 2, "source": "test.md", "title": "", "text": "body"},
    ]
    ctx, injected = mod.format_context(ranked, docstore=mock_ds)
    assert "[doc1:0]" in ctx
    assert "[doc1:2]" in ctx
    assert ("doc1", 0) in injected


def test_format_context_no_injected_headers_when_chunk0_in_results():
    """When chunk 0 is already in ranked results, no header is injected."""
    mod = _reload()
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {}
    ranked = [
        {"doc_id": "doc1", "chunk_id": 0, "source": "test.md", "title": "", "text": "chunk0"},
        {"doc_id": "doc1", "chunk_id": 1, "source": "test.md", "title": "", "text": "chunk1"},
    ]
    _ctx, injected = mod.format_context(ranked, docstore=mock_ds)
    assert injected == set()


def test_format_context_empty_with_docstore():
    """Empty ranked chunks with docstore returns empty string and empty set."""
    mod = _reload()
    mock_ds = MagicMock()
    ctx, injected = mod.format_context([], docstore=mock_ds)
    assert ctx == ""
    assert injected == set()


def test_header_citation_passes_validation_with_injected_set():
    """Regression test for #38: citations to header chunks must pass validation
    when the injected headers are added to retrieved_set."""
    mod = _reload()
    mock_ds = MagicMock()
    # Docstore has both the header chunk and the retrieved chunk
    mock_ds.get_chunks.return_value = {
        ("doc1", 0): {"text": "header", "title": "Title", "source": "s"},
        ("doc1", 2): {"text": "body", "title": "Title", "source": "s"},
    }

    # Simulate the retrieval pipeline: ranked has chunk 2, not chunk 0
    ranked = [
        {"doc_id": "doc1", "chunk_id": 2, "source": "s", "title": "", "text": "body"},
    ]
    all_candidates = ranked  # pre-rerank candidates
    retrieved_set = {(c["doc_id"], c["chunk_id"]) for c in all_candidates}

    # format_context injects the header
    _ctx, injected_headers = mod.format_context(ranked, docstore=mock_ds)
    retrieved_set |= injected_headers

    # Model cites both the header and the retrieved chunk
    citations = [("doc1", 0), ("doc1", 2)]
    valid, errors = mod.validate_citations(citations, retrieved_set, mock_ds)

    assert len(valid) == 2, f"Expected both citations valid, got errors: {errors}"
    assert len(errors) == 0


def test_header_citation_fails_without_injected_set():
    """Before the fix, header citations would fail validation."""
    mod = _reload()
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("doc1", 0): {"text": "header", "title": "Title", "source": "s"},
        ("doc1", 2): {"text": "body", "title": "Title", "source": "s"},
    }

    # Without injecting headers into retrieved_set, chunk 0 fails
    retrieved_set = {("doc1", 2)}
    citations = [("doc1", 0), ("doc1", 2)]
    _valid, errors = mod.validate_citations(citations, retrieved_set, mock_ds)

    assert len(errors) == 1
    assert errors[0]["chunk_id"] == 0
    assert errors[0]["reason"] == "not_in_retrieved_set"


# ── Title in rag_metadata ────────────────────────────────────────────────────


def test_rag_metadata_includes_title():
    mod = _reload()
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("a", 0): {"text": "t0", "title": "Doc A", "source": "gdrive://a.pdf"},
    }
    meta = mod.build_metadata("Answer [a:0]", [("a", 0)], "full", docstore=mock_ds)
    assert len(meta["cited_chunks"]) == 1
    assert meta["cited_chunks"][0] == {"id": "a:0", "title": "Doc A", "source": "gdrive://a.pdf"}


def test_rag_metadata_empty_title():
    mod = _reload()
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("a", 0): {"text": "t0", "title": "", "source": ""},
    }
    meta = mod.build_metadata("Answer [a:0]", [("a", 0)], "full", docstore=mock_ds)
    assert meta["cited_chunks"][0]["title"] == ""
    assert meta["cited_chunks"][0]["source"] == ""
    assert "title" in meta["cited_chunks"][0]
    assert "source" in meta["cited_chunks"][0]


# ── Streaming audit includes title ───────────────────────────────────────────


def test_streaming_audit_includes_title(caplog):
    mod = _reload()
    import logging

    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {
        ("d1", 0): {"text": "t0", "title": "Streamed Doc", "source": "gdrive://s.pdf"},
    }

    with caplog.at_level(logging.INFO, logger="ragpipe.audit"):
        mod.log_audit(
            q_hash="stream-hash",
            ranked_chunks=[{"doc_id": "d1", "chunk_id": 0, "reranker_score": 0.9}],
            corpus_coverage="full",
            grounding="corpus",
            valid_citations=[("d1", 0)],
            citation_validation="pass",
            cited_chunk_titles={("d1", 0): {"text": "t0", "title": "Streamed Doc", "source": "gdrive://s.pdf"}},
        )

    entry = json.loads(caplog.records[-1].getMessage())
    assert entry["cited_chunks"][0]["title"] == "Streamed Doc"
    assert entry["cited_chunks"][0]["source"] == "gdrive://s.pdf"


# ── Audit log includes title field ───────────────────────────────────────────


def test_query_log_writes_titles(caplog):
    mod = _reload()
    import logging

    with caplog.at_level(logging.INFO, logger="ragpipe.audit"):
        mod.log_audit(
            q_hash="abc123",
            ranked_chunks=[{"doc_id": "d1", "chunk_id": 0, "reranker_score": 0.95}],
            corpus_coverage="full",
            grounding="corpus",
            valid_citations=[("d1", 0)],
            citation_validation="pass",
            cited_chunk_titles={("d1", 0): {"text": "text", "title": "Titled Doc", "source": "gdrive://f.pdf"}},
        )

    entry = json.loads(caplog.records[-1].getMessage())
    assert entry["cited_chunks"][0]["title"] == "Titled Doc"
    assert entry["cited_chunks"][0]["source"] == "gdrive://f.pdf"
    assert entry["cited_chunks"][0]["doc_id"] == "d1"
    assert entry["cited_chunks"][0]["chunk_id"] == 0
