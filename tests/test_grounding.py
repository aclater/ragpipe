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


# ── Citation validation ──────────────────────────────────────────────────────


def test_validate_citations_all_valid():
    mod = _reload()
    citations = [("doc1", 0), ("doc1", 1)]
    retrieved = {("doc1", 0), ("doc1", 1)}
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {("doc1", 0): "text0", ("doc1", 1): "text1"}

    valid, errors = mod.validate_citations(citations, retrieved, mock_ds)
    assert len(valid) == 2
    assert len(errors) == 0


def test_validate_citations_not_in_retrieved():
    mod = _reload()
    citations = [("doc1", 0), ("doc2", 5)]
    retrieved = {("doc1", 0)}
    mock_ds = MagicMock()
    mock_ds.get_chunks.return_value = {("doc1", 0): "text0", ("doc2", 5): "text5"}

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
    assert meta["cited_chunks"] == ["a:0", "b:1"]
    assert meta["corpus_coverage"] == "full"


def test_metadata_general():
    mod = _reload()
    meta = mod.build_metadata("⚠️ Not in corpus: answer", [], "none")
    assert meta["grounding"] == "general"
    assert meta["cited_chunks"] == []
    assert meta["corpus_coverage"] == "none"


# ── Audit logging ────────────────────────────────────────────────────────────


def test_audit_log_no_text_content(caplog):
    """Audit log must never contain query text, response text, or document content."""
    mod = _reload()
    import logging

    with caplog.at_level(logging.INFO, logger="ragpipe.audit"):
        mod.log_audit(
            q_hash="abc123",
            retrieved_chunks=[{"doc_id": "d1", "chunk_id": 0, "text": "SECRET TEXT"}],
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
            retrieved_chunks=[],
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
