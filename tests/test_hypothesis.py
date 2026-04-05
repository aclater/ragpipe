"""Property-based tests for ragpipe — padding invariants and citation parsing.

Uses hypothesis to generate arbitrary inputs and verify that core functions
never crash and maintain their documented invariants.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from ragpipe.grounding import parse_citations, strip_invalid_citations


@given(text=st.text(min_size=0, max_size=5000))
@settings(max_examples=200)
def test_parse_citations_never_crashes(text: str) -> None:
    """parse_citations must not raise on arbitrary input."""
    result = parse_citations(text)
    assert isinstance(result, list)
    for doc_id, chunk_id in result:
        assert isinstance(doc_id, str)
        assert isinstance(chunk_id, int)
        assert chunk_id >= 0


@given(
    text=st.text(min_size=0, max_size=2000),
    n_invalid=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=100)
def test_strip_invalid_citations_never_crashes(text: str, n_invalid: int) -> None:
    """strip_invalid_citations must not raise on arbitrary input."""
    invalid = [{"doc_id": f"abcd{i:04x}", "chunk_id": i} for i in range(n_invalid)]
    result = strip_invalid_citations(text, invalid)
    assert isinstance(result, str)


@given(text=st.from_regex(r"(\[([a-fA-F0-9-]+):(\d+)\]\s*)+", fullmatch=True))
@settings(max_examples=100)
def test_parse_citations_roundtrip(text: str) -> None:
    """Every well-formed citation in the input is parsed."""
    import re

    expected = re.findall(r"\[([a-fA-F0-9-]+):(\d+)\]", text)
    actual = parse_citations(text)
    assert len(actual) == len(expected)
    for (exp_doc, exp_chunk), (act_doc, act_chunk) in zip(expected, actual, strict=True):
        assert exp_doc == act_doc
        assert int(exp_chunk) == act_chunk


@given(
    chunks=st.lists(
        st.fixed_dictionaries(
            {
                "doc_id": st.text(alphabet="abcdef0123456789", min_size=4, max_size=8),
                "chunk_id": st.integers(min_value=0, max_value=100),
                "text": st.text(min_size=0, max_size=200),
                "source": st.text(min_size=0, max_size=50),
            }
        ),
        min_size=0,
        max_size=20,
    )
)
@settings(max_examples=50)
def test_format_context_never_crashes(chunks: list) -> None:
    """format_context must not raise on arbitrary chunk data."""
    from ragpipe.grounding import format_context

    result = format_context(chunks)
    assert isinstance(result, str)
    # Empty input must produce empty output
    if not chunks:
        assert result == ""
