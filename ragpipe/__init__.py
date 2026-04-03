"""Ragpipe — RAG proxy with corpus-preferring grounding and citation validation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ragpipe")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
