# Ragpipe — RAG proxy with corpus-preferring grounding
# Base: UBI9 Python 3.11, pinned to digest for reproducibility
FROM registry.access.redhat.com/ubi9/python-311@sha256:8fb94e142e1093b82ca64b334264a5da8e874567d64ce4b6d614f86da3e38813

# Install the package
COPY pyproject.toml /app/pyproject.toml
COPY ragpipe/ /app/ragpipe/
WORKDIR /app
RUN pip install --no-cache-dir .

# Pre-download ONNX models so first request isn't slow
RUN python3 -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-base-en-v1.5')" && \
    python3 -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

USER 1001
EXPOSE 8090
CMD ["ragpipe"]
