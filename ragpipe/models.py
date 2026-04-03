"""Lightweight ONNX Runtime wrappers for embedding and reranking.

Replaces fastembed with direct ONNX Runtime + tokenizers for lower
memory footprint (~60% less), fewer dependencies, and full control
over thread count and session options.

Models are downloaded from HuggingFace Hub on first use and cached
locally. The tokenizer uses the HuggingFace tokenizers library (Rust).
"""

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

log = logging.getLogger("ragpipe.models")

CACHE_DIR = Path(os.environ.get("RAGPIPE_MODEL_CACHE", Path.home() / ".cache" / "ragpipe"))
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))


def _session_options() -> ort.SessionOptions:
    """Create ONNX Runtime session options with controlled threading."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ONNX_THREADS
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False  # reduces per-session memory overhead
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _ensure_model(repo_id: str, filenames: list[str]) -> Path:
    """Download model files from HuggingFace Hub if not cached.

    Returns the local directory containing the model files.
    """
    from huggingface_hub import hf_hub_download

    model_dir = CACHE_DIR / repo_id.replace("/", "--")
    all_present = all((model_dir / f).exists() for f in filenames)
    if all_present:
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        dest = model_dir / filename
        if not dest.exists():
            log.info("Downloading %s/%s", repo_id, filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            # hf_hub_download returns a cache path; symlink to our dir
            if not dest.exists():
                os.symlink(downloaded, dest)
    return model_dir


class Embedder:
    """ONNX Runtime text embedder with CLS pooling and L2 normalization.

    Default model: qdrant/bge-base-en-v1.5-onnx-q (quantized, 208 MB).
    """

    def __init__(
        self,
        repo_id: str = "qdrant/bge-base-en-v1.5-onnx-q",
        model_file: str = "model_optimized.onnx",
        tokenizer_file: str = "tokenizer.json",
    ):
        self.repo_id = repo_id
        self._model_file = model_file
        self._tokenizer_file = tokenizer_file
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._input_names: set[str] | None = None

    def load(self) -> None:
        """Download and load model + tokenizer."""
        model_dir = _ensure_model(self.repo_id, [self._model_file, self._tokenizer_file])
        log.info("Loading embedder %s (ONNX, threads=%d)", self.repo_id, ONNX_THREADS)
        self._session = ort.InferenceSession(
            str(model_dir / self._model_file),
            sess_options=_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / self._tokenizer_file))
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=512)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (batch_size, embedding_dim) float32 array."""
        if self._session is None:
            self.load()

        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        onnx_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            onnx_input["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, onnx_input)
        # CLS pooling: take the first token's hidden state
        embeddings = outputs[0][:, 0]
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (embedding_dim,) float32 array."""
        return self.embed([text])[0]

    @property
    def embedding_size(self) -> int:
        """Return the embedding dimension."""
        if self._session is None:
            self.load()
        return self._session.get_outputs()[0].shape[-1]


class Reranker:
    """ONNX Runtime cross-encoder reranker.

    Default model: Xenova/ms-marco-MiniLM-L-6-v2 (87 MB).
    """

    def __init__(
        self,
        repo_id: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        model_file: str = "onnx/model.onnx",
        tokenizer_file: str = "tokenizer.json",
    ):
        self.repo_id = repo_id
        self._model_file = model_file
        self._tokenizer_file = tokenizer_file
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._input_names: set[str] | None = None

    def load(self) -> None:
        """Download and load model + tokenizer."""
        model_dir = _ensure_model(self.repo_id, [self._model_file, self._tokenizer_file])
        log.info("Loading reranker %s (ONNX, threads=%d)", self.repo_id, ONNX_THREADS)
        self._session = ort.InferenceSession(
            str(model_dir / self._model_file),
            sess_options=_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / self._tokenizer_file))
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=512)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score (query, document) pairs. Returns list of relevance scores."""
        if not documents:
            return []
        if self._session is None:
            self.load()

        # Tokenize as (query, document) pairs
        encoded = self._tokenizer.encode_batch([(query, doc) for doc in documents])
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        onnx_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            onnx_input["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        outputs = self._session.run(None, onnx_input)
        # Cross-encoder output: logits column 0 is the relevance score
        return [float(s) for s in outputs[0][:, 0]]
