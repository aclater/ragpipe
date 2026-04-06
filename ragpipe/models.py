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
MXR_CACHE_DIR = Path(os.environ.get("RAGPIPE_MXR_CACHE", Path.home() / ".cache" / "ragpipe" / "mxr"))
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "4"))
# Fixed batch size for MIGraphX graph compilation. MIGraphX JIT-compiles
# for a specific tensor shape and errors on shape mismatch. Padding all
# inputs to this batch size means the graph is compiled once at startup
# and reused for every subsequent call. Must match or exceed the largest
# batch size used by any caller (ragstuffer defaults to 64, reranker
# receives up to RAG_TOP_K candidates). See CLAUDE.md § MIGraphX.
MIGRAPHX_BATCH_SIZE = int(os.environ.get("MIGRAPHX_BATCH_SIZE", "64"))

# Map RAGPIPE_DEVICE env var values to ONNX Runtime provider names.
# MIGraphX is the only AMD GPU provider in ORT 1.23+ (ROCMExecutionProvider removed).
# "rocm" is kept as an alias for migraphx for backward compatibility.
_DEVICE_TO_PROVIDER = {
    "cuda": "CUDAExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "MIGraphXExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

# Preferred GPU providers in priority order.
# CUDAExecutionProvider for NVIDIA GPUs.
# MIGraphXExecutionProvider for AMD GPUs (graph-level compilation).
_GPU_PROVIDERS = ["CUDAExecutionProvider", "MIGraphXExecutionProvider"]


def _is_gfx1151() -> bool:
    """Detect if running on gfx1151 (Strix Halo) UMA APU.

    On gfx1151, MIGraphX tensors land in GTT (system RAM) not VRAM because
    ROCm VMM is not supported on UMA APUs by design. CPU is faster for
    small models in this configuration.
    """
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "Strix Halo" in cpuinfo or "Strix_Halo" in cpuinfo:
            return True
        if "Family 26" in cpuinfo and "Model 112" in cpuinfo:
            return True
    except OSError:
        pass
    return False


def _get_providers() -> list[str]:
    """Select ONNX Runtime execution providers.

    Priority:
    1. If RAGPIPE_FORCE_CPU=1 or RAGPIPE_DEVICE=cpu, use CPU only.
    2. If RAGPIPE_DEVICE is set to a specific provider, use it (with CPU fallback).
    3. If gfx1151 (Strix Halo) is detected, skip MIGraphX and use CPU.
       This is because on UMA APUs, MIGraphX tensors land in GTT (system RAM)
       and CPU outperforms MIGraphX for small models like gte-modernbert-base.
    4. Otherwise, auto-detect: prefer GPU providers over CPU.
    5. Always include CPUExecutionProvider as a fallback.

    Returns a list suitable for ``ort.InferenceSession(providers=...)``.
    """
    available = set(ort.get_available_providers())
    forced = os.environ.get("RAGPIPE_DEVICE", "").strip().lower()

    if forced:
        provider = _DEVICE_TO_PROVIDER.get(forced)
        if provider is None:
            log.warning(
                "RAGPIPE_DEVICE=%r is not recognized (valid: %s); falling back to auto-detect",
                forced,
                ", ".join(sorted(_DEVICE_TO_PROVIDER)),
            )
        elif provider not in available:
            log.warning(
                "RAGPIPE_DEVICE=%r requested %s but it is not available; falling back to CPU",
                forced,
                provider,
            )
            provider = None
        else:
            providers = [provider]
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
            log.info("ONNX providers (forced via RAGPIPE_DEVICE=%s): %s", forced, providers)
            return providers

    # RAGPIPE_FORCE_CPU=1 override — useful for UMA APUs where CPU is faster
    if os.environ.get("RAGPIPE_FORCE_CPU", "").strip().lower() in ("1", "true", "yes"):
        log.info("ONNX providers: [CPUExecutionProvider] (forced via RAGPIPE_FORCE_CPU)")
        return ["CPUExecutionProvider"]

    # On gfx1151 (Strix Halo), skip MIGraphX — tensors land in GTT, CPU is faster
    if _is_gfx1151() and "MIGraphXExecutionProvider" in available:
        log.info(
            "ONNX providers: [CPUExecutionProvider] (gfx1151 detected, MIGraphX skipped — CPU is faster on UMA)"
        )
        return ["CPUExecutionProvider"]

    # Auto-detect: pick the first available GPU provider.
    for gpu in _GPU_PROVIDERS:
        if gpu in available:
            providers = [gpu, "CPUExecutionProvider"]
            log.info("ONNX providers (auto-detected GPU): %s", providers)
            return providers

    log.info("ONNX providers: [CPUExecutionProvider]")
    return ["CPUExecutionProvider"]


def _session_options() -> ort.SessionOptions:
    """Create ONNX Runtime session options with controlled threading."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ONNX_THREADS
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False  # reduces per-session memory overhead
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _mxr_cache_path(model_name: str) -> Path:
    """Return the MXR cache directory for a model, encoding batch size to detect shape mismatches.

    MIGraphX compiles static graphs for a fixed tensor shape. If MIGRAPHX_BATCH_SIZE
    changes, the old .mxr is invalid. Encoding the batch size in the path forces a
    recompile when the shape changes.
    """
    pad_length = int(os.environ.get("ONNX_PAD_LENGTH", "128"))
    safe_name = model_name.replace("/", "--")
    return MXR_CACHE_DIR / f"{safe_name}_b{MIGRAPHX_BATCH_SIZE}_p{pad_length}"


def _get_providers_with_options() -> list:
    """Select ONNX Runtime providers with MXR cache options for MIGraphX.

    Returns a list suitable for ort.InferenceSession(providers=...).
    When MIGraphX is selected, returns tuples of (provider_name, options_dict)
    with the MXR cache path configured. Otherwise returns plain strings.
    """
    providers = _get_providers()
    if "MIGraphXExecutionProvider" not in providers:
        return providers

    result = []
    for p in providers:
        if p == "MIGraphXExecutionProvider":
            # Use a generic cache path — the actual model-specific path is set
            # per-session via the model_name parameter in Embedder/Reranker.
            result.append(p)
        else:
            result.append(p)
    return result


def _create_session(model_path: str, model_name: str, providers: list[str]) -> ort.InferenceSession:
    """Create an ONNX Runtime InferenceSession with MXR caching for MIGraphX.

    If MIGraphX is in the provider list, sets ORT_MIGRAPHX_MODEL_CACHE_PATH
    environment variable so the provider automatically saves/loads compiled
    .mxr files. This eliminates the ~3 minute JIT compilation on subsequent
    startups.
    """
    cache_path = _mxr_cache_path(model_name)
    use_mxr = "MIGraphXExecutionProvider" in providers

    if use_mxr:
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = str(cache_path)
        cached_files = list(cache_path.glob("*.mxr"))
        if cached_files:
            log.info("MXR cache hit for %s — loading pre-compiled graph from %s", model_name, cache_path)
        else:
            log.info("MXR cache miss for %s — will JIT compile and save to %s", model_name, cache_path)

    session = ort.InferenceSession(
        model_path,
        sess_options=_session_options(),
        providers=providers,
    )

    if use_mxr and not list(cache_path.glob("*.mxr")):
        log.info("MXR compilation complete for %s — cached to %s", model_name, cache_path)

    return session


def get_mxr_status() -> dict:
    """Return MXR cache status for all known models. Used by /admin/mxr-status."""
    status = {"cache_dir": str(MXR_CACHE_DIR), "models": {}}
    if not MXR_CACHE_DIR.exists():
        return status
    for subdir in sorted(MXR_CACHE_DIR.iterdir()):
        if subdir.is_dir():
            mxr_files = list(subdir.glob("*.mxr"))
            status["models"][subdir.name] = {
                "cached": len(mxr_files) > 0,
                "files": [f.name for f in mxr_files],
                "size_mb": round(sum(f.stat().st_size for f in mxr_files) / 1048576, 1),
            }
    return status


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

    Default model: Alibaba-NLP/gte-modernbert-base (quantized, 143 MB, 768d).
    """

    def __init__(
        self,
        repo_id: str = "Alibaba-NLP/gte-modernbert-base",
        model_file: str = "onnx/model_quantized.onnx",
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
        providers = _get_providers()
        log.info("Loading embedder %s (ONNX, threads=%d, providers=%s)", self.repo_id, ONNX_THREADS, providers)
        self._session = _create_session(
            str(model_dir / self._model_file),
            self.repo_id,
            providers,
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / self._tokenizer_file))
        # Pad to fixed length so MIGraphX only compiles once per model.
        # Without this, every unique sequence length triggers a full graph
        # recompilation (~3 min on gfx1151). 128 tokens covers most queries
        # and short passages; longer inputs are truncated.
        self._pad_length = int(os.environ.get("ONNX_PAD_LENGTH", "128"))
        self._tokenizer.enable_padding(length=self._pad_length)
        self._tokenizer.enable_truncation(max_length=self._pad_length)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (batch_size, embedding_dim) float32 array.

        Inputs are padded to MIGRAPHX_BATCH_SIZE so MIGraphX only compiles
        one graph shape. Batches larger than MIGRAPHX_BATCH_SIZE are split
        and processed in sub-batches.
        """
        if self._session is None:
            self.load()

        if len(texts) > MIGRAPHX_BATCH_SIZE:
            parts = []
            for i in range(0, len(texts), MIGRAPHX_BATCH_SIZE):
                parts.append(self.embed(texts[i : i + MIGRAPHX_BATCH_SIZE]))
            return np.concatenate(parts, axis=0)

        actual = len(texts)
        # Pad to fixed batch size for stable MIGraphX graph shape
        padded_texts = texts + [""] * (MIGRAPHX_BATCH_SIZE - actual) if actual < MIGRAPHX_BATCH_SIZE else texts

        encoded = self._tokenizer.encode_batch(padded_texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        onnx_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            onnx_input["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, onnx_input)
        # CLS pooling: take the first token's hidden state
        embeddings = outputs[0][:actual, 0]
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
        """Download and load model + tokenizer.

        Always uses CPUExecutionProvider — MIGraphX compiles the MiniLM-L-6
        graph on gfx1151 but fails at inference with "Not computable:
        gpu::precompile_op". The reranker is small (87 MB) and fast on CPU.
        """
        model_dir = _ensure_model(self.repo_id, [self._model_file, self._tokenizer_file])
        providers = ["CPUExecutionProvider"]
        log.info("Loading reranker %s (ONNX, threads=%d, providers=%s)", self.repo_id, ONNX_THREADS, providers)
        self._session = ort.InferenceSession(
            str(model_dir / self._model_file),
            sess_options=_session_options(),
            providers=providers,
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / self._tokenizer_file))
        # Fixed padding length for consistent sequence dimension
        self._pad_length = int(os.environ.get("ONNX_PAD_LENGTH", "128"))
        self._tokenizer.enable_padding(length=self._pad_length)
        self._tokenizer.enable_truncation(max_length=self._pad_length)
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
