# ragpipe

RAG proxy with semantic routing, corpus-preferring grounding, and citation validation. Routes queries to different LLMs and RAG backends based on semantic classification, keeping data classifications separate.

## Architecture
```
POST /v1/chat/completions
  → classify query semantically (cosine similarity, <1ms)
  → select route → per-route LLM, Qdrant, docstore, system prompt
  → embed query (ONNX Runtime, gte-modernbert-base, LRU cached)
  → search route's Qdrant (top-K vectors, reference payloads only)
  → hydrate chunk text from Postgres docstore (asyncpg pool, LRU cached)
  → rerank with cross-encoder (ONNX Runtime, MiniLM-L-6-v2)
  → filter chunks below RERANKER_MIN_SCORE (-5 default)
  → inject system prompt + context with [doc_id:chunk_id] labels
  → forward to LLM via persistent httpx client (stream or non-stream)
  → parse citations from response
  → validate against retrieved set + docstore
  → strip invalid citations
  → classify grounding (corpus / general / mixed, with negative finding detection)
  → attach rag_metadata, emit audit log
```

## Package structure
```
ragpipe/
  __init__.py      — public API + __version__
  __main__.py      — python -m ragpipe entry point
  app.py           — FastAPI app, request pipeline, endpoints, admin API
  router.py        — SemanticRouter, RouteConfig, RoutePipeline, YAML config loader
  models.py        — ONNX Runtime Embedder + Reranker wrappers
  grounding.py     — system prompt (hot-reloadable), citation parsing/validation, audit
  reranker.py      — reranker stage with min score filtering
  docstore.py      — Postgres (asyncpg) / SQLite backends + CachedDocstore LRU wrapper
tests/
  test_admin.py      — 4 tests (reload endpoint auth)
  test_router.py     — 14 tests (config parsing, classification, pipeline lifecycle)
  test_grounding.py  — 36 tests (prompt, citations, grounding, negative findings, audit, reload)
  test_docstore.py   — 25 tests (backends + cache layer)
  test_reranker.py   — 12 tests (enabled/disabled/threshold/model swap)
  test_models.py     — 7 tests (embedder + reranker ONNX wrappers)
examples/
  routes-multi-host.yaml — cross-host routing config example
```

## Key design decisions
- Semantic router: cosine similarity on pre-embedded examples, <1ms classification, per-route LLM/Qdrant/docstore/prompt
- ONNX Runtime directly (no fastembed) — 708 MB RSS vs 4.1 GB, 370ms startup
- asyncpg connection pool (2-8 conns) for async hydration, psycopg2 retained for sync ingestion
- LRU chunk cache (2,048 entries) — 55% faster repeated queries, invalidated on upsert/delete
- Persistent httpx client — reuses TCP connections to model, 1-5ms saved per request
- Reranker min score threshold (-5) — filters irrelevant chunks, saves prompt tokens on adversarial queries
- Qdrant stores vectors + reference payloads only — no text
- Full chunk text lives in Postgres (or SQLite for dev)
- Citations are parsed and validated by code, not by the LLM
- Audit log captures grounding decisions without logging text content
- System prompt hot-reloadable via POST /admin/reload-prompt (secured with RAGPIPE_ADMIN_TOKEN)
- Hydration runs as native async (no thread pool hop), embedding/reranking in thread pool
- ONNX Runtime threads capped at 4, CPU memory arenas disabled
- Models downloaded from HuggingFace Hub, cached in RAGPIPE_MODEL_CACHE
- Default embedding model (Alibaba-NLP/gte-modernbert-base) is quantized ONNX, 768d, CLS pooling

## MIGraphX — AMD GPU inference

MIGraphXExecutionProvider is the correct and intended AMD GPU execution
provider for ONNX Runtime on this system. ROCMExecutionProvider is
ABI-incompatible with ROCm 7.2 (`onnxruntime-rocm` 1.22.2 links against
ROCm 6.x `.so` versions — `libhipblas.so.2` vs `.so.3`,
`libamdhip64.so.6` vs `.so.7`). It silently falls back to CPU.

MIGraphX compiles static computation graphs at first use (JIT). All inputs
must be padded to a fixed batch size (`MIGRAPHX_BATCH_SIZE=64`) before
inference and sliced after. The startup warmup must use exactly 64 inputs
so the compiled graph matches production traffic — one compile, cached
forever.

`RAG_TOP_K` must never exceed `MIGRAPHX_BATCH_SIZE`. An assertion at startup
enforces this. Do not remove it.

Do not attempt to switch to ROCMExecutionProvider — it will fail with ABI
errors against ROCm 7.x and is no longer maintained by AMD. The only
alternative is CPUExecutionProvider, which works but is significantly slower.

## Known issues
- Streaming responses are audited post-hoc (dual-path accumulation) but invalid citations cannot be stripped in-flight — logged as errors instead
- LLM phrasing variance: negative finding classifier depends on recognizable negation patterns before the ⚠️ marker — when the model phrases differently, classification may vary between runs
- /v1/models passthrough returns global upstream's model list, not routed model's
- No upstream failover — route's LLM down → 502, no automatic fallback

## Performance history
| Change | Impact |
|--------|--------|
| Drop fastembed → raw ONNX Runtime | 83% memory reduction (4.1 GB → 708 MB), 5x faster startup, ~17% faster avg query |
| asyncpg connection pool | Native async hydration, frees thread pool worker per request |
| LRU chunk cache (2,048 entries) | 55% faster repeated queries (eliminates Postgres round-trip on cache hit) |
| Persistent httpx client | Saves 1-5ms/request TCP handshake overhead |
| Reranker min score threshold (-5) | Filters irrelevant chunks, adversarial queries get clean empty context |
| Dual-path streaming audit | Streaming responses now audited + validated post-hoc, zero latency impact |
| Negative finding classifier | Citations supporting "X is not mentioned" classified as general, not mixed |
| Semantic router | <1ms query classification, cross-host routing verified |

## Running tests
```bash
pip install '.[dev]'
python -m pytest tests/ -v    # 97 tests
ruff check && ruff format --check
```

## Container image

Two variants are published, both UBI9 Python 3.11, models pre-downloaded, non-root (USER 1001):

| Variant | Tag | Containerfile | ONNX Runtime package | Base | GPU support |
|---------|-----|---------------|---------------------|------|-------------|
| CPU | `ghcr.io/aclater/ragpipe:main` | `Containerfile` | `onnxruntime` | UBI9 Python 3.11 | None (CPU only) |
| ROCm | `ghcr.io/aclater/ragpipe:main-rocm` | `Containerfile.rocm` | `onnxruntime-migraphx` 1.23.2 (from AMD repo) | rocm/dev-ubuntu-24.04:7.2.1 | MIGraphXExecutionProvider |

```bash
# CPU variant
podman build -t ragpipe -f Containerfile .

# ROCm variant (AMD GPU)
podman build -t ragpipe-rocm -f Containerfile.rocm .
```

The ROCm quadlet requires `/dev/kfd` + `/dev/dri` passthrough, `HSA_OVERRIDE_GFX_VERSION=11.5.1`, `SecurityLabelDisable=true` for SELinux `/dev/kfd` access, and `--ipc=host` for MIGraphX shared memory.


## GPU acceleration

- This system may have an AMD, NVIDIA, or Intel GPU. All services and scripts must detect the available GPU at runtime and select the appropriate acceleration stack — do not hardcode a vendor.
- Detection priority: NVIDIA CUDA > AMD ROCm > Intel XPU/OpenVINO > CPU. Fall back to CPU only when no GPU is available, and log a clear warning when doing so.
- Never default to CPU for any workload that can run on GPU. CPU fallback is acceptable only when a specific library or operation has no GPU support, and must be explicitly noted in a comment explaining why.
- For Python workloads: use torch.cuda.is_available(), torch.version.hip (ROCm), or torch.xpu.is_available() (Intel) to detect and select the correct device at runtime. Do not hardcode "cuda", "rocm", or "cpu".
- For ONNX Runtime: select ExecutionProvider based on runtime detection — CUDAExecutionProvider, ROCMExecutionProvider, OpenVINOExecutionProvider, or CPUExecutionProvider — in that priority order.
- For container workloads:
  - NVIDIA: pass --device /dev/nvidia0 (or --gpus all with nvidia-container-toolkit)
  - AMD ROCm: pass --device /dev/kfd --device /dev/dri
  - Intel: pass --device /dev/dri
  - Document any container that cannot use GPU and why.
- AMD ROCm on gfx1151: HSA_OVERRIDE_GFX_VERSION=11.5.1 is required. Set this env var in any quadlet, container, or script that uses ROCm on this hardware.
- Do not recommend or implement CPU-only solutions without first investigating whether a GPU-accelerated alternative exists for all three vendors.
- When benchmarking or profiling, always compare GPU vs CPU and report both. Never present CPU-only results as the baseline.
- When writing GPU detection code, always write it once as a shared utility function — do not duplicate vendor detection logic across files.


## Always verify current versions before using them

This is a hard requirement, not a suggestion. Using stale version numbers
wastes time, breaks builds, and has caused real incidents on this stack.

- BEFORE referencing any version number — for a container image, Python
  package, ROCm release, CUDA toolkit, npm package, system package, LLM
  model, or any other software — look it up. Do not use version numbers
  from training knowledge. They are outdated.
- For container images: check the registry (quay.io, ghcr.io,
  registry.access.redhat.com, docker.io) for the current stable tag
  before writing it. Verify the tag exists. Never use :latest in
  production quadlets.
- For Python packages: check PyPI for the current stable release
  before pinning.
- For ROCm: check https://rocm.docs.amd.com and
  https://github.com/RadeonOpenCompute/ROCm/releases for the current
  stable release. ROCm versions change frequently and using an old
  version is a primary cause of GPU acceleration failures on this stack.
- For CUDA: check https://developer.nvidia.com/cuda-downloads for the
  current stable release.
- For npm packages: check https://www.npmjs.com or run
  npm show <package> version.
- For LLM models: check Hugging Face and the model provider directly
  for current releases.
- For system packages (dnf/rpm/apt): do not pin versions unless
  explicitly asked — let the package manager resolve current stable.
- If you cannot verify a version, say so explicitly and ask.
  Do not guess. Do not use what you think the version is.


## Repository location

All code, projects, and repositories live exclusively under ~/git/.

- Never clone, create, or initialize a repository anywhere else on this
  system — not in ~/, not in /tmp, not in ~/Documents, or any other path.
- Before cloning or creating any repo, verify the target path is under
  ~/git/. If it is not, stop and correct the path.
- If you find a repository outside ~/git/, do not work in it. Move it
  to ~/git/ first, update any remotes if needed, and confirm the old
  location is removed before proceeding.
- When referencing local repos, always use ~/git/<reponame> as the path.


## User scripts and tools

User scripts and tools live in ~/.local/bin/, not ~/bin/.

- Always install scripts to ~/.local/bin/
- When referencing or running user scripts, always use ~/.local/bin/<script>
- Never create or reference scripts in ~/bin/ — that path is not used on
  this system
