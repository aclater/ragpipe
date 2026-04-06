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
    → titles extracted per source type, surfaced in rag_metadata.cited_chunks[].title
  → rerank with cross-encoder (ONNX Runtime, MiniLM-L-6-v2, CPU-only on gfx1151)
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
docs/
  configuration.md  — full environment variable reference
  api.md           — endpoints, rag_metadata, streaming behavior
  routing.md       — semantic routing configuration and debugging
  architecture.md  — performance benchmarks and pipeline details
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
- Titles stored alongside chunks in Postgres, surfaced in rag_metadata.cited_chunks[].title
- Citations are parsed and validated by code, not by the LLM
- Audit log captures grounding decisions without logging text content
- System prompt hot-reloadable via POST /admin/reload-prompt (secured with RAGPIPE_ADMIN_TOKEN)
- Routes hot-reloadable via POST /admin/reload-routes (no restart needed)
- Hydration runs as native async (no thread pool hop), embedding/reranking in thread pool
- ONNX Runtime threads capped at 4, CPU memory arenas disabled
- Models downloaded from HuggingFace Hub, cached in RAGPIPE_MODEL_CACHE
- Default embedding model (Alibaba-NLP/gte-modernbert-base) is quantized ONNX, 768d, CLS pooling

## MIGraphX — AMD GPU inference (non-gfx1151 only)

**On gfx1151 (Strix Halo), both embedder and reranker use CPU.** MIGraphX
is skipped automatically via `_is_gfx1151()` detection (checks /proc/cpuinfo
for "Strix Halo" or Family 26 + Model 112). This is because:
- MIGraphX tensors land in GTT (system RAM), not VRAM — ROCm VMM is not
  supported on UMA APUs by design
- CPU is faster than MIGraphX-on-GTT for small models like gte-modernbert-base

**On other AMD GPUs (non-UMA), MIGraphX is used normally.**

**MXR pre-compilation cache — 39x startup improvement:**

The MXR cache uses `ORT_MIGRAPHX_MODEL_CACHE_PATH` to store pre-compiled
ONNX models in `.mxr` format. On first startup, models are compiled and
cached (~149 MB per model). Subsequent startups load from cache directly:

| Startup | Time | Mechanism |
|---|---|---|
| Cold (first ever) | ~3:53 | JIT compilation |
| Warm (MXR cached) | ~6 seconds | Load from `ORT_MIGRAPHX_MODEL_CACHE_PATH` |

**`RAGPIPE_FORCE_CPU=1`** can force CPU-only mode on any platform.

**`RAGPIPE_DEVICE=cpu|migraphx|cuda`** to select a specific provider.

**⚠️ Cold start: ~3:53 on first boot.** Warm start (MXR cached): ~6 seconds.
Do not restart ragpipe in production unless critical.

`RAG_TOP_K` must never exceed `MIGRAPHX_BATCH_SIZE` when MIGraphX is used (non-gfx1151).
An assertion at startup enforces this. Do not remove it.

The reranker always runs on CPU — MIGraphX fails at inference with
"Not computable: gpu::precompile_op" for this model. On gfx1151, the
embedder also runs on CPU (auto-detected).

Do not attempt to switch to ROCMExecutionProvider — it will fail with ABI
errors against ROCm 7.x. CPUExecutionProvider is the fallback.

## Multi-collection routing

Routes file (`RAGPIPE_ROUTES_FILE`) configures semantic routing to multiple
Qdrant collections. Each route has its own collection, LLM, docstore, and
system prompt. Live collections: `personnel`, `nato`, `mpep`, `documents`.

Routes can be reloaded without restart:
```bash
curl -X POST http://localhost:8090/admin/reload-routes \
  -H "Authorization: Bearer $RAGPIPE_ADMIN_TOKEN"
```

## rag_metadata schema (v3)

`cited_chunks` is a list of objects with `id`, `title`, and `source`:
```python
cited_chunks = [
    {"id": "abc-123:0", "title": "Q3 Red Hat Strategy", "source": "gdrive://filename.pdf"},
    {"id": "abc-123:1", "title": "Q3 Red Hat Strategy", "source": "gdrive://filename.pdf"}
]
```

## Known issues
- ⚠️ MIGraphX startup takes ~3 minutes on first query — plan restarts accordingly
- Streaming responses are audited post-hoc (dual-path accumulation) but invalid citations cannot be stripped in-flight — logged as errors instead
- LLM phrasing variance: negative finding classifier depends on recognizable negation patterns before the ⚠️ marker — when the model phrases differently, classification may vary between runs
- /v1/models passthrough returns global upstream's model list, not routed model's
- No upstream failover — route's LLM down → 502, no automatic fallback

## Performance history
| Change | Impact |
|--------|--------|
| Drop fastembed → raw ONNX Runtime | 83% memory reduction (4.1 GB → 708 MB), 5x faster startup, ~17% faster avg query |
| MXR pre-compilation cache | 39x startup improvement (3:53 cold → 6s warm) via `ORT_MIGRAPHX_MODEL_CACHE_PATH` |
| CPU on gfx1151 (PR #42) | Embedder + reranker run on CPU — MIGraphX GTT on UMA APUs is slower than CPU for small models |
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
python -m pytest tests/ -v    # 164 tests
ruff check && ruff format --check
```

## Container image

Two variants are published, both UBI9 Python 3.11, models pre-downloaded, non-root (USER 1001):

| Variant | Tag | Containerfile | ONNX Runtime package | Base | GPU support |
|---------|-----|---------------|---------------------|------|-------------|
| CPU | `ghcr.io/aclater/ragpipe:main` | `Containerfile` | `onnxruntime` | UBI9 Python 3.11 | None (CPU only) |
| ROCm | `ghcr.io/aclater/ragpipe:main-rocm` | `Containerfile.rocm` | `onnxruntime-migraphx` (from AMD repo) | rocm/dev-ubuntu-24.04:7.2.1 | MIGraphXExecutionProvider |

The ROCm quadlet requires `/dev/kfd` + `/dev/dri` passthrough, `HSA_OVERRIDE_GFX_VERSION=11.5.1`, `SecurityLabelDisable=true` for SELinux `/dev/kfd` access, and `--ipc=host` for MIGraphX shared memory.


## Always verify current versions before using them

This is a hard requirement, not a suggestion. Using stale version numbers
wastes time, breaks builds, and has caused real incidents on this stack.

- BEFORE referencing any version number — for a container image, Python
  package, GitHub Action, ROCm release, npm package, LLM model, or any
  other software — look it up. Do not use version numbers from training
  knowledge. They are outdated.
- For GitHub Actions: verify via API before writing any workflow file:
    gh api repos/<owner>/<action>/releases/latest | jq .tag_name
  Use the exact tag returned. Never use what you think the version is.
  This has caused broken CI multiple times on this stack.
- For container images: check the registry (quay.io, ghcr.io,
  registry.access.redhat.com, docker.io) for the current stable tag.
  Never use :latest in production quadlets — pin to a specific tag or digest.
- For Python packages: check PyPI for the current stable release before pinning.
- For ROCm: check https://rocm.docs.amd.com for the current stable release.
- For npm packages: run npm show <package> version before pinning.
- For LLM models: check Hugging Face directly for current releases.
- If you cannot verify a version, say so explicitly. Do not guess.


## GPU acceleration

This system uses a Ryzen AI Max+ 395 APU (gfx1151) with 128GB unified memory.
There is no discrete VRAM — all GPU memory is GTT (system RAM mapped for GPU
access via ROCm). This is normal and expected for this hardware.

Memory architecture:
- VRAM: 512MB (GPU housekeeping only)
- GTT: ~113GB (all model weights, KV cache, and inference use GTT)
- GPU executes compute against GTT — this is full GPU inference, not CPU fallback
- rocm-smi --showmeminfo gtt confirms current GTT allocation

ROCm constraints on gfx1151:
- HSA_OVERRIDE_GFX_VERSION=11.5.1 required in all quadlets and scripts using ROCm
- MIGraphXExecutionProvider is the only working AMD GPU path for ONNX Runtime on ROCm 7.x
- ROCMExecutionProvider is deprecated and removed since ORT 1.23 — do not use it
- MIGRAPHX_BATCH_SIZE=64 — MIGraphX uses static shapes, pad all batches to this size
- ORT_MIGRAPHX_MODEL_CACHE_PATH — use this env var for MXR caching (not the
  model_cache_dir provider option — AMD does not compile that into their .so)
- MXR cache: 149MB .mxr file, cached on ragpipe-model-cache volume
  Cold start (no cache): ~3 minutes 53 seconds
  Warm start (cache hit): ~6 seconds (39x improvement)
  Do not treat a 6-second ragpipe startup as a problem — the cache is working

GPU detection for multi-vendor code:
- Detection priority: NVIDIA CUDA > AMD ROCm/MIGraphX > Intel XPU > CPU
- Never hardcode a vendor — detect at runtime
- For ONNX Runtime: CUDAExecutionProvider > MIGraphXExecutionProvider >
  OpenVINOExecutionProvider > CPUExecutionProvider

Container GPU passthrough:
- AMD ROCm: --device /dev/kfd --device /dev/dri
- NVIDIA: --device /dev/nvidia0 (or --gpus all with nvidia-container-toolkit)
- Intel: --device /dev/dri


## Repository location

All permanent repositories live under ~/git/.

- Never clone or initialize a repository anywhere else — not in ~/,
  not in /tmp, not in ~/Documents.
- Temporary PR work goes in ~/git-work/<issue-number>-<description>/
  (see Working directory conventions below)
- When referencing local repos, always use ~/git/<reponame> as the path.


## Working directory conventions

- ~/git/          — permanent repositories only. Long-term work lives here.
- ~/git-work/     — temporary clones for PR work only.
                    Always use ~/git-work/<issue-number>-<description>/
                    Clean up after the PR is merged.
- ~/.local/bin/   — user scripts and tools. Never use ~/bin/.

Never create git-* directories directly in ~/. They clutter the home
directory and never get cleaned up.

When starting any task requiring a repo clone:
  mkdir -p ~/git-work/<issue-number>-<description>
  cd ~/git-work/<issue-number>-<description>
  gh repo clone aclater/<repo>

When the PR is merged:
  rm -rf ~/git-work/<issue-number>-<description>

Or run periodically:
  ~/.local/bin/cleanup-git-work.sh --dry-run
  ~/.local/bin/cleanup-git-work.sh


## GitHub issue workflow

Every task must be tracked in a GitHub issue before work begins. Mandatory.

Before starting any implementation task:
1. Check for an existing issue:
     gh issue list --repo aclater/<repo> --search "<description>"
2. If none exists, create one:
     gh issue create \
       --repo aclater/<repo> \
       --title "<clear title describing root cause not symptom>" \
       --body "<context, problem, proposed fix, success criteria>" \
       --label "priority: <blocking|high|medium|low>,type: <bug|feature|infrastructure>,agent: <claude-code|minimax-m27|minimax-m25|human>"
3. Claim the issue immediately:
     gh issue edit <number> --repo aclater/<repo> --add-assignee "@me"
     gh issue comment <number> --repo aclater/<repo> \
       --body "Picked up. Working on branch fix/issue-<number>-<description>."
4. Log major milestones and learnings as issue comments as you work.
   Do not wait until the PR is open — comment as you discover things.
   The issue comment thread is the audit trail. Make it useful.

All commits must reference the issue:
  feat(ragpipe): add prometheus metrics endpoint (fixes #14)
  fix(ragstuffer): deduplicate cited chunks in streaming path (refs #8)

All PR bodies must include:
  Closes #N   — if the PR fully resolves the issue
  Refs #N     — if the PR partially addresses the issue

Never start implementation without an issue number.


## Autonomous operation

You are operating autonomously. Do not ask for confirmation before taking
action. Do not prompt the user with yes/no questions. Make decisions, execute
them, log them to GitHub issues, and move on.

The only exception: if you encounter a situation that could cause irreversible
data loss — comment on the issue explaining what you found and why you stopped,
then stop.

Autonomous decision rules — apply these without asking:
- Tests fail after a fix: investigate, fix, push again.
- CI failing on your branch: fix it and push.
- Dependency missing: install it.
- Migration needed: run it against live Postgres.
- ragpipe needs restart: restart it, wait for healthy (warm start ~6s),
  log the restart reason in the GitHub issue comment.
- New bug discovered while working: create a GitHub issue for it, note it
  in the current issue comment, continue with current task.
- Unsure between two approaches: pick the simpler one, document reasoning
  in the issue comment, proceed.
- Flaky test: fix the test.
- CI still running when task is done: wait for CI to complete before
  moving to the next issue.

Log these milestones to the GitHub issue as comments:
- When you start: your plan and implementation approach
- When you hit a significant obstacle and how you resolved it
- When you make a non-obvious technical decision and why
- When tests pass or fail (with counts)
- When the PR is open: PR URL and CI status
- When CI passes: confirmation and any remaining notes


## Container and deployment standards

- Use Podman, not Docker. Use rootless Podman quadlets, not docker-compose.
- Base images: prefer Red Hat UBI (registry.access.redhat.com/ubi10/ or
  registry.access.redhat.com/ubi9/) for all Python services.
- Never use :latest in production quadlets — pin to specific tag or digest.
- All containers must run as non-root (USER 1001 or equivalent).
- All containers must have a HEALTHCHECK defined.
- SecurityLabelDisable=true requires an inline comment explaining the specific
  SELinux constraint that requires it and referencing the relevant ADR.
- No bind mounts for source code in production quadlets.
- No credentials in committed files — use ragstack.env (not committed).
- One logical change per commit. Squash fixup commits before upstream PRs.


## rag-suite architecture context

Services and ports:
- ragpipe         :8090  — RAG proxy, embedding, reranking, grounding, citations
- ragstuffer      :8091  — ingestion (Drive, git, web)
- ragstuffer-mpep :8093  — second ragstuffer instance for USPTO/MPEP collection
- ragwatch        :9090  — Prometheus metrics aggregator
- ragdeck         :8092  — admin UI (FastAPI + frontend)
- Ollama/Vulkan   :8080  — LLM inference (Qwen3-32B dense Q4_K_M, ~19GB GTT)
- Qdrant          :6333  — vector store (4 collections: personnel, nato, mpep, documents)
- Postgres        :5432  — docstore (chunks+titles, collections, query_log partitioned)
- LiteLLM         :4000  — model proxy
- Open WebUI      :3000  — chat interface

Key architectural decisions:
- Collections split: personnel/nato/mpep/documents — separate Qdrant collections
  per domain. Reranker scores improved dramatically after this split.
- Title hydration: chunks have title column. ragpipe surfaces titles in
  rag_metadata.cited_chunks as objects {id, title, source}. System prompt
  instructs model to cite by title in prose while emitting [doc_id:chunk_id].
- Citation format: [doc_id:chunk_id] e.g. [133abba5-9eeb-5a99-8a5c:2]
  NOT [doc_id:133abba5...:chunk_id:2] — the verbose format is a bug.
- Grounding classification: corpus | general | mixed
- Hot-reload: POST /admin/reload-routes and POST /admin/reload-prompt
  avoid restarts for config changes. Use these instead of restarting ragpipe.
- MXR cache: ORT_MIGRAPHX_MODEL_CACHE_PATH env var enables caching.
  Warm start is ~6 seconds. Cold start (empty cache) is ~3m53s.
- LLM model: Qwen3-32B dense Q4_K_M (~19GB GTT). 32B fully activated
  parameters. Use /nothink flag for structured output tasks to prevent
  thinking mode consuming all output tokens.
- Qdrant IPv4: always use curl -4 or set QDRANT__SERVICE__HOST=:: in quadlet.
  Qdrant binds IPv4 only; Fedora resolves localhost to ::1 by default.
- Phase 0 Ragas baseline (ragprobe PR #11):
    Faithfulness: 0.700 | Answer Relevance: 0.843
    Context Precision: 0.714 | Context Recall: 0.250
  Personnel route strongest (F=0.967). MPEP/patent weakest (F=0.333).
  CRAG implementation (Phase 1) targets MPEP improvement.
