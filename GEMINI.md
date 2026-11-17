# Antigravity Agent — Repository Execution & Acceptance Spec
# agentic-rag-eval-engine v2.0

> This file is the **single source of truth** for Antigravity (AGY).
> All constraints below are **hard limits**. Never deviate without explicit confirmation.

---

## 1. Hardware Architecture Constraints (天选3 AMD Edition)

| Resource       | Spec                                      | Constraint                                   |
|----------------|-------------------------------------------|----------------------------------------------|
| CPU            | AMD Ryzen 7 6800H                         | Async I/O only on host; no blocking calls    |
| GPU            | NVIDIA RTX 3060 — **6 GB VRAM**           | ≤ 2.3 GB allocated (bge-m3 + reranker only)  |
| RAM            | 16 GB DDR5                                | WSL2 memory cap enforced                     |
| OS             | Windows 11 (PowerShell)                   | Native `uv` venv; no Conda                   |

### VRAM Allocation (strict ceiling)

| Model                        | Size   | VRAM    | Precision  |
|------------------------------|--------|---------|------------|
| `BAAI/bge-m3` (embed)        | ~1.1 GB| ~1.1 GB | `float16`  |
| `BAAI/bge-reranker-v2-m3`   | ~1.1 GB| ~1.1 GB | `float16`  |
| **Total**                    |        | **~2.3 GB** |         |

**NEVER** load any LLM (Ollama / vLLM / llama.cpp) locally. Cloud LLMs only via API.

---

## 2. Service Topology

```
Windows Host (uv .venv)
  ├── FastAPI + Uvicorn      → :8000
  ├── PyTorch CUDA FP16      → RTX 3060 direct
  └── LangGraph StateGraph   → AsyncSqliteSaver → data/langgraph_checkpoints.sqlite3

Docker Desktop (WSL2) — always-on:
  ├── qdrant/qdrant          → :6333 (int8 scalar quant, mem ≤ 350 MB)
  └── redis:8-alpine         → :6379 (256 MB LRU, RediSearch built-in)

Docker profiles: [observability] — on-demand only:
  ├── langfuse/langfuse      → :3000
  └── postgres:16-alpine     → :5432
```

---

## 3. Technology Stack (non-negotiable versions)

| Layer             | Technology                                  | Notes                                    |
|-------------------|---------------------------------------------|------------------------------------------|
| Runtime           | Python 3.11+, uv                            | pyproject.toml with cu124 torch index    |
| API               | FastAPI, Uvicorn, sse-starlette             | Async-first, SSE dual-channel            |
| Validation        | Pydantic V2                                 | Strict mode, no `.dict()` calls          |
| Orchestration     | LangGraph (StateGraph, AsyncSqliteSaver)    | No MemorySaver, no sync SqliteSaver      |
| Embedding         | sentence-transformers (NOT FlagEmbedding)   | FP16 on cuda                             |
| Vector DB         | Qdrant (int8 Dense + BM25 Sparse + RRF)     | Server-side fusion                       |
| Cache             | Redis 8 alpine (native vector search)       | Cosine threshold 0.92                    |
| Fast LLM          | Groq `llama-3.1-8b-instant` / `gemini-2.5-flash-lite` | Intent routing, rewrite      |
| Strong LLM        | `gemini-2.5-flash` via Google AI Studio     | Generation, hallucination check, Ragas   |
| Eval              | Ragas v0.2+ OOP metrics                     | RunConfig(max_workers=2, max_retries=5)  |
| Linting           | Ruff                                        | CI enforced, no warnings allowed         |

---

## 4. Code Quality Gates (enforced before every commit)

1. **Type safety**: All public functions annotated; Pydantic V2 models for all I/O.
2. **No placeholders**: Zero `TODO`, `pass`, `...`, or stub bodies in committed code.
3. **Async purity**: No `time.sleep`, no sync DB calls inside async functions.
4. **Exception boundaries**: All external I/O wrapped with specific exception types + structured logging.
5. **VRAM guard**: Any code loading a model must assert `torch.cuda.is_available()` and log VRAM delta.

---

## 5. Git Discipline

- **Branch strategy**: `main` (always green CI) → `feat/phase*` → merge via PR.
- **Commit format**: Conventional Commits — `feat(retrieval):`, `eval(ragas):`, `chore(init):`, etc.
- **Forbidden commands** (require explicit user confirmation): `git push --force`, `git reset --hard`, remote URL changes.
- **Commit cadence**: One commit per high-cohesion module, after smoke-test validation.

---

## 6. Pipeline Version Switch

Controlled via `PIPELINE_VERSION` env var in `app/core/config.py`:

| Value | Description                                                     |
|-------|-----------------------------------------------------------------|
| `v1`  | Fixed-length chunking + Qdrant Dense only + single-step gen     |
| `v2`  | Semantic chunking + Dense & Sparse + RRF + local CUDA reranker  |
| `v3`  | LangGraph agentic loop + citation tracing + Redis cache + router|

---

## 7. Acceptance Criteria (per module)

Each module is accepted when:
- [ ] `uv run ruff check .` → 0 errors
- [ ] `uv run pytest tests/ -x --no-header -q` → all pass (mocked GPU/API where needed)
- [ ] Structured log emitted on startup with VRAM delta (GPU modules)
- [ ] `git commit` archived with Conventional Commits message
