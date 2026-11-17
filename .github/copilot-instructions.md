# VS Code Copilot — Repository-Level Constraints
# agentic-rag-eval-engine v2.0

> Mirror of GEMINI.md. Both files must be kept in sync.
> These are hard architectural constraints for all AI-assisted edits.

---

## Hardware Red Lines

- **GPU**: RTX 3060, 6 GB VRAM. **VRAM budget = 2.3 GB** (bge-m3 + reranker only).
- **Never suggest** loading any local LLM (Ollama, llama.cpp, vLLM). Cloud APIs only.
- **Always** load embedding/reranker models with `torch.float16` on `device="cuda"`.
- **Never** install `FlagEmbedding`; use `sentence-transformers` for all model loading.

## Python Style Rules

- Python **3.11+** only. Use `uv` for dependency management (`pyproject.toml`).
- **Pydantic V2** everywhere. Use `.model_validate()`, not `.parse_obj()`. Use `model_config = ConfigDict(...)`.
- **Async-first**: every function touching I/O must be `async def`. No `time.sleep`. No sync DB calls.
- Full type annotations on all public functions and class attributes.
- No `TODO`, `pass`, or placeholder bodies — complete implementations only.
- **Ruff** is the linter/formatter. All code must pass `ruff check` with zero warnings.

## LangGraph Rules

- Use `AsyncSqliteSaver` for checkpoint persistence → `./data/langgraph_checkpoints.sqlite3`.
- **Never** use `MemorySaver` (memory leak) or sync `SqliteSaver` (blocks event loop).
- State must be a `TypedDict` defined in `app/graph/state.py`.
- Max retries for grader/rewriter loop: **`max_retries = 2`** (hard cap, prevents infinite loops).

## Service Topology (Docker)

- **Always-on** in `docker-compose.yml`: `qdrant` (int8, ≤350 MB) + `redis:8-alpine` (256 MB LRU).
- **On-demand** only via `profiles: ["observability"]`: `langfuse` + `postgres`.
- Do not add `mem_limit` overrides without updating `GEMINI.md`.

## LLM Routing Policy

| Route         | Model                              | Use Case                               |
|---------------|------------------------------------|----------------------------------------|
| Fast Route    | `llama-3.1-8b-instant` (Groq)      | Intent routing, query rewrite, simple QA |
| Fast Route    | `gemini-2.5-flash-lite`            | Fallback fast route                    |
| Strong Route  | `gemini-2.5-flash` (Google AI Studio) | Multi-hop gen, hallucination check, Ragas judge |

Target: **P95 latency < 1.8 s** end-to-end.

## Git Rules

- Conventional Commits: `feat(scope):`, `fix(scope):`, `eval(scope):`, `chore(scope):`, `test(scope):`.
- **Never** run `git push --force` or `git reset --hard` without explicit user confirmation.
- Commit after each validated module.
