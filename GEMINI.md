# Antigravity Agent — Repository Execution & Acceptance Spec
# agentic-rag-eval-engine v2.0

> This file is the **single source of truth** for Antigravity (AGY).
> All constraints below are **hard limits**. Never deviate without explicit confirmation.

---

## 0. 最高优先级零成本安全约束 (Zero-Cost Safety Lock)

- **永久红线**：**严禁开启 `USE_VERTEX_AI=true` 或发起任何真实云端付费 LLM / Vertex AI API 调用**。
- **默认运行模式**：所有开发、单元测试、集成测试、并发压测（`evals/load_test.py`）、Ragas 评测（`evals/run_ragas.py`）与 `/verify` 验证流程，**必须默认运行在 `USE_VERTEX_AI=false` 与 `offline_mode=True` 的零成本离线模式下**，严禁绕过该限制，除非获得人工显式授权。

---

## 1. Hardware Architecture Constraints (Local Dev Machine (6GB VRAM Baseline))

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
- [ ] `HANDOVER.md` overwritten and updated after commit

---

## 8. 自动交接协议 (Automated Handover Protocol)

- **Git Tag 与 GitHub Release 全自动托管**：所有语义化版本 Tag（`v*.*.*`）与正式 GitHub Release（含自动生成的 Commit Changelog）均由 GitHub Actions（`.github/workflows/ci.yml` 中的 `release` 作业）在 `push origin main` 且质量门禁（`ruff` + `pytest`）全部通过后全自动生成，且**不在 CI 中反向提交修改回 `main` 分支**。本地代理严禁手动创建本地版本 Tag（避免与远端产生冲突），只需严格按 Conventional Commits（`feat:` / `fix:` / `perf:` / `refactor:` / `build:` / `docs:` / `BREAKING CHANGE` / `!:`，或 `[skip release]` 跳过发版）提交并推送至 `origin main`；需要同步远端 Tag 时执行 `git fetch --tags --force` 即可。

每次执行 `git commit` 后，**必须**立即覆盖更新根目录的 `HANDOVER.md`（已被 `.gitignore` 忽略，严禁提交到 Git），确保内容包含以下 4 项交接要素：

1. **最新 Git 状态**：`git log -n 3 --oneline` 与 `git status` 实际输出。
2. **质量门禁结果**：`uv run ruff check .` 与 `uv run pytest` 的实际执行汇总与通过状态。
3. **新增/修改模块接口摘要**：本次及近期新增模块的关键类（Class）与公开函数签名（Function Signatures）。
4. **下一阶段衔接点**：明确下一步待开发模块、依赖契约与注意事项。

---

## Agentic Workflow & Blast-Radius Gates

本仓库采用 **对话自动路由 + 6 阶段技能工作流（`.agents/skills/`）+ 项目专属验证门禁（`.agent/workflows/`）** 协同开发。用户无需手动输入斜杠命令或手写规约文件，只需在对话中自然提出需求，代理即自动完成闭环并输出最终结果：

- **对话自适应分流（Conversational Autopilot）**：
  - **轻量任务**（问答、文档校对、或 `≤ 2` 个文件的明确小改）：直接执行改动；若涉及代码或测试，收尾自动运行 `/verify`。
  - **复杂 / 跨模块任务**（新特性、架构重构、新增模块、或预计修改 `> 2` 个文件）：自动进入下方 **6 阶段闭环**（`DEFINE -> PLAN -> BUILD/DEBUG -> VERIFY -> REVIEW -> SHIP`）。
  - **双端协同模式**（用户显式调用 `/gemini-pair-dev` 或要求联合审查时触发）：基于 `bsk`（CLI + Skill）连接网页版 Gemini，配合 `.pair-state.md` 状态板、3 轮熔断（`bsk` 未启动/超时/额度耗尽自动降级本地自审）、`A+B` 防退化迁移与 `/verify` 验收（详细链路见 `.agent/workflows/gemini-pair-dev.md`）。

- **6 阶段技能路由（`.agents/skills/`）**：
  1. **DEFINE**：使用 `@skills/spec-driven-development`，先扫描现有代码对齐边界（明确改什么、**不改什么**），自动生成或更新根目录 `UPDATE_SPEC.md`（已被 `.gitignore` 忽略，严禁提交到 Git）。
  2. **PLAN**：使用 `@skills/planning-and-task-breakdown`（配合 `/plan` 规则），读取 `UPDATE_SPEC.md` 拆解为 2～5 分钟可验证的原子步骤；**若未触发下方爆炸半径门禁且无重大需求歧义，则自动放行（`auto_approve: true`）直接进入编码**；仅当触发爆炸半径门禁或存在高风险歧义时才暂停等待人工确认。
  3. **BUILD / DEBUG**：核心逻辑变更遵循 `@skills/test-driven-development`；遇到报错或回归故障时自动切换至 `@skills/debugging-and-error-recovery`（`/debug`），严格按 `Reproduce -> Isolate -> Fix -> Verify` 自愈修复，严禁盲猜打补丁。
  4. **VERIFY**：执行项目专属 `/verify`（`.agent/workflows/verify.md`），在前台非沙箱环境（`BypassSandbox: true` 允许 `127.0.0.1` 回环通信，严格遵守 Section 0 零成本离线模式）实跑 `uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest -v`，附上真实退出码与结果摘要。
  5. **REVIEW & CLEANUP**：收尾前使用 `@skills/code-review-and-quality` 与 `@skills/code-simplification` 检查 `git --no-pager diff`，清除过度设计与无关改动。
  6. **SHIP**：默认汇报改动摘要与验证结果由用户决定是否提交；若执行 `git commit`，则严格按 Section 5 与 Section 8 规范提交并立即覆盖更新根目录 `HANDOVER.md`。

- **外科手术式修改原则（Existing Codebase Guardrail）**：
  - 仅修改与当前 `UPDATE_SPEC.md` 直接相关的文件与代码行，**严禁顺手重构、重命名或格式化无关的历史代码**。

- **爆炸半径拦截门禁（Blast-Radius Gates）**：
  遇到以下高影响变更时，代理**必须立即暂停并请求人工确认**，严禁擅自修改或提交：
  1. 修改 `pyproject.toml` 新增、升级或移除第三方依赖；
  2. 更改 `/api/v1/` 公共接口签名、端点路径或请求/响应 Pydantic 契约；
  3. 破坏或改写现有全量 `pytest` 单测基线断言（当前 65 项）；
  4. 修改 GitHub Actions CI/CD 配置文件（`.github/workflows/`）。

- **全局语言规范 (Language Policy)**:
  1. 所有面向用户的自然语言输出（包括 `Implementation Plan`、`Task` 清单、`Walkthrough`、代码审查说明、`HANDOVER.md` 说明正文及日常对话）必须默认使用**简体中文**。
  2. 仅代码标识符、文件路径、终端命令、技术专有名词以及 Git Commit Message（遵循 Conventional Commits）保持英文。
