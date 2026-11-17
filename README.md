# agentic-rag-eval-engine

Production-grade **Agentic RAG pipeline** with automated evaluation — v1 / v2 / v3 architecture comparison.

## Stack

- **Retrieval**: Qdrant (`int8` Dense + BM25 Sparse) → server-side `Fusion.RRF` → local CUDA `bge-reranker-v2-m3`
- **Orchestration**: LangGraph `StateGraph` with `AsyncSqliteSaver` checkpoints
- **LLM Routing**: Groq (`llama-3.1-8b-instant`) ↔ Google AI Studio (`gemini-2.5-flash`)
- **Cache**: Redis 8 vector semantic cache (cosine ≥ 0.92)
- **Eval**: Ragas v0.2+ (`LLMContextRecall`, `Faithfulness`, `LLMContextPrecisionWithReference`)

## Quick Start

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install dependencies
uv sync --group dev

# 3. Configure environment
cp .env.example .env  # fill in GROQ_API_KEY and GOOGLE_API_KEY

# 4. Run API server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

| Version | Description |
|---------|-------------|
| `v1`    | Fixed-length chunking + Qdrant Dense + single-step generation |
| `v2`    | Semantic chunking + Dense & Sparse (BM25) + RRF + local CUDA reranker |
| `v3`    | LangGraph agentic loop + citation tracing + Redis cache + model router |

Switch via `PIPELINE_VERSION=v1|v2|v3` in `.env`.
