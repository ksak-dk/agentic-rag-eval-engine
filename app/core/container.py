"""
app/core/container.py
=====================
Lifespan dependency injection container with automatic ``live`` / ``demo`` / ``test``
mode detection and graceful degradation.

Design Contract
---------------
- In ``demo`` and ``test`` modes, NEVER loads CUDA weights or contacts external
  Qdrant / Redis / Cloud LLM endpoints. Indexes local Markdown project docs into
  an in-memory hybrid retriever and wires ``InMemorySemanticCache`` (with
  ``SingleFlight`` lock) + ``AsyncSqliteSaver`` + ``AgenticRAGWorkflow``.
- In ``live`` mode, wires real ``HybridVectorStore`` (FP16 ``bge-m3`` +
  ``bge-reranker-v2-m3``, ~2.3 GB VRAM total, zero local LLMs), ``LLMRouter``,
  Redis 8 ``SemanticCache``, and ``AsyncSqliteSaver``. Any external dependency
  outage falls back gracefully to the corresponding ``demo`` component so
  FastAPI startup is never blocked.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.cache.semantic_cache import InMemorySemanticCache, SemanticCache, SingleFlight
from app.core.config import Settings, get_settings
from app.core.logging import update_request_metrics
from app.core.router import LLMGenerationResult, LLMRouter, RouteDecision, RouteTier, TaskType
from app.graph.nodes import DocumentGradeOutput, GraphDependencies, HallucinationGradeOutput
from app.graph.workflow import AgenticRAGWorkflow, open_async_sqlite_saver
from app.retrieval.chunker import ChunkerConfig, ChunkMetadata, ChunkMode, MarkdownHeaderChunker
from app.retrieval.hybrid_store import (
    BGEDenseEmbedder,
    BGEReranker,
    BM25SparseEmbedder,
    HybridVectorStore,
    RetrievedChunk,
)

try:
    import torch
except ImportError:  # pragma: no cover - defensive fallback for minimal envs
    torch = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

ResolvedRuntimeMode = Literal["live", "demo", "test"]

_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your_",
    "placeholder",
    "changeme",
    "replace_me",
    "example",
    "gsk_your_",
    "aizasy_your_",
)

_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-zA-Z0-9_.-]+|[\u4e00-\u9fff]")
_CONTEXT_BLOCK_RE: re.Pattern[str] = re.compile(
    r"\[([^\[\]\s]+)\]\s*\(source:\s*([^,\)]+),\s*section:\s*([^\)]+)\)\s*\n(.*?)(?=\n\n---\n\n|\Z)",
    re.DOTALL,
)

_BILINGUAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "显存": ("vram", "3060", "fp16", "2.3", "6gb", "bge-m3"),
    "缓存": ("cache", "redis", "cosine", "0.92", "semantic"),
    "重排": ("rerank", "reranker", "bge-reranker-v2-m3", "rrf"),
    "评测": ("ragas", "eval", "faithfulness", "recall", "precision", "benchmark"),
    "架构": ("architecture", "langgraph", "qdrant", "redis", "fastapi"),
    "循环": ("loop", "max_retries", "fallback", "retry"),
    "流式": ("sse", "stream", "token", "citations", "metrics"),
    "向量": ("vector", "dense", "sparse", "bm25", "int8"),
    "量化": ("quantization", "int8", "ragas", "benchmark"),
}

_DEFAULT_FALLBACK_CORPUS: str = """# agentic-rag-eval-engine Architecture Overview

## Hardware & 6GB VRAM Budget (RTX 3060)
The host runs Windows 11 + NVIDIA RTX 3060 (6 GB VRAM). Only two local models are allowed on CUDA in FP16:
- `BAAI/bge-m3` dense embedder (~1.1 GB VRAM, 1024-dim)
- `BAAI/bge-reranker-v2-m3` cross-encoder reranker (~1.1 GB VRAM)
Total VRAM budget is strictly ~2.3 GB (well under 6 GB). Zero local LLMs are ever loaded.

## v1 vs v2 vs v3 Pipeline Evolution
- `v1` Baseline: Fixed-length chunking + Qdrant INT8 dense-only retrieval + single-step fast generation (`llama-3.1-8b-instant`).
- `v2` Hybrid + CUDA Rerank: Markdown header semantic chunking (`H1/H2/H3`) + Qdrant Dense (1024-dim INT8) and BM25 Sparse prefetch + server-side `Fusion.RRF` + local FP16 `BAAI/bge-reranker-v2-m3` top-k reranking + strong generation (`gemini-2.5-flash`).
- `v3` Agentic Self-Correcting Loop: Adds Redis 8 native vector semantic cache (cosine >= 0.92), LangGraph `StateGraph` with document relevance grading, bounded query rewriting (`max_retries=2`), hallucination verification, `[doc_id]` citation tracing, and `AsyncSqliteSaver` persistence.

## Ragas v0.2+ & Async Load Benchmark
- Ragas Overall Score improves from `0.6759` (`v1`, Context Recall `0.3884`, Faithfulness `0.6392`) to `0.8554` (`v2`, Context Recall `0.7679`, Faithfulness `0.7984`) and `0.9950` (`v3`, Context Recall `1.0000`, Faithfulness `0.9851`).
- Under concurrency=10 load testing, `v3` achieves 30.0% semantic cache hit rate, P50 latency 445.69 ms, and P95 latency 678.77 ms (< 1800 ms SLA).

## Dual-Channel SSE & Cloud LLM Router
FastAPI `POST /api/v1/chat` streams dual-channel SSE events (`token`, `citations`, `metrics`, `done`). Cloud routing uses Groq `llama-3.1-8b-instant` (fast route with `gemini-2.5-flash-lite` failover) and Google AI Studio `gemini-2.5-flash` (strong route).
"""


# ---------------------------------------------------------------------------
# Environment & Service Probing Helpers
# ---------------------------------------------------------------------------


def is_cuda_available() -> bool:
    """Safely check whether CUDA GPU acceleration is available."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_placeholder_api_key(value: str | None) -> bool:
    """Return True when an API key is empty, whitespace, or an unedited template value."""
    if value is None:
        return True
    cleaned = value.strip()
    if len(cleaned) < 8:
        return True
    lowered = cleaned.lower()
    if lowered.startswith("<") or lowered.endswith(">"):
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def has_valid_cloud_api_keys(settings: Settings) -> bool:
    """Check whether at least one real cloud LLM API key is configured."""
    candidates = (
        settings.groq_api_key,
        settings.google_api_key,
        settings.gemini_api_key,
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("GEMINI_API_KEY", ""),
    )
    return any(not is_placeholder_api_key(key) for key in candidates)


async def probe_qdrant_ready(qdrant_url: str, *, timeout_seconds: float = 1.0) -> bool:
    """Perform a lightweight non-blocking HTTP health probe against Qdrant."""
    base = qdrant_url.strip().rstrip("/")
    if not base:
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for endpoint in ("/readyz", "/collections"):
                try:
                    resp = await client.get(f"{base}{endpoint}")
                    if resp.status_code < 500:
                        return True
                except httpx.HTTPError:
                    continue
    except Exception:
        return False
    return False


async def probe_redis_ready(redis_url: str, *, timeout_seconds: float = 1.0) -> bool:
    """Perform a lightweight non-blocking Redis PING check."""
    cleaned = redis_url.strip()
    if not cleaned:
        return False
    client: Redis | None = None
    try:
        client = Redis.from_url(
            cleaned,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            decode_responses=False,
        )
        pong = await client.ping()
        return bool(pong)
    except Exception:
        return False
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# In-Memory Hybrid Retriever (Demo / Test / Live-Fallback)
# ---------------------------------------------------------------------------


def _tokenize_text(text: str) -> list[str]:
    """Tokenize multilingual text into normalized terms with bilingual expansion."""
    raw_tokens = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    cjk_chars: list[str] = []
    for tok in raw_tokens:
        if "\u4e00" <= tok <= "\u9fff":
            cjk_chars.append(tok)
            tokens.append(tok)
        else:
            tokens.append(tok)
            if len(tok) >= 4:
                tokens.append(tok[:4])
    for idx in range(len(cjk_chars) - 1):
        bigram = f"{cjk_chars[idx]}{cjk_chars[idx + 1]}"
        tokens.append(bigram)
        expansions = _BILINGUAL_EXPANSIONS.get(bigram)
        if expansions:
            tokens.extend(expansions)
    return tokens


def _slugify_source(source: str) -> str:
    stem = Path(source).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "corpus"


class _IndexedEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: RetrievedChunk
    tf: dict[str, float] = Field(default_factory=dict)
    norm: float = 1.0
    header_text: str = ""


class InMemoryHybridRetriever:
    """Zero-dependency in-memory hybrid retriever simulating dense + BM25 + RRF + rerank."""

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self._entries: list[_IndexedEntry] = []
        self._idf: dict[str, float] = {}
        self._build_index(chunks)

    def _build_index(self, chunks: Sequence[RetrievedChunk]) -> None:
        doc_freq: Counter[str] = Counter()
        tokenized_docs: list[tuple[RetrievedChunk, Counter[str], str]] = []
        for chunk in chunks:
            header_text = " ".join(chunk.metadata.section_path)
            tokens = _tokenize_text(f"{header_text} {chunk.text}")
            counts = Counter(tokens)
            for term in counts:
                doc_freq[term] += 1
            tokenized_docs.append((chunk, counts, header_text.lower()))

        total_docs = max(1, len(tokenized_docs))
        self._idf = {
            term: math.log((total_docs + 1.0) / (df + 0.5)) + 1.0 for term, df in doc_freq.items()
        }

        entries: list[_IndexedEntry] = []
        for chunk, counts, header_lower in tokenized_docs:
            total_terms = max(1, sum(counts.values()))
            tf = {term: cnt / total_terms for term, cnt in counts.items()}
            norm = math.sqrt(sum((w * self._idf.get(t, 1.0)) ** 2 for t, w in tf.items())) or 1.0
            entries.append(
                _IndexedEntry(
                    chunk=chunk,
                    tf=tf,
                    norm=norm,
                    header_text=header_lower,
                )
            )
        self._entries = entries

    @classmethod
    async def from_corpus_paths(
        cls,
        corpus_paths: Sequence[str],
        *,
        base_dir: Path | None = None,
    ) -> InMemoryHybridRetriever:
        """Load and semantically chunk Markdown files off the event loop."""
        root = base_dir or Path.cwd()

        def _load_sync() -> list[RetrievedChunk]:
            chunker = MarkdownHeaderChunker(
                ChunkerConfig(
                    mode=ChunkMode.SEMANTIC,
                    max_tokens_per_chunk=320,
                    min_chunk_chars=40,
                )
            )
            collected: list[RetrievedChunk] = []
            for raw_path in corpus_paths:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if not candidate.is_file():
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8")
                except OSError:
                    continue
                if not content.strip():
                    continue
                slug = _slugify_source(raw_path)
                doc_chunks = chunker.chunk(
                    content,
                    source=Path(raw_path).as_posix(),
                    doc_id=f"doc-{slug}",
                )
                for c in doc_chunks:
                    # Assign section-level doc_id so multi-section citations remain distinct.
                    section_doc_id = f"doc-{slug}-{c.metadata.chunk_index}"
                    collected.append(
                        RetrievedChunk(
                            text=c.text,
                            metadata=ChunkMetadata(
                                doc_id=section_doc_id,
                                source=c.metadata.source,
                                section_path=list(c.metadata.section_path),
                                chunk_index=c.metadata.chunk_index,
                            ),
                            score=0.5,
                            rank=len(collected),
                        )
                    )

            if not collected:
                fallback_chunks = chunker.chunk(
                    _DEFAULT_FALLBACK_CORPUS,
                    source="builtin://architecture.md",
                    doc_id="doc-builtin",
                )
                for c in fallback_chunks:
                    collected.append(
                        RetrievedChunk(
                            text=c.text,
                            metadata=ChunkMetadata(
                                doc_id=f"doc-builtin-{c.metadata.chunk_index}",
                                source=c.metadata.source,
                                section_path=list(c.metadata.section_path),
                                chunk_index=c.metadata.chunk_index,
                            ),
                            score=0.5,
                            rank=len(collected),
                        )
                    )
            return collected

        chunks = await asyncio.to_thread(_load_sync)
        return cls(chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._entries)

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        """Search indexed chunks using v1 dense-like or v2/v3 RRF + rerank scoring."""
        if not self._entries:
            return []

        await asyncio.sleep(0.004 if pipeline_version == "v1" else 0.008)
        query_tokens = _tokenize_text(query)
        query_counts = Counter(query_tokens)
        q_total = max(1, sum(query_counts.values()))
        q_tf = {term: cnt / q_total for term, cnt in query_counts.items()}
        q_norm = math.sqrt(sum((w * self._idf.get(t, 1.0)) ** 2 for t, w in q_tf.items())) or 1.0
        q_unique = set(query_counts.keys())

        dense_scores: list[tuple[int, float]] = []
        bm25_scores: list[tuple[int, float]] = []

        for idx, entry in enumerate(self._entries):
            dot = sum(
                q_tf[t] * entry.tf[t] * (self._idf.get(t, 1.0) ** 2) for t in q_tf if t in entry.tf
            )
            cosine = dot / (q_norm * entry.norm)
            dense_scores.append((idx, cosine))

            bm25 = 0.0
            for term in q_unique:
                tf_val = entry.tf.get(term, 0.0)
                if tf_val > 0.0:
                    idf = self._idf.get(term, 1.0)
                    bm25 += idf * (tf_val * 2.5) / (tf_val + 1.5)
            bm25_scores.append((idx, bm25))

        if pipeline_version == "v1":
            dense_ranked = sorted(dense_scores, key=lambda pair: pair[1], reverse=True)[:top_k]
            return [
                RetrievedChunk(
                    text=self._entries[idx].chunk.text,
                    metadata=self._entries[idx].chunk.metadata,
                    score=round(min(0.99, max(0.35, 0.55 + score * 0.45)), 4),
                    rank=rank,
                )
                for rank, (idx, score) in enumerate(dense_ranked)
            ]

        dense_top = sorted(dense_scores, key=lambda pair: pair[1], reverse=True)[:top_n]
        bm25_top = sorted(bm25_scores, key=lambda pair: pair[1], reverse=True)[:top_n]

        rrf_scores: dict[int, float] = {}
        for rank_pos, (idx, _) in enumerate(dense_top, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60.0 + rank_pos)
        for rank_pos, (idx, _) in enumerate(bm25_top, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60.0 + rank_pos)

        # Simulated cross-encoder rerank boost for exact query/header coverage
        reranked: list[tuple[int, float]] = []
        for idx, rrf in rrf_scores.items():
            entry = self._entries[idx]
            overlap_ratio = (
                len(q_unique & set(entry.tf.keys())) / len(q_unique) if q_unique else 0.0
            )
            header_hits = sum(1 for t in q_unique if len(t) > 1 and t in entry.header_text)
            header_boost = min(0.25, 0.08 * header_hits)
            combined = min(0.99, 0.45 + rrf * 6.0 + overlap_ratio * 0.35 + header_boost)
            reranked.append((idx, round(combined, 4)))

        reranked.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(
                text=self._entries[idx].chunk.text,
                metadata=self._entries[idx].chunk.metadata,
                score=score,
                rank=rank,
            )
            for rank, (idx, score) in enumerate(reranked[:top_k])
        ]


# ---------------------------------------------------------------------------
# Deterministic Demo LLM Router (Zero API Key / Streaming Support)
# ---------------------------------------------------------------------------


class DemoLLMRouter:
    """Deterministic dual-tier router producing cited streaming answers from retrieved context."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings

    def select_route(
        self,
        *,
        task_type: TaskType | str | None = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
    ) -> RouteDecision:
        """Mirror production Fast vs Strong routing policy deterministically."""
        _ = (task_type, query)
        resolved_force = (
            force_tier
            if isinstance(force_tier, RouteTier)
            else (RouteTier(force_tier.strip().lower()) if isinstance(force_tier, str) else None)
        )
        if resolved_force == RouteTier.STRONG or retry_count > 0:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name=self._settings.strong_llm_model,
                fallback_model_name=None,
                reason="Demo strong route for grounded multi-hop synthesis",
            )
        return RouteDecision(
            tier=RouteTier.FAST,
            model_name=self._settings.fast_llm_model,
            fallback_model_name=self._settings.fast_llm_fallback_model,
            reason="Demo fast route for routing, grading, and v1 baseline",
        )

    @staticmethod
    def _extract_question_and_contexts(messages: Any) -> tuple[str, list[tuple[str, str]]]:
        user_msg = ""
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            for item in messages:
                if isinstance(item, tuple) and len(item) >= 2 and item[0] == "user":
                    user_msg = str(item[1])
                    break
        if not user_msg:
            user_msg = str(messages)

        if "\n\nContext:\n" in user_msg:
            q_part, ctx_part = user_msg.split("\n\nContext:\n", 1)
        else:
            q_part, ctx_part = user_msg, ""
        question = q_part.removeprefix("Question:").strip()

        contexts: list[tuple[str, str]] = []
        for match in _CONTEXT_BLOCK_RE.finditer(ctx_part):
            doc_id = match.group(1).strip()
            body = re.sub(r"\s+", " ", match.group(4)).strip()
            if doc_id and body:
                contexts.append((doc_id, body))
        return question, contexts

    @classmethod
    def _synthesize_answer(cls, messages: Any, tier: RouteTier) -> str:
        question, contexts = cls._extract_question_and_contexts(messages)
        if not contexts:
            return (
                "No supporting context chunks were retrieved from the demo knowledge base "
                f"for query: {question}."
            )

        q_lower = question.lower()
        q_words = set(re.findall(r"[a-z0-9_.-]+|[\u4e00-\u9fff]+", q_lower))
        doc_ids = [doc_id for doc_id, _ in contexts]
        primary_id = doc_ids[0]
        secondary_id = doc_ids[1] if len(doc_ids) > 1 else primary_id

        # Extract concise evidence snippet from top retrieved chunks
        lead_snippets: list[str] = []
        limit = 1 if tier == RouteTier.FAST else min(2, len(contexts))
        for doc_id, body in contexts[:limit]:
            clipped = body[:220].rstrip()
            if len(body) > 220:
                clipped = f"{clipped}..."
            lead_snippets.append(f"{clipped} [{doc_id}]")

        if q_words & {"v1", "v2", "v3", "infinite", "langgraph", "loop", "loops"}:
            return (
                "In this engine, v1 uses fixed chunking with Qdrant INT8 dense-only search; "
                "v2 adds Markdown header semantic chunking, BM25 sparse + Fusion.RRF, and local "
                "FP16 bge-reranker-v2-m3 reranking; v3 wraps retrieval in a LangGraph StateGraph "
                "with Redis 8 semantic caching (cosine >= 0.92), bounded retries (max_retries=2), "
                f"and AsyncSqliteSaver checkpoints [{primary_id}]. "
                f"Evidence: {lead_snippets[-1]}"
            )

        if q_words & {"bm25", "rrf", "rerank", "reranking", "qdrant", "dense", "sparse"}:
            if tier == RouteTier.FAST:
                return (
                    "The v1 baseline stores 1024-dim BAAI/bge-m3 dense vectors in Qdrant with "
                    "INT8 scalar quantization (always_ram=True) for fast single-step ANN "
                    f"retrieval [{primary_id}]. Evidence: {lead_snippets[0]}"
                )
            return (
                "Hybrid retrieval prefetches 1024-dim INT8 dense vectors (BAAI/bge-m3) alongside "
                "BM25 sparse vectors in Qdrant, combines candidates with server-side Fusion.RRF, "
                f"and reranks to Top-K via FP16 BAAI/bge-reranker-v2-m3 [{primary_id}]. "
                f"Evidence: {lead_snippets[-1]}"
            )

        if (q_words & {"vram", "3060", "gpu", "6gb", "显存"}) or "6 gb" in q_lower:
            return (
                "Under the RTX 3060 6 GB VRAM hard budget, zero local LLMs are deployed: "
                "only FP16 BAAI/bge-m3 (~1.1 GB) and BAAI/bge-reranker-v2-m3 (~1.1 GB) run on "
                f"CUDA (~2.3 GB total), while LLM inference routes to cloud APIs [{primary_id}]. "
                f"Details: {lead_snippets[-1]}"
            )

        if q_words & {"ragas", "benchmark", "benchmarks", "eval", "evals", "sla", "p95", "评测"}:
            return (
                "Ragas v0.2+ benchmarks show Overall Score rising from 0.6759 (v1) to 0.8554 (v2) "
                "and 0.9950 (v3, Context Recall 1.0000, Faithfulness 0.9851), while concurrent load "
                f"testing passes the P95 < 1800 ms SLA at 678.77 ms [{primary_id}]. "
                f"Context: {lead_snippets[-1]}"
            )

        if tier == RouteTier.FAST:
            return f"Based on baseline dense retrieval: {lead_snippets[0]}"

        joined = " | ".join(lead_snippets)
        if secondary_id != primary_id:
            return f"Grounded synthesis across retrieved documentation [{primary_id}] [{secondary_id}]: {joined}"
        return f"Grounded synthesis from project documentation [{primary_id}]: {joined}"

    async def ainvoke(
        self,
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> LLMGenerationResult:
        _ = temperature
        decision = self.select_route(
            task_type=task_type,
            retry_count=retry_count,
            force_tier=force_tier,
        )
        start = time.perf_counter()
        await asyncio.sleep(0.012 if decision.tier == RouteTier.FAST else 0.02)
        content = self._synthesize_answer(messages, decision.tier)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        prompt_tokens = max(16, len(str(messages)) // 4)
        completion_tokens = max(8, len(content) // 4)
        update_request_metrics(
            route_tier=decision.tier.value,
            model_name=decision.model_name,
            ttft_ms=elapsed_ms,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return LLMGenerationResult(
            content=content,
            route_tier=decision.tier,
            model_name=decision.model_name,
            used_fallback=False,
            mocked=True,
            ttft_ms=elapsed_ms,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def astream(
        self,
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        _ = temperature
        decision = self.select_route(
            task_type=task_type,
            retry_count=retry_count,
            force_tier=force_tier,
        )
        start = time.perf_counter()
        await asyncio.sleep(0.01 if decision.tier == RouteTier.FAST else 0.016)
        content = self._synthesize_answer(messages, decision.tier)
        words = content.split(" ")
        ttft_ms: float | None = None
        try:
            for idx, word in enumerate(words):
                piece = f"{word} " if idx < len(words) - 1 else word
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - start) * 1000.0, 2)
                else:
                    await asyncio.sleep(0.004)
                yield piece
        finally:
            total_ms = round((time.perf_counter() - start) * 1000.0, 2)
            update_request_metrics(
                route_tier=decision.tier.value,
                model_name=decision.model_name,
                ttft_ms=ttft_ms,
                latency_ms=total_ms,
                prompt_tokens=max(16, len(str(messages)) // 4),
                completion_tokens=max(8, len(content) // 4),
            )

    async def ainvoke_structured(
        self,
        schema: type[Any],
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> Any:
        _ = (messages, task_type, force_tier, retry_count, temperature)
        await asyncio.sleep(0.003)
        if schema is DocumentGradeOutput:
            return DocumentGradeOutput(relevant=True, reason="Verified relevant demo corpus chunk")
        if schema is HallucinationGradeOutput:
            return HallucinationGradeOutput(
                grounded=True,
                reason="Answer grounded in retrieved demo corpus citations",
            )
        return schema.model_validate({})


# ---------------------------------------------------------------------------
# ServiceContainer (Lifespan Dependency Injection & Dual-Mode Auto-Wiring)
# ---------------------------------------------------------------------------


class ServiceContainer:
    """Manages startup wiring and graceful shutdown for RAG infrastructure components."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        mode_override: ResolvedRuntimeMode | None = None,
        checkpoint_db: str | Path | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self._mode_override = mode_override
        self._checkpoint_db_override = checkpoint_db

        self.mode: ResolvedRuntimeMode = "demo"
        self.retriever: Any = None
        self.router: Any = None
        self.semantic_cache: Any = None
        self.workflow: Any = None
        self.checkpointer: Any = None
        self.component_status: dict[str, str] = {}
        self.singleflight = SingleFlight()

        self._exit_stack: AsyncExitStack | None = None
        self._owned_qdrant_client: AsyncQdrantClient | None = None
        self._initialized: bool = False

    async def _resolve_mode(self) -> tuple[ResolvedRuntimeMode, str]:
        if self._mode_override is not None:
            return self._mode_override, "explicit container mode_override"

        configured = self.settings.runtime_mode
        if configured in ("live", "demo", "test"):
            return configured, f"explicit settings.runtime_mode={configured}"

        if self.settings.mock_llm_stream:
            return "demo", "MOCK_LLM_STREAM=true forces offline demo mode"

        if not has_valid_cloud_api_keys(self.settings):
            return "demo", "Cloud LLM API keys missing or template placeholders"

        if not await probe_qdrant_ready(self.settings.qdrant_url):
            return "demo", f"Qdrant unreachable at {self.settings.qdrant_url}"

        return "live", "Cloud API keys and Qdrant service are available"

    def _resolve_checkpoint_target(self, mode: ResolvedRuntimeMode) -> str | Path:
        if self._checkpoint_db_override is not None:
            return self._checkpoint_db_override
        if mode == "test":
            return ":memory:"
        return self.settings.langgraph_checkpoint_db

    async def _open_checkpointer(self, target: str | Path) -> Any:
        assert self._exit_stack is not None
        try:
            return await self._exit_stack.enter_async_context(open_async_sqlite_saver(target))
        except Exception as exc:
            logger.warning(
                "AsyncSqliteSaver failed to open target path, falling back to :memory:",
                checkpoint_db=str(target),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return await self._exit_stack.enter_async_context(open_async_sqlite_saver(":memory:"))

    async def _init_demo_or_test(self, mode: ResolvedRuntimeMode) -> None:
        self.retriever = await InMemoryHybridRetriever.from_corpus_paths(
            self.settings.demo_corpus_paths
        )
        self.router = DemoLLMRouter(self.settings)
        self.semantic_cache = InMemorySemanticCache(
            settings=self.settings,
            similarity_threshold=self.settings.cache_similarity_threshold,
            ttl_seconds=self.settings.cache_ttl_seconds,
        )

        checkpoint_target = self._resolve_checkpoint_target(mode)
        self.checkpointer = await self._open_checkpointer(checkpoint_target)

        graph_deps = GraphDependencies(
            retriever=self.retriever,
            router=self.router,
            cache=self.semantic_cache,
            top_k=self.settings.retrieval_top_k,
            top_n=self.settings.retrieval_top_n,
            max_retries=self.settings.max_retries,
        )
        self.workflow = AgenticRAGWorkflow(
            graph_deps,
            db_path=checkpoint_target,
            max_retries=self.settings.max_retries,
            checkpointer=self.checkpointer,
        )
        self.component_status = {
            "runtime_mode": mode,
            "retriever": f"ready ({mode})",
            "router": f"ready ({mode})",
            "semantic_cache": f"ready ({mode})",
            "workflow": f"ready ({mode})",
            "checkpointer": f"ready ({mode})",
        }

    async def _init_live(self) -> None:
        dense_embedder: BGEDenseEmbedder | None = None
        retriever_status = "ready (live)"
        router_status = "ready (live)"
        cache_status = "ready (live)"

        if is_cuda_available() and await probe_qdrant_ready(self.settings.qdrant_url):
            qdrant_client: AsyncQdrantClient | None = None
            try:
                qdrant_client = AsyncQdrantClient(url=self.settings.qdrant_url)
                dense_embedder = BGEDenseEmbedder(model_name=self.settings.embed_model_name)
                sparse_embedder = BM25SparseEmbedder()
                reranker = BGEReranker(model_name=self.settings.reranker_model_name)
                store = HybridVectorStore(
                    client=qdrant_client,
                    dense_embedder=dense_embedder,
                    sparse_embedder=sparse_embedder,
                    reranker=reranker,
                    collection_name=self.settings.qdrant_collection_name,
                )
                await store.ensure_collection()
                self._owned_qdrant_client = qdrant_client
                self.retriever = store
            except Exception as exc:
                logger.warning(
                    "Live HybridVectorStore initialization failed; falling back to in-memory retriever",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                if qdrant_client is not None:
                    try:
                        await qdrant_client.close()
                    except Exception:
                        pass
                dense_embedder = None
                self.retriever = await InMemoryHybridRetriever.from_corpus_paths(
                    self.settings.demo_corpus_paths
                )
                retriever_status = "ready (demo-fallback)"
        else:
            logger.warning(
                "Live retrieval prerequisites unavailable (CUDA or Qdrant); using demo retriever fallback",
                cuda_available=is_cuda_available(),
                qdrant_url=self.settings.qdrant_url,
            )
            self.retriever = await InMemoryHybridRetriever.from_corpus_paths(
                self.settings.demo_corpus_paths
            )
            retriever_status = "ready (demo-fallback)"

        if has_valid_cloud_api_keys(self.settings) and not self.settings.mock_llm_stream:
            try:
                self.router = LLMRouter(settings=self.settings)
            except Exception as exc:
                logger.warning(
                    "Live LLMRouter initialization failed; falling back to DemoLLMRouter",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                self.router = DemoLLMRouter(self.settings)
                router_status = "ready (demo-fallback)"
        else:
            self.router = DemoLLMRouter(self.settings)
            router_status = "ready (demo-fallback)"

        if dense_embedder is not None and await probe_redis_ready(self.settings.redis_url):
            live_cache = SemanticCache.from_url(
                dense_embedder,
                redis_url=self.settings.redis_url,
                settings=self.settings,
            )
            try:
                index_ok = await live_cache.ensure_index()
                if index_ok:
                    self.semantic_cache = live_cache
                else:
                    await live_cache.aclose()
                    raise RuntimeError("RediSearch index initialization returned False")
            except Exception as exc:
                logger.warning(
                    "Live Redis SemanticCache unavailable; falling back to InMemorySemanticCache",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                try:
                    await live_cache.aclose()
                except Exception:
                    pass
                self.semantic_cache = InMemorySemanticCache(
                    settings=self.settings,
                    similarity_threshold=self.settings.cache_similarity_threshold,
                    ttl_seconds=self.settings.cache_ttl_seconds,
                )
                cache_status = "ready (in-memory-fallback)"
        else:
            self.semantic_cache = InMemorySemanticCache(
                settings=self.settings,
                similarity_threshold=self.settings.cache_similarity_threshold,
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
            cache_status = "ready (in-memory-fallback)"

        checkpoint_target = self._resolve_checkpoint_target("live")
        self.checkpointer = await self._open_checkpointer(checkpoint_target)

        graph_deps = GraphDependencies(
            retriever=self.retriever,
            router=self.router,
            cache=self.semantic_cache,
            top_k=self.settings.retrieval_top_k,
            top_n=self.settings.retrieval_top_n,
            max_retries=self.settings.max_retries,
        )
        self.workflow = AgenticRAGWorkflow(
            graph_deps,
            db_path=checkpoint_target,
            max_retries=self.settings.max_retries,
            checkpointer=self.checkpointer,
        )
        self.component_status = {
            "runtime_mode": "live",
            "retriever": retriever_status,
            "router": router_status,
            "semantic_cache": cache_status,
            "workflow": "ready (live)",
            "checkpointer": "ready (live)",
        }

    async def initialize(self) -> None:
        """Resolve runtime mode and wire all RAG components safely."""
        if self._initialized:
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        try:
            resolved_mode, reason = await self._resolve_mode()
            self.mode = resolved_mode
            if resolved_mode == "live":
                logger.info(
                    "ServiceContainer activating live mode",
                    runtime_mode=resolved_mode,
                    reason=reason,
                )
                await self._init_live()
            else:
                logger.warning(
                    "ServiceContainer activating offline mode",
                    runtime_mode=resolved_mode,
                    reason=reason,
                )
                await self._init_demo_or_test(resolved_mode)
            self._initialized = True
        except Exception:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Close AsyncSqliteSaver, SemanticCache, and Qdrant client connections."""
        if self.semantic_cache is not None:
            aclose_fn = getattr(self.semantic_cache, "aclose", None)
            if callable(aclose_fn):
                try:
                    await aclose_fn()
                except Exception as exc:
                    logger.warning(
                        "Error closing semantic_cache",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            self.semantic_cache = None

        if self._owned_qdrant_client is not None:
            try:
                await self._owned_qdrant_client.close()
            except Exception as exc:
                logger.warning(
                    "Error closing AsyncQdrantClient",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            self._owned_qdrant_client = None

        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.warning(
                    "Error closing ServiceContainer AsyncExitStack",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            self._exit_stack = None

        self.checkpointer = None
        self.workflow = None
        self._initialized = False


__all__: Sequence[str] = [
    "DemoLLMRouter",
    "InMemoryHybridRetriever",
    "ResolvedRuntimeMode",
    "ServiceContainer",
    "has_valid_cloud_api_keys",
    "is_cuda_available",
    "is_placeholder_api_key",
    "probe_qdrant_ready",
    "probe_redis_ready",
]
