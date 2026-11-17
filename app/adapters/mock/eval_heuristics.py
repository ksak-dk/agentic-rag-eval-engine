"""
app/adapters/mock/eval_heuristics.py
====================================
Centralized offline simulation and heuristic evaluation adapters extracted from
``evals/dataset_gen.py``, ``evals/run_ragas.py``, and ``evals/load_test.py``.

Provides:
- Deterministic single-hop & multi-hop offline dataset candidate builders
- Offline Ragas lexical heuristic scorer & v1/v2/v3 pipeline evolution runner
- Lightweight ASGI mock adapters (``LoadTestRetriever``, ``LoadTestSemanticCache``,
  ``LoadTestRouter``) satisfying ``app.ports`` contracts for high-concurrency load tests
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from ragas.dataset_schema import SingleTurnSample

from app.cache.semantic_cache import CachedResponse, SingleFlight
from app.core.config import LogLevel, Settings
from app.core.config import PipelineVersion as ConfigPipelineVersion
from app.core.router import LLMGenerationResult, RouteDecision, RouteTier, TaskType
from app.graph.nodes import DocumentGradeOutput, HallucinationGradeOutput
from app.retrieval.chunker import ChunkMetadata, DocumentChunk
from app.retrieval.hybrid_store import RetrievedChunk

if TYPE_CHECKING:
    from evals.dataset_gen import EvalSample

_WORD_RE: re.Pattern[str] = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")

DEFAULT_LOAD_TEST_QUERIES: tuple[str, ...] = (
    "How does Qdrant INT8 quantization reduce memory footprint?",
    "How are dense bge-m3 and sparse BM25 vectors fused via RRF?",
    "Why does bge-reranker-v2-m3 run in FP16 on RTX 3060?",
    "How does LangGraph enforce max_retries=2 on irrelevant documents?",
    "How does Redis 8 semantic cache isolate v1, v2, and v3 namespaces?",
    "What triggers Groq to Gemini Flash-Lite automatic fallback?",
    "How does AsyncSqliteSaver persist LangGraph conversation checkpoints?",
    "What SSE event channels does /api/v1/chat emit in streaming mode?",
)

BENCHMARK_CORPUS: tuple[tuple[str, str, str], ...] = (
    (
        "doc-qdrant",
        "docs/retrieval.md",
        "# Hybrid Vector Retrieval\n\n"
        "Qdrant stores 1024-dimensional BAAI/bge-m3 dense vectors with INT8 scalar "
        "quantization (always_ram=True) alongside in-memory BM25 sparse vectors. "
        "Prefetch top-20 dense and sparse candidates are merged via server-side "
        "Fusion.RRF and reranked in FP16 by BAAI/bge-reranker-v2-m3 to top-5 chunks.",
    ),
    (
        "doc-langgraph",
        "docs/orchestration.md",
        "# Self-Correcting LangGraph Workflow\n\n"
        "LangGraph StateGraph executes Redis 8 semantic cache lookup (cosine >= 0.92), "
        "fast-vs-strong routing, hybrid retrieval, document relevance grading, bounded "
        "query rewriting (max_retries=2), strong generation with [doc_id] citations, "
        "and faithfulness verification persisted via AsyncSqliteSaver.",
    ),
    (
        "doc-serving",
        "docs/serving.md",
        "# Dual-Channel FastAPI SSE Serving\n\n"
        "FastAPI exposes /api/v1/chat over sse-starlette EventSourceResponse, streaming "
        "token events on the text channel followed by structured citations, "
        "RequestMetrics telemetry, and done events within a P95 < 1.8s latency target.",
    ),
)


# ---------------------------------------------------------------------------
# Shared Numeric & Lexical Helpers
# ---------------------------------------------------------------------------


def clamp_score(raw: Any) -> float:
    """Clamp numeric scores safely into ``[0.0, 1.0]``, mapping NaN/Inf to 0.0."""
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return round(max(0.0, min(1.0, val)), 4)


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    """Compute linear-interpolated percentile rounded to 2 decimal places."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = (len(ordered) - 1) * (percentile / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return round(ordered[lo], 2)
    interpolated = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return round(interpolated, 2)


def tokenize_lexical(text: str) -> set[str]:
    """Tokenize bilingual text into lowercase lexical terms."""
    return {tok.lower() for tok in _WORD_RE.findall(text) if tok}


def truncate_words(text: str, ratio: float) -> str:
    """Truncate whitespace-delimited words to a proportional prefix."""
    words = text.split()
    if not words:
        return ""
    keep = max(1, math.floor(len(words) * ratio))
    return " ".join(words[:keep])


# ---------------------------------------------------------------------------
# Offline Dataset Selection & Deterministic Builder
# ---------------------------------------------------------------------------


def select_single_hop_candidates(
    chunks: Sequence[DocumentChunk],
    count: int,
    rng: random.Random,
) -> list[DocumentChunk]:
    """Select eligible document chunks for single-hop evaluation synthesis."""
    eligible = [chunk for chunk in chunks if len(chunk.text.strip()) >= 20]
    if not eligible or count <= 0:
        return []
    ordered = list(eligible)
    rng.shuffle(ordered)
    if count <= len(ordered):
        return ordered[:count]
    selected: list[DocumentChunk] = []
    while len(selected) < count:
        selected.extend(ordered)
    return selected[:count]


def select_multi_hop_candidates(
    chunks: Sequence[DocumentChunk],
    count: int,
    rng: random.Random,
    *,
    chunks_per_group: int = 2,
) -> list[list[DocumentChunk]]:
    """Pair distinct chunks (preferring cross-document) for multi-hop synthesis."""
    eligible = [chunk for chunk in chunks if len(chunk.text.strip()) >= 20]
    if len(eligible) < 2 or count <= 0:
        return []

    group_size = max(2, min(chunks_per_group, len(eligible)))
    cross_doc_pairs: list[list[DocumentChunk]] = []
    same_doc_pairs: list[list[DocumentChunk]] = []

    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            first, second = eligible[i], eligible[j]
            if first.metadata.doc_id != second.metadata.doc_id:
                cross_doc_pairs.append([first, second])
            elif first.metadata.section_path != second.metadata.section_path or (
                abs(first.metadata.chunk_index - second.metadata.chunk_index) > 1
            ):
                same_doc_pairs.append([first, second])

    rng.shuffle(cross_doc_pairs)
    rng.shuffle(same_doc_pairs)
    pool = cross_doc_pairs + same_doc_pairs

    if not pool:
        pool = [[eligible[i], eligible[(i + 1) % len(eligible)]] for i in range(len(eligible) - 1)]

    if group_size > 2 and len(eligible) >= group_size:
        expanded: list[list[DocumentChunk]] = []
        for pair in pool:
            extras = [c for c in eligible if c not in pair]
            rng.shuffle(extras)
            expanded.append([*pair, *extras[: group_size - 2]])
        pool = expanded

    if count <= len(pool):
        return pool[:count]
    selected: list[list[DocumentChunk]] = []
    while len(selected) < count:
        selected.extend(pool)
    return selected[:count]


def build_offline_sample_dicts(
    chunks: Sequence[DocumentChunk],
    *,
    single_hop_count: int = 2,
    multi_hop_count: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build deterministic offline sample dictionaries without external LLM calls."""
    rng = random.Random(seed)
    payloads: list[dict[str, Any]] = []

    for idx, chunk in enumerate(select_single_hop_candidates(chunks, single_hop_count, rng)):
        topic = " > ".join(chunk.metadata.section_path) or chunk.metadata.source
        snippet = chunk.text.strip()
        payloads.append(
            {
                "sample_id": f"offline-single-{idx}",
                "question": f"What does {topic} specify in document {chunk.metadata.doc_id}?",
                "ground_truth": snippet[:240],
                "reference_contexts": [snippet],
                "hop_type": "single_hop",
                "source_doc_ids": [chunk.metadata.doc_id],
            }
        )

    for idx, group in enumerate(select_multi_hop_candidates(chunks, multi_hop_count, rng)):
        doc_ids = [c.metadata.doc_id for c in group]
        contexts = [c.text.strip() for c in group]
        combined_truth = " ".join(c.text.strip()[:120] for c in group)
        payloads.append(
            {
                "sample_id": f"offline-multi-{idx}",
                "question": (
                    f"How do the specifications in {' and '.join(dict.fromkeys(doc_ids))} "
                    "relate to each other?"
                ),
                "ground_truth": combined_truth,
                "reference_contexts": contexts,
                "hop_type": "multi_hop",
                "source_doc_ids": doc_ids,
            }
        )

    return payloads


def build_offline_dataset(
    chunks: Sequence[DocumentChunk],
    *,
    single_hop_count: int = 2,
    multi_hop_count: int = 2,
    seed: int = 42,
) -> list[EvalSample]:
    """Deterministically construct offline ``EvalSample`` items without any LLM call."""
    from evals.dataset_gen import EvalSample

    raw = build_offline_sample_dicts(
        chunks,
        single_hop_count=single_hop_count,
        multi_hop_count=multi_hop_count,
        seed=seed,
    )
    return [EvalSample.model_validate(item) for item in raw]


# ---------------------------------------------------------------------------
# Offline Ragas Heuristic Scorer & Simulated Pipeline Runner
# ---------------------------------------------------------------------------


async def heuristic_ragas_scorer(
    sample: SingleTurnSample,
    pipeline_version: str,
) -> dict[str, float]:
    """Deterministic lexical scorer mimicking Ragas metrics for offline testing."""
    _ = pipeline_version
    retrieved_list = list(sample.retrieved_contexts or [])
    reference_list = list(sample.reference_contexts or [])
    if not reference_list and sample.reference:
        reference_list = [sample.reference]

    retrieved_union: set[str] = set()
    for ctx in retrieved_list:
        retrieved_union.update(tokenize_lexical(ctx))

    ref_union: set[str] = set()
    for ref in reference_list:
        ref_union.update(tokenize_lexical(ref))

    if not reference_list or not retrieved_union:
        context_recall = 0.0
    else:
        recalls: list[float] = []
        for ref in reference_list:
            ref_tokens = tokenize_lexical(ref)
            if not ref_tokens:
                continue
            recalls.append(len(ref_tokens & retrieved_union) / len(ref_tokens))
        context_recall = sum(recalls) / len(recalls) if recalls else 0.0

    resp_tokens = tokenize_lexical(sample.response or "")
    if not resp_tokens or not retrieved_union:
        faithfulness = 0.0
    else:
        faithfulness = len(resp_tokens & retrieved_union) / len(resp_tokens)

    if not retrieved_list or not ref_union:
        context_precision = 0.0
    else:
        weighted_hits = 0.0
        relevant_count = 0
        for rank_idx, ctx in enumerate(retrieved_list):
            ctx_tokens = tokenize_lexical(ctx)
            if not ctx_tokens:
                continue
            overlap_ratio = len(ctx_tokens & ref_union) / len(ctx_tokens)
            if overlap_ratio >= 0.35:
                relevant_count += 1
                weighted_hits += relevant_count / (rank_idx + 1)
        context_precision = weighted_hits / relevant_count if relevant_count > 0 else 0.0

    return {
        "context_recall": clamp_score(context_recall),
        "faithfulness": clamp_score(faithfulness),
        "context_precision": clamp_score(context_precision),
    }


async def offline_simulated_runner(
    sample: Any,
    pipeline_version: str,
) -> dict[str, Any]:
    """Deterministic offline runner modeling v1 < v2 < v3 retrieval & grounding gains."""
    ref_contexts = list(sample.reference_contexts)
    first_ctx = ref_contexts[0] if ref_contexts else ""

    if pipeline_version == "v1":
        retrieved = [
            truncate_words(first_ctx, 0.45),
            "Legacy system telemetry note unrelated to vector index quantization.",
        ]
        answer = (
            f"{truncate_words(sample.ground_truth, 0.65)} "
            "Additionally unverified legacy defaults apply indefinitely."
        )
        return {
            "answer": answer.strip(),
            "retrieved_contexts": [c for c in retrieved if c],
            "latency_ms": 410.0,
            "ttft_ms": 135.0,
            "rewrite_count": 0,
            "cache_hit": False,
        }

    if pipeline_version == "v2":
        if len(ref_contexts) >= 2:
            retrieved = [
                first_ctx,
                "General overview paragraph without secondary hop details.",
            ]
            answer = truncate_words(sample.ground_truth, 0.8)
        else:
            retrieved = [
                first_ctx,
                "Secondary non-essential documentation note.",
            ]
            answer = sample.ground_truth
        return {
            "answer": answer.strip(),
            "retrieved_contexts": retrieved,
            "latency_ms": 680.0,
            "ttft_ms": 185.0,
            "rewrite_count": 0,
            "cache_hit": False,
        }

    return {
        "answer": sample.ground_truth,
        "retrieved_contexts": ref_contexts,
        "latency_ms": 940.0 if sample.hop_type == "multi_hop" else 760.0,
        "ttft_ms": 215.0,
        "rewrite_count": 1 if sample.hop_type == "multi_hop" else 0,
        "cache_hit": False,
    }


# ---------------------------------------------------------------------------
# Load-Test Mock Adapters (Implementing app.ports Protocols)
# ---------------------------------------------------------------------------


class LoadTestRetriever:
    """Async retriever adapter simulating Qdrant INT8 + RRF + FP16 rerank latency."""

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        _ = (query, top_n)
        await asyncio.sleep(0.005 if pipeline_version == "v1" else 0.012)
        chunks = [
            RetrievedChunk(
                text=(
                    "Qdrant fuses 1024-dim INT8 dense vectors and BM25 sparse vectors via RRF "
                    "before FP16 bge-reranker-v2-m3 scoring."
                ),
                metadata=ChunkMetadata(
                    doc_id="doc-qdrant",
                    source="docs/retrieval.md",
                    section_path=["Hybrid Vector Retrieval"],
                    chunk_index=0,
                ),
                score=0.95,
                rank=0,
            ),
            RetrievedChunk(
                text=(
                    "LangGraph enforces max_retries=2 and persists checkpoints in "
                    "AsyncSqliteSaver while Redis 8 caches cosine >= 0.92 hits."
                ),
                metadata=ChunkMetadata(
                    doc_id="doc-langgraph",
                    source="docs/orchestration.md",
                    section_path=["Self-Correcting LangGraph Workflow"],
                    chunk_index=0,
                ),
                score=0.91,
                rank=1,
            ),
        ]
        return chunks[:top_k]


class LoadTestSemanticCache:
    """In-memory async semantic cache adapter simulating Redis 8 HNSW lookup & TTL write."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CachedResponse] = {}
        self.singleflight = SingleFlight()

    async def get(
        self,
        query: str,
        *,
        namespace: str | None = "v3",
        threshold: float | None = None,
    ) -> CachedResponse | None:
        _ = threshold
        active_ns = namespace or "v3"
        await asyncio.sleep(0.0005)
        return self._entries.get((active_ns, query.strip()))

    async def set(
        self,
        query: str,
        response: str,
        *,
        namespace: str | None = "v3",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        _ = ttl_seconds
        active_ns = namespace or "v3"
        await asyncio.sleep(0.0005)
        cleaned = query.strip()
        if not cleaned:
            return None
        self._entries[(active_ns, cleaned)] = CachedResponse(
            query=cleaned,
            response=response,
            namespace=active_ns,
            similarity=0.99,
            metadata=dict(metadata or {}),
        )
        return f"rag:cache:{active_ns}:loadtest"

    async def invalidate_namespace(self, namespace: str | None = "v3") -> int:
        active_ns = namespace or "v3"
        keys = [k for k in self._entries if k[0] == active_ns]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)

    async def aclose(self) -> None:
        self._entries.clear()

    lookup = get
    put = set
    store = set


class LoadTestRouter:
    """Deterministic async cloud LLM router adapter modeling fast/strong TTFT and streaming."""

    def select_route(
        self,
        *,
        task_type: TaskType | str | None = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
    ) -> RouteDecision:
        _ = (task_type, query)
        if force_tier == RouteTier.STRONG or retry_count > 0:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name="gemini-2.5-flash",
                fallback_model_name=None,
                reason="Load test strong route",
            )
        return RouteDecision(
            tier=RouteTier.FAST,
            model_name="llama-3.1-8b-instant",
            fallback_model_name="gemini-2.5-flash-lite",
            reason="Load test fast route",
        )

    async def ainvoke(
        self,
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMGenerationResult:
        _ = (messages, task_type, query, retry_count, temperature, kwargs)
        tier = force_tier if isinstance(force_tier, RouteTier) else RouteTier.STRONG
        delay = 0.015 if tier == RouteTier.FAST else 0.022
        await asyncio.sleep(delay)
        return LLMGenerationResult(
            content=(
                "Hybrid retrieval combines INT8 dense and BM25 sparse vectors [doc-qdrant] "
                "inside a bounded LangGraph workflow [doc-langgraph]."
            ),
            route_tier=tier,
            model_name=("llama-3.1-8b-instant" if tier == RouteTier.FAST else "gemini-2.5-flash"),
            mocked=True,
            ttft_ms=round(delay * 1000.0, 2),
            latency_ms=round(delay * 1000.0, 2),
            prompt_tokens=64,
            completion_tokens=24,
        )

    async def astream(
        self,
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        _ = (messages, task_type, query, retry_count, temperature, kwargs)
        tier = force_tier if isinstance(force_tier, RouteTier) else RouteTier.STRONG
        await asyncio.sleep(0.012 if tier == RouteTier.FAST else 0.018)
        for token in (
            "Hybrid retrieval combines ",
            "INT8 dense and BM25 sparse vectors [doc-qdrant] ",
            "inside a bounded LangGraph workflow [doc-langgraph].",
        ):
            yield token
            await asyncio.sleep(0.002)

    async def ainvoke_structured(
        self,
        schema: type[Any],
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        _ = (messages, task_type, query, force_tier, retry_count, temperature, kwargs)
        await asyncio.sleep(0.002)
        if schema is DocumentGradeOutput:
            return DocumentGradeOutput(relevant=True, reason="Mock relevant chunk")
        if schema is HallucinationGradeOutput:
            return HallucinationGradeOutput(grounded=True, reason="Mock grounded answer")
        return schema.model_validate({})


def build_mock_load_test_app(
    *,
    checkpoint_db: Path,
    checkpointer: Any | None = None,
    mock_llm_stream: bool = True,
) -> FastAPI:
    """Build an in-memory FastAPI ASGI instance wired with deterministic load-test adapters."""
    from app.main import create_app

    settings = Settings(
        groq_api_key="load-test-groq-key",
        google_api_key="load-test-google-key",
        pipeline_version=ConfigPipelineVersion.V3,
        runtime_mode="test",
        mock_llm_stream=mock_llm_stream,
        langgraph_checkpoint_db=checkpoint_db,
        log_level=LogLevel.WARNING,
    )
    return create_app(
        settings=settings,
        retriever=LoadTestRetriever(),
        router=LoadTestRouter(),
        cache=LoadTestSemanticCache(),
        checkpointer=checkpointer,
        checkpoint_db=checkpoint_db,
        testing=True,
    )


__all__: Sequence[str] = [
    "BENCHMARK_CORPUS",
    "DEFAULT_LOAD_TEST_QUERIES",
    "LoadTestRetriever",
    "LoadTestRouter",
    "LoadTestSemanticCache",
    "build_mock_load_test_app",
    "build_offline_dataset",
    "build_offline_sample_dicts",
    "clamp_score",
    "compute_percentile",
    "heuristic_ragas_scorer",
    "offline_simulated_runner",
    "select_multi_hop_candidates",
    "select_single_hop_candidates",
    "tokenize_lexical",
    "truncate_words",
]
