from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.adapters.mock import (
    DemoLLMRouter,
    InMemoryHybridRetriever,
    LoadTestRetriever,
    LoadTestRouter,
    LoadTestSemanticCache,
    build_offline_dataset,
    heuristic_ragas_scorer,
    offline_simulated_runner,
)
from app.cache.semantic_cache import InMemorySemanticCache, SemanticCache
from app.core.config import LogLevel, Settings
from app.core.router import LLMRouter, RouteTier, TaskType
from app.graph.nodes import DocumentGradeOutput, HallucinationGradeOutput, QueryRewriteOutput
from app.ports import LLMRouterPort, RagasEvaluatorPort, RetrieverPort, SemanticCachePort
from app.retrieval.chunker import make_chunker
from app.retrieval.hybrid_store import DENSE_DIM, HybridVectorStore
from evals.load_test import AsyncLoadTester
from evals.run_ragas import RagasPipelineEvaluator


class _DummyDenseEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * DENSE_DIM for _ in texts]

    def get_vram_usage_mb(self) -> float:
        return 0.0


class _DummySparseEmbedder:
    def embed(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [([1], [1.0]) for _ in texts]


class _DummyReranker:
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        _ = query
        return [1.0 for _ in texts]

    def get_vram_usage_mb(self) -> float:
        return 0.0


def test_ports_runtime_protocol_conformance() -> None:
    """Verify real infrastructure classes and mock adapters satisfy @runtime_checkable ports."""
    settings = Settings(
        groq_api_key="gsk_test_key_12345",
        google_api_key="AIza_test_key_12345",
        runtime_mode="test",
        mock_llm_stream=True,
        log_level=LogLevel.WARNING,
    )

    # 1. RetrieverPort: InMemoryHybridRetriever, LoadTestRetriever, and real HybridVectorStore
    in_memory_retriever = InMemoryHybridRetriever.from_markdown("# Title\n\nQdrant RRF text.")
    load_retriever = LoadTestRetriever()
    real_store = HybridVectorStore(
        client=SimpleNamespace(),  # type: ignore[arg-type]
        dense_embedder=_DummyDenseEmbedder(),
        sparse_embedder=_DummySparseEmbedder(),
        reranker=_DummyReranker(),
    )
    assert isinstance(in_memory_retriever, RetrieverPort)
    assert isinstance(load_retriever, RetrieverPort)
    assert isinstance(real_store, RetrieverPort)

    # 2. LLMRouterPort: DemoLLMRouter, LoadTestRouter, and real LLMRouter
    demo_router = DemoLLMRouter(settings)
    load_router = LoadTestRouter()
    real_router = LLMRouter(settings=settings)
    assert isinstance(demo_router, LLMRouterPort)
    assert isinstance(load_router, LLMRouterPort)
    assert isinstance(real_router, LLMRouterPort)

    # 3. SemanticCachePort: InMemorySemanticCache, LoadTestSemanticCache, and real SemanticCache
    in_mem_cache = InMemorySemanticCache(settings=settings)
    load_cache = LoadTestSemanticCache()
    real_cache = SemanticCache(
        redis_client=SimpleNamespace(),
        embedder=_DummyDenseEmbedder(),
        settings=settings,
    )
    assert isinstance(in_mem_cache, SemanticCachePort)
    assert isinstance(load_cache, SemanticCachePort)
    assert isinstance(real_cache, SemanticCachePort)

    # 4. RagasEvaluatorPort
    evaluator = RagasPipelineEvaluator(offline_mode=True)
    assert isinstance(evaluator, RagasEvaluatorPort)


async def test_mock_retriever_and_router_adapters_work_independently() -> None:
    """Directly exercise InMemoryHybridRetriever and DemoLLMRouter outside ServiceContainer."""
    corpus = (
        "# Vector Retrieval\n\n"
        "Qdrant stores 1024-dim BAAI/bge-m3 dense vectors with INT8 scalar quantization "
        "and BM25 sparse indices combined via Fusion.RRF before bge-reranker-v2-m3.\n\n"
        "# Hardware Constraints\n\n"
        "RTX 3060 6GB VRAM hosts only FP16 bge-m3 (~1.1GB) and bge-reranker-v2-m3 (~1.1GB), "
        "totalling ~2.3GB VRAM with zero local LLMs.\n\n"
        "# Agentic Loop\n\n"
        "LangGraph StateGraph uses AsyncSqliteSaver and max_retries=2."
    )
    retriever = InMemoryHybridRetriever.from_markdown(
        corpus,
        source="docs/test.md",
        doc_id_prefix="doc-test",
    )
    assert retriever.chunk_count == 3

    v1_hits = await retriever.search(
        "How does Qdrant combine BM25 sparse vectors with RRF?",
        pipeline_version="v1",
        top_k=3,
    )
    v3_hits = await retriever.search(
        "How does Qdrant combine BM25 sparse vectors with RRF?",
        pipeline_version="v3",
        top_k=3,
        top_n=10,
    )
    assert len(v1_hits) == 3
    assert len(v3_hits) == 3
    assert v3_hits[0].metadata.section_path == ["Vector Retrieval"]
    assert v3_hits[0].score > v1_hits[-1].score
    assert [chunk.rank for chunk in v3_hits] == [0, 1, 2]

    settings = Settings(runtime_mode="demo", mock_llm_stream=True, log_level=LogLevel.WARNING)
    router = DemoLLMRouter(settings)
    decision = router.select_route(task_type=TaskType.GENERATION, force_tier=RouteTier.STRONG)
    assert decision.tier == RouteTier.STRONG

    messages = [
        ("system", "Use context only."),
        (
            "user",
            "Question: What is the 6GB VRAM budget?\n\nContext:\n"
            f"[{v3_hits[1].metadata.doc_id}] (source: docs/test.md, section: Hardware Constraints)\n"
            f"{v3_hits[1].text}",
        ),
    ]
    gen = await router.ainvoke(messages, task_type=TaskType.GENERATION, force_tier=RouteTier.STRONG)
    assert gen.mocked is True
    assert "2.3 GB" in gen.content
    assert f"[{v3_hits[1].metadata.doc_id}]" in gen.content

    streamed = [
        tok
        async for tok in router.astream(
            messages,
            task_type=TaskType.GENERATION,
            force_tier=RouteTier.STRONG,
        )
    ]
    assert len(streamed) >= 3
    assert "".join(streamed) == gen.content

    doc_grade = await router.ainvoke_structured(DocumentGradeOutput, messages)
    hallucination_grade = await router.ainvoke_structured(HallucinationGradeOutput, messages)
    rewrite_out = await router.ainvoke_structured(
        QueryRewriteOutput,
        [("user", "Original query: qdrant rrf")],
    )
    assert doc_grade.relevant is True
    assert hallucination_grade.grounded is True
    assert "qdrant rrf" in rewrite_out.rewritten_query


async def test_eval_heuristics_adapter_evolution_and_load_sla() -> None:
    """Verify decoupled offline dataset/Ragas heuristics (v1 < v2 < v3) and load SLA."""
    chunker = make_chunker("v3")
    chunks = [
        *chunker.chunk(
            "# Qdrant Hybrid\n\nQdrant fuses INT8 bge-m3 dense and BM25 sparse vectors with RRF.",
            source="docs/qdrant.md",
            doc_id="doc-qdrant",
        ),
        *chunker.chunk(
            "# LangGraph\n\nLangGraph enforces max_retries=2 and persists via AsyncSqliteSaver.",
            source="docs/graph.md",
            doc_id="doc-graph",
        ),
    ]
    samples = build_offline_dataset(chunks, single_hop_count=2, multi_hop_count=2, seed=7)
    assert len(samples) == 4

    async def _runner(sample: Any, version: Any) -> dict[str, Any]:
        return await offline_simulated_runner(sample, str(version))

    async def _scorer(turn: Any, version: Any) -> dict[str, float]:
        return await heuristic_ragas_scorer(turn, str(version))

    evaluator = RagasPipelineEvaluator(runner=_runner, scorer=_scorer, offline_mode=True)
    report = await evaluator.compare_pipelines(samples, versions=("v1", "v2", "v3"))
    assert (
        report.summaries["v3"].overall_score
        > report.summaries["v2"].overall_score
        > report.summaries["v1"].overall_score
    )

    tester = AsyncLoadTester(concurrency=4, mock_llm_stream=True, cache_ratio=0.3)
    summary, _ = await tester.run_benchmark(
        total_requests=10,
        pipeline_version="v3",
        stream=True,
    )
    assert summary.success_rate == 1.0
    assert summary.p95_sla_passed is True
    assert summary.p95_latency_ms < 1800.0
    assert summary.cache_hit_rate > 0.0
