from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.cache.semantic_cache import CachedResponse
from app.core.logging import get_request_metrics, request_metrics_context
from app.core.router import LLMGenerationResult, RouteDecision, RouteTier
from app.graph.nodes import FALLBACK_ANSWER, GraphDependencies
from app.graph.state import make_initial_state
from app.graph.workflow import (
    AgenticRAGWorkflow,
    compile_workflow,
    create_workflow_app,
)
from app.retrieval.chunker import ChunkMetadata
from app.retrieval.hybrid_store import RetrievedChunk


def _make_chunk(doc_id: str, text: str, *, score: float = 0.9, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            source=f"docs/{doc_id}.md",
            section_path=["Architecture", "RAG"],
            chunk_index=rank,
        ),
        score=score,
        rank=rank,
    )


class StubSemanticCache:
    """In-memory async semantic cache stub for isolated graph boundary tests."""

    def __init__(
        self, initial_entries: dict[tuple[str, str], CachedResponse] | None = None
    ) -> None:
        self.entries: dict[tuple[str, str], CachedResponse] = dict(initial_entries or {})
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str, str, dict[str, Any]]] = []

    async def get(self, query: str, *, namespace: str = "v3") -> CachedResponse | None:
        self.get_calls.append((namespace, query))
        return self.entries.get((namespace, query))

    async def set(
        self,
        query: str,
        response: str,
        *,
        namespace: str = "v3",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = dict(metadata or {})
        self.set_calls.append((namespace, query, response, meta))
        self.entries[(namespace, query)] = CachedResponse(
            query=query,
            response=response,
            namespace=namespace,
            similarity=1.0,
            metadata=meta,
        )
        return f"rag:cache:{namespace}:stub"


class StubRetriever:
    """Async retriever stub recording every query issued across retry loops."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        return list(self._chunks[:top_k])


class StubRouter:
    """Async LLMRouter stub with call tracking and zero external API calls."""

    def __init__(self, answer_text: str = "Synthesized answer.") -> None:
        self.answer_text = answer_text
        self.route_calls: list[tuple[str, int]] = []
        self.invoke_calls: list[dict[str, Any]] = []

    def select_route(
        self,
        *,
        task_type: Any = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: Any = None,
    ) -> RouteDecision:
        self.route_calls.append((query or "", retry_count))
        if retry_count > 0:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name="gemini-2.5-flash",
                fallback_model_name=None,
                reason="Escalated on retry",
            )
        return RouteDecision(
            tier=RouteTier.FAST,
            model_name="llama-3.1-8b-instant",
            fallback_model_name="gemini-2.5-flash-lite",
            reason="Default fast route",
        )

    async def ainvoke(
        self,
        messages: Any,
        *,
        task_type: Any = None,
        force_tier: Any = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> LLMGenerationResult:
        self.invoke_calls.append(
            {
                "task_type": str(task_type),
                "force_tier": str(force_tier),
                "retry_count": retry_count,
            }
        )
        return LLMGenerationResult(
            content=self.answer_text,
            route_tier=RouteTier.STRONG,
            model_name="gemini-2.5-flash",
            mocked=True,
            latency_ms=5.0,
        )


async def test_cache_hit_short_circuits_graph_execution() -> None:
    cached_entry = CachedResponse(
        query="What is RRF?",
        response="Reciprocal Rank Fusion combines dense and sparse rankings.",
        namespace="v3",
        similarity=0.97,
        metadata={
            "route_tier": "fast",
            "citations": [
                {
                    "doc_id": "doc-rrf",
                    "source": "docs/rrf.md",
                    "section_path": ["Retrieval"],
                    "chunk_index": 0,
                    "score": 0.95,
                    "snippet": "RRF merges candidate lists.",
                }
            ],
        },
    )
    cache = StubSemanticCache({("v3", "What is RRF?"): cached_entry})
    retriever = StubRetriever([_make_chunk("doc-rrf", "RRF details")])
    router = StubRouter()
    deps = GraphDependencies(retriever=retriever, router=router, cache=cache)
    app = compile_workflow(deps)

    async with request_metrics_context(session_id="sess-cache"):
        result = await app.ainvoke(
            make_initial_state("What is RRF?", session_id="sess-cache", pipeline_version="v3")
        )
        metrics = get_request_metrics()

    assert result["cache_hit"] is True
    assert result["answer"] == "Reciprocal Rank Fusion combines dense and sparse rankings."
    assert [c.doc_id for c in result["citations"]] == ["doc-rrf"]
    assert retriever.queries == []
    assert router.route_calls == []
    assert router.invoke_calls == []
    assert metrics.cache_hit is True


async def test_irrelevant_retrieval_hits_max_retries_cap_and_enters_fallback() -> None:
    cache = StubSemanticCache()
    retriever = StubRetriever([_make_chunk("doc-noise", "Unrelated text about cooking.")])
    router = StubRouter()

    async def always_irrelevant(query: str, chunk: RetrievedChunk) -> bool:
        return False

    async def rewrite_step(query: str, retry_count: int) -> str:
        return f"{query} refined-{retry_count}"

    deps = GraphDependencies(
        retriever=retriever,
        router=router,
        cache=cache,
        document_grader=always_irrelevant,
        query_rewriter=rewrite_step,
        max_retries=2,
    )
    app = compile_workflow(deps, max_retries=2)

    async with request_metrics_context(session_id="sess-retry"):
        result = await app.ainvoke(
            make_initial_state("How does Qdrant INT8 work?", session_id="sess-retry")
        )

    assert result["retry_count"] == 2
    assert result["fallback_triggered"] is True
    assert result["hallucination_passed"] is False
    assert result["answer"] == FALLBACK_ANSWER
    assert result["citations"] == []
    assert len(retriever.queries) == 3
    assert retriever.queries == [
        "How does Qdrant INT8 work?",
        "How does Qdrant INT8 work? refined-1",
        "How does Qdrant INT8 work? refined-1 refined-2",
    ]
    assert cache.set_calls == []


async def test_multihop_generation_passes_hallucination_check_and_writes_cache() -> None:
    cache = StubSemanticCache()
    chunks = [
        _make_chunk("doc-arch", "Qdrant stores INT8 quantized dense vectors.", score=0.94, rank=0),
        _make_chunk("doc-rrf", "RRF merges dense and BM25 sparse candidates.", score=0.91, rank=1),
    ]
    retriever = StubRetriever(chunks)
    router = StubRouter(
        answer_text="Hybrid search uses INT8 dense vectors [doc-arch] fused via RRF [doc-rrf]."
    )

    async def all_relevant(query: str, chunk: RetrievedChunk) -> bool:
        return True

    async def grounded_ok(answer: str, docs: list[RetrievedChunk]) -> bool:
        return "[doc-arch]" in answer and "[doc-rrf]" in answer and len(docs) == 2

    deps = GraphDependencies(
        retriever=retriever,
        router=router,
        cache=cache,
        document_grader=all_relevant,
        hallucination_checker=grounded_ok,
    )
    app = compile_workflow(deps)

    async with request_metrics_context(session_id="sess-gen"):
        result = await app.ainvoke(
            make_initial_state("Explain hybrid search architecture.", session_id="sess-gen")
        )

    assert result["cache_hit"] is False
    assert result["hallucination_passed"] is True
    assert result["fallback_triggered"] is False
    assert result["route_tier"] == "strong"
    assert [c.doc_id for c in result["citations"]] == ["doc-arch", "doc-rrf"]
    assert len(cache.set_calls) == 1

    cached = await cache.get("Explain hybrid search architecture.", namespace="v3")
    assert cached is not None
    assert "RRF [doc-rrf]" in cached.response


async def test_hallucination_failure_is_bounded_and_falls_back() -> None:
    cache = StubSemanticCache()
    retriever = StubRetriever([_make_chunk("doc-1", "Valid chunk text.")])
    router = StubRouter(answer_text="Unsupported claim [doc-1].")

    async def relevant_ok(query: str, chunk: RetrievedChunk) -> bool:
        return True

    async def rewrite_step(query: str, retry_count: int) -> str:
        return f"{query} retry-{retry_count}"

    async def always_hallucinated(answer: str, docs: list[RetrievedChunk]) -> bool:
        return False

    deps = GraphDependencies(
        retriever=retriever,
        router=router,
        cache=cache,
        document_grader=relevant_ok,
        query_rewriter=rewrite_step,
        hallucination_checker=always_hallucinated,
        max_retries=2,
    )
    app = compile_workflow(deps, max_retries=2)

    result = await app.ainvoke(make_initial_state("Explain unsupported claim"))

    assert result["fallback_triggered"] is True
    assert result["hallucination_passed"] is False
    assert result["answer"] == FALLBACK_ANSWER
    assert len(retriever.queries) <= 3
    assert cache.set_calls == []


async def test_async_sqlite_saver_persists_and_recovers_checkpoint_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "langgraph_checkpoints.sqlite3"
    cache = StubSemanticCache()
    retriever = StubRetriever(
        [_make_chunk("doc-langgraph", "LangGraph persists state with AsyncSqliteSaver.", rank=0)]
    )
    router = StubRouter(
        answer_text="LangGraph persists state with AsyncSqliteSaver [doc-langgraph]."
    )

    async def relevant_ok(query: str, chunk: RetrievedChunk) -> bool:
        return True

    async def grounded_ok(answer: str, docs: list[RetrievedChunk]) -> bool:
        return True

    deps = GraphDependencies(
        retriever=retriever,
        router=router,
        cache=cache,
        document_grader=relevant_ok,
        hallucination_checker=grounded_ok,
    )
    workflow = AgenticRAGWorkflow(deps, db_path=db_path)

    result = await workflow.ainvoke(
        "How does LangGraph persist checkpoints?",
        session_id="thread-persist-1",
    )
    assert db_path.exists()
    assert result["hallucination_passed"] is True

    # Re-open a brand new AsyncSqliteSaver connection to verify durable recovery
    async with create_workflow_app(deps, db_path=db_path) as recovered_app:
        snapshot = await recovered_app.aget_state(
            {"configurable": {"thread_id": "thread-persist-1"}}
        )

    assert snapshot.values["query"] == "How does LangGraph persist checkpoints?"
    assert snapshot.values["session_id"] == "thread-persist-1"
    assert snapshot.values["hallucination_passed"] is True
    assert snapshot.values["answer"] == (
        "LangGraph persists state with AsyncSqliteSaver [doc-langgraph]."
    )
    assert snapshot.values["citations"][0].doc_id == "doc-langgraph"


def test_compile_workflow_rejects_non_async_sqlite_checkpointer() -> None:
    deps = GraphDependencies(
        retriever=StubRetriever([]),
        router=SimpleNamespace(),
    )
    with pytest.raises(TypeError, match="AsyncSqliteSaver"):
        compile_workflow(deps, checkpointer=SimpleNamespace())  # type: ignore[arg-type]
