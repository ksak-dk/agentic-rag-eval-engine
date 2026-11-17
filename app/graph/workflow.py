"""
app/graph/workflow.py
=====================
Self-correcting LangGraph ``StateGraph`` orchestration and ``AsyncSqliteSaver``
checkpoint persistence for the Agentic RAG v3 pipeline.

Persistence contract
--------------------
- Strictly uses ``AsyncSqliteSaver`` (``./data/langgraph_checkpoints.sqlite3``).
- Never uses ``MemorySaver`` or synchronous ``SqliteSaver``.
- Hard cap ``max_retries = 2`` prevents infinite grader -> rewriter cycles.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.logging import request_metrics_context
from app.graph.nodes import GraphDependencies, bind_nodes
from app.graph.state import RAGGraphState, make_initial_state

logger = structlog.get_logger(__name__)

DEFAULT_CHECKPOINT_DB: Path = Path("./data/langgraph_checkpoints.sqlite3")
DEFAULT_MAX_RETRIES: int = 2

# Explicit msgpack allowlist for Pydantic V2 models persisted in checkpoints.
_CHECKPOINT_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    ("app.retrieval.chunker", "ChunkMetadata"),
    ("app.retrieval.hybrid_store", "RetrievedChunk"),
    ("app.graph.state", "Citation"),
)


def _try_get_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


def _resolve_checkpoint_path(db_path: str | Path | None) -> str:
    if db_path is not None and str(db_path) == ":memory:":
        return ":memory:"
    if db_path is not None:
        resolved = Path(db_path)
    else:
        settings = _try_get_settings()
        resolved = (
            settings.langgraph_checkpoint_db if settings is not None else DEFAULT_CHECKPOINT_DB
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


# ---------------------------------------------------------------------------
# Conditional Edge Routers
# ---------------------------------------------------------------------------


def route_after_cache_lookup(state: RAGGraphState) -> str:
    """Short-circuit to END on semantic cache hit; otherwise enter router."""
    if state.get("cache_hit"):
        return "end"
    return "router"


def route_after_grading(state: RAGGraphState, *, max_retries: int = DEFAULT_MAX_RETRIES) -> str:
    """Proceed to generation when graded chunks exist; rewrite or fallback otherwise."""
    if state.get("graded_documents"):
        return "generate"
    if int(state.get("retry_count", 0)) < max_retries:
        return "rewrite_query"
    return "fallback"


def route_after_hallucination_check(
    state: RAGGraphState,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Write cache if grounded; bounded retry or fallback when hallucinated."""
    if state.get("hallucination_passed"):
        return "cache_write"
    if (
        int(state.get("retry_count", 0)) < max_retries
        and int(state.get("hallucination_retries", 0)) <= 1
    ):
        return "rewrite_query"
    return "fallback"


# ---------------------------------------------------------------------------
# Graph Builder & Compiler
# ---------------------------------------------------------------------------


def build_state_graph(
    deps: GraphDependencies,
    *,
    max_retries: int | None = None,
) -> StateGraph:
    """Construct the uncompiled self-correcting ``StateGraph``."""
    effective_max_retries = deps.max_retries if max_retries is None else int(max_retries)
    nodes = bind_nodes(deps)

    builder: StateGraph = StateGraph(RAGGraphState)
    for node_name, node_fn in nodes.items():
        builder.add_node(node_name, node_fn)

    def _grade_edge(state: RAGGraphState) -> str:
        return route_after_grading(state, max_retries=effective_max_retries)

    def _hallucination_edge(state: RAGGraphState) -> str:
        return route_after_hallucination_check(state, max_retries=effective_max_retries)

    builder.add_edge(START, "cache_lookup")
    builder.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {
            "end": END,
            "router": "router",
        },
    )
    builder.add_edge("router", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        _grade_edge,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )
    builder.add_edge("rewrite_query", "router")
    builder.add_edge("generate", "hallucination_check")
    builder.add_conditional_edges(
        "hallucination_check",
        _hallucination_edge,
        {
            "cache_write": "cache_write",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )
    builder.add_edge("cache_write", END)
    builder.add_edge("fallback", END)
    return builder


def compile_workflow(
    deps: GraphDependencies,
    *,
    checkpointer: AsyncSqliteSaver | None = None,
    max_retries: int | None = None,
) -> CompiledStateGraph:
    """Compile the ``StateGraph``, enforcing ``AsyncSqliteSaver`` when persisted."""
    if checkpointer is not None and not isinstance(checkpointer, AsyncSqliteSaver):
        raise TypeError(
            f"Checkpointer must be an AsyncSqliteSaver instance; got {type(checkpointer).__name__}."
        )
    graph = build_state_graph(deps, max_retries=max_retries)
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_async_sqlite_saver(
    db_path: str | Path | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open and initialize an ``AsyncSqliteSaver`` with msgpack allowlist."""
    resolved_path = _resolve_checkpoint_path(db_path)
    async with AsyncSqliteSaver.from_conn_string(str(resolved_path)) as saver:
        await saver.setup()
        configured = saver.with_allowlist(_CHECKPOINT_ALLOWLIST)
        yield configured  # type: ignore[misc]


@asynccontextmanager
async def create_workflow_app(
    deps: GraphDependencies,
    *,
    db_path: str | Path | None = None,
    max_retries: int | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """Context manager yielding a compiled ``StateGraph`` backed by ``AsyncSqliteSaver``."""
    async with open_async_sqlite_saver(db_path) as checkpointer:
        yield compile_workflow(deps, checkpointer=checkpointer, max_retries=max_retries)


# ---------------------------------------------------------------------------
# High-Level Async Runner
# ---------------------------------------------------------------------------


class AgenticRAGWorkflow:
    """Convenience runner managing checkpointed execution and thread recovery."""

    def __init__(
        self,
        deps: GraphDependencies,
        *,
        db_path: str | Path | None = None,
        max_retries: int | None = None,
        checkpointer: AsyncSqliteSaver | None = None,
    ) -> None:
        self._deps = deps
        self._db_path = db_path
        self._max_retries = deps.max_retries if max_retries is None else int(max_retries)
        self._checkpointer = checkpointer
        self._compiled: CompiledStateGraph | None = (
            compile_workflow(deps, checkpointer=checkpointer, max_retries=self._max_retries)
            if checkpointer is not None
            else None
        )

    async def ainvoke(
        self,
        query: str,
        *,
        session_id: str | None = None,
        pipeline_version: str = "v3",
    ) -> RAGGraphState:
        """Execute one query end-to-end inside an async metrics + checkpoint scope."""
        resolved_session = session_id or str(uuid.uuid4())
        initial_state = make_initial_state(
            query,
            session_id=resolved_session,
            pipeline_version=pipeline_version,
        )
        config: dict[str, Any] = {"configurable": {"thread_id": resolved_session}}

        if self._compiled is not None:
            async with request_metrics_context(session_id=resolved_session):
                result: RAGGraphState = await self._compiled.ainvoke(initial_state, config=config)
                return result

        async with (
            request_metrics_context(session_id=resolved_session),
            create_workflow_app(
                self._deps,
                db_path=self._db_path,
                max_retries=self._max_retries,
            ) as app,
        ):
            result = await app.ainvoke(initial_state, config=config)
            return result

    async def aget_state(self, session_id: str) -> Any:
        """Load the latest persisted checkpoint snapshot for ``session_id``."""
        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
        if self._compiled is not None:
            return await self._compiled.aget_state(config)
        async with create_workflow_app(
            self._deps,
            db_path=self._db_path,
            max_retries=self._max_retries,
        ) as app:
            return await app.aget_state(config)


__all__: Sequence[str] = [
    "DEFAULT_CHECKPOINT_DB",
    "DEFAULT_MAX_RETRIES",
    "AgenticRAGWorkflow",
    "build_state_graph",
    "compile_workflow",
    "create_workflow_app",
    "open_async_sqlite_saver",
    "route_after_cache_lookup",
    "route_after_grading",
    "route_after_hallucination_check",
]
