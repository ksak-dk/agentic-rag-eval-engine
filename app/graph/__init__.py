"""
app/graph/__init__.py
=====================
Public re-exports for the LangGraph agentic RAG state machine.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.graph.nodes import (
    FALLBACK_ANSWER,
    DocumentGradeOutput,
    GraphDependencies,
    HallucinationGradeOutput,
    QueryRewriteOutput,
    bind_nodes,
    cache_lookup_node,
    cache_write_node,
    fallback_node,
    generate_node,
    grade_documents_node,
    hallucination_check_node,
    retrieve_node,
    rewrite_query_node,
    router_node,
)
from app.graph.state import (
    Citation,
    GraphState,
    RAGGraphState,
    coerce_citation,
    coerce_retrieved_chunk,
    make_initial_state,
)
from app.graph.workflow import (
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_MAX_RETRIES,
    AgenticRAGWorkflow,
    build_state_graph,
    compile_workflow,
    create_workflow_app,
    open_async_sqlite_saver,
    route_after_cache_lookup,
    route_after_grading,
    route_after_hallucination_check,
)

__all__: Sequence[str] = [
    "DEFAULT_CHECKPOINT_DB",
    "DEFAULT_MAX_RETRIES",
    "FALLBACK_ANSWER",
    "AgenticRAGWorkflow",
    "Citation",
    "DocumentGradeOutput",
    "GraphDependencies",
    "GraphState",
    "HallucinationGradeOutput",
    "QueryRewriteOutput",
    "RAGGraphState",
    "bind_nodes",
    "build_state_graph",
    "cache_lookup_node",
    "cache_write_node",
    "coerce_citation",
    "coerce_retrieved_chunk",
    "compile_workflow",
    "create_workflow_app",
    "fallback_node",
    "generate_node",
    "grade_documents_node",
    "hallucination_check_node",
    "make_initial_state",
    "open_async_sqlite_saver",
    "retrieve_node",
    "rewrite_query_node",
    "route_after_cache_lookup",
    "route_after_grading",
    "route_after_hallucination_check",
    "router_node",
]
