"""
app/graph/state.py
==================
Strongly-typed LangGraph state schema and citation models for the
self-correcting agentic RAG workflow.

Contract (aligned with GEMINI.md / copilot-instructions.md)
-----------------------------------------------------------
- State is a ``TypedDict`` (``RAGGraphState``).
- Embedded Pydantic V2 models (``Citation``, ``RetrievedChunk``) are frozen
  and JSON-serializable for ``AsyncSqliteSaver`` checkpoint persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.hybrid_store import RetrievedChunk


class Citation(BaseModel):
    """Structured provenance citation attached to a generated RAG answer."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Document identifier cited in the generated answer.")
    source: str = Field(description="Source file path or URL.")
    section_path: list[str] = Field(
        default_factory=list,
        description="Markdown header ancestry leading to the cited chunk.",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based chunk index within the source document.",
    )
    score: float = Field(default=0.0, description="Retrieval or reranker relevance score.")
    snippet: str = Field(default="", description="Short excerpt from the cited chunk.")


class RAGGraphState(TypedDict, total=False):
    """Mutable state flowing across nodes in the LangGraph ``StateGraph``."""

    query: str
    rewritten_query: str
    session_id: str
    pipeline_version: str
    route_tier: str
    cache_hit: bool
    documents: list[RetrievedChunk]
    graded_documents: list[RetrievedChunk]
    retry_count: int
    hallucination_retries: int
    answer: str
    citations: list[Citation]
    hallucination_passed: bool
    fallback_triggered: bool
    is_fallback: bool
    generation_mode: str
    error: str | None


GraphState = RAGGraphState


def make_initial_state(
    query: str,
    *,
    session_id: str = "",
    pipeline_version: str = "v3",
) -> RAGGraphState:
    """Return a clean, fully-initialized ``RAGGraphState`` for graph invocation."""
    return RAGGraphState(
        query=query,
        rewritten_query=query,
        session_id=session_id,
        pipeline_version=pipeline_version,
        route_tier="fast",
        cache_hit=False,
        documents=[],
        graded_documents=[],
        retry_count=0,
        hallucination_retries=0,
        answer="",
        citations=[],
        hallucination_passed=False,
        fallback_triggered=False,
        is_fallback=False,
        generation_mode="",
        error=None,
    )


def coerce_retrieved_chunk(item: RetrievedChunk | dict[str, Any]) -> RetrievedChunk:
    """Reconstruct a ``RetrievedChunk`` when deserialized from checkpoint dicts."""
    if isinstance(item, RetrievedChunk):
        return item
    return RetrievedChunk.model_validate(item)


def coerce_citation(item: Citation | dict[str, Any]) -> Citation:
    """Reconstruct a ``Citation`` when deserialized from checkpoint dicts."""
    if isinstance(item, Citation):
        return item
    return Citation.model_validate(item)


__all__: Sequence[str] = [
    "Citation",
    "GraphState",
    "RAGGraphState",
    "coerce_citation",
    "coerce_retrieved_chunk",
    "make_initial_state",
]
