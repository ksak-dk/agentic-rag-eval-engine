"""
app/graph/nodes.py
==================
Async LangGraph node functions for the self-correcting Agentic RAG pipeline.

Nodes
-----
1. ``cache_lookup_node``       — Redis 8 vector semantic cache lookup (short-circuit on hit)
2. ``router_node``             — Zero-cost Fast vs Strong LLM route selection
3. ``retrieve_node``           — Hybrid Qdrant + CUDA rerank retrieval
4. ``grade_documents_node``    — Fast-route relevance grading & irrelevant chunk filtering
5. ``rewrite_query_node``      — Query rewriting bounded by ``max_retries = 2``
6. ``generate_node``           — Strong-route answer synthesis with ``[doc_id]`` citations
7. ``hallucination_check_node``— Strong-route grounding verification against graded chunks
8. ``cache_write_node``        — Persist verified answers to Redis semantic cache
9. ``fallback_node``           — Safe deterministic fallback when retries are exhausted
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.vertex.router import VertexAIRouter, VertexAIRouterError
from app.cache.semantic_cache import is_cacheable_response
from app.core.config import Settings, get_settings
from app.core.logging import log_stage_latency, update_request_metrics
from app.core.router import RouteTier, TaskType
from app.graph.state import (
    Citation,
    RAGGraphState,
    coerce_citation,
    coerce_retrieved_chunk,
)
from app.retrieval.hybrid_store import RetrievedChunk

logger = structlog.get_logger(__name__)

FALLBACK_ANSWER: str = (
    "I could not find sufficiently relevant and verified context in the knowledge base "
    "to answer your question reliably after exhausting retrieval retries."
)

_DOC_TAG_RE: re.Pattern[str] = re.compile(r"\[([^\[\]\s]+)\]")
_WORD_RE: re.Pattern[str] = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff\uac00-\ud7af]+")
_KO_DETECT_RE: re.Pattern[str] = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_ZH_DETECT_RE: re.Pattern[str] = re.compile(r"[\u4e00-\u9fff]")


def detect_query_language(query: str) -> str:
    """Detect whether a user query is Korean ('ko'), Chinese ('zh'), or English ('en')."""
    if _KO_DETECT_RE.search(query):
        return "ko"
    if _ZH_DETECT_RE.search(query):
        return "zh"
    return "en"


def build_multilingual_system_prompt(query: str) -> str:
    """Build a language-aligned generation system prompt preserving [doc_id] citations."""
    lang = detect_query_language(query)
    base = (
        "Answer the user question using ONLY the provided context snippets. "
        "Cite supporting sources inline using their exact [doc_id] tag. "
        "You MUST respond strictly in the exact same language as the user question."
    )
    if lang == "ko":
        return (
            f"{base} "
            "사용자의 질문이 한국어(Korean)이므로, 제공된 컨텍스트(Context)에 중국어나 영어가 포함되어 있더라도 "
            "모든 기술적 사실과 수치를 자연스럽고 전문적인 한국어로 완벽히 번역 및 종합하여 답변하십시오. "
            "답변 본문에 중국어 원문 단락을 그대로 복사하지 마시고, 인용 태그 [doc_id]만 정확히 유지하십시오."
        )
    if lang == "zh":
        return (
            f"{base} "
            "用户提问为中文, 请严格使用专业、流畅的简体中文综合回答, 并准确保留 [doc_id] 格式的内联引用标签。"
        )
    return base


# ---------------------------------------------------------------------------
# Structured LLM Output Schemas (Pydantic V2)
# ---------------------------------------------------------------------------


class DocumentGradeOutput(BaseModel):
    """Binary relevance verdict produced by ``grade_documents_node``."""

    model_config = ConfigDict(frozen=True)

    relevant: bool = Field(
        description="True when the retrieved chunk contains facts useful for answering the query."
    )
    reason: str = Field(default="", description="Short justification for the relevance grade.")


class QueryRewriteOutput(BaseModel):
    """Refined retrieval query produced by ``rewrite_query_node``."""

    model_config = ConfigDict(frozen=True)

    rewritten_query: str = Field(description="Rewritten query optimized for hybrid search.")


class HallucinationGradeOutput(BaseModel):
    """Faithfulness check verdict produced by ``hallucination_check_node``."""

    model_config = ConfigDict(frozen=True)

    grounded: bool = Field(
        description="True when every claim in the answer is supported by graded documents."
    )
    reason: str = Field(default="", description="Explanation of grounding verdict.")


# ---------------------------------------------------------------------------
# Dependency Container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GraphDependencies:
    """Injected runtime services and optional test overrides for graph nodes."""

    retriever: Any
    router: Any
    cache: Any | None = None
    settings: Settings | None = None
    vertex_router: Any | None = None
    document_grader: Callable[[str, RetrievedChunk], Awaitable[bool]] | None = None
    query_rewriter: Callable[[str, int], Awaitable[str]] | None = None
    hallucination_checker: Callable[[str, list[RetrievedChunk]], Awaitable[bool]] | None = None
    top_k: int = 5
    top_n: int = 20
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _lexical_relevance_fallback(query: str, text: str) -> bool:
    """Lightweight keyword overlap fallback if structured LLM grading fails."""
    query_terms = {tok.lower() for tok in _WORD_RE.findall(query) if len(tok) > 1}
    if not query_terms:
        return bool(text.strip())
    text_terms = {tok.lower() for tok in _WORD_RE.findall(text)}
    return bool(query_terms & text_terms)


def _format_context_entries(chunks: list[RetrievedChunk]) -> list[str]:
    entries: list[str] = []
    for chunk in chunks:
        header_trail = " > ".join(chunk.metadata.section_path) or "root"
        entries.append(
            f"[{chunk.metadata.doc_id}] "
            f"(source: {chunk.metadata.source}, section: {header_trail})\n"
            f"{chunk.text}"
        )
    return entries


def _format_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(_format_context_entries(chunks))


def _build_rag_messages(query: str, docs: list[RetrievedChunk]) -> list[tuple[str, str]]:
    """Build trilingual-aligned system and user messages for RAG synthesis."""
    context_block = _format_context_block(docs)
    if context_block:
        return [
            (
                "system",
                build_multilingual_system_prompt(query),
            ),
            ("user", f"Question: {query}\n\nContext:\n{context_block}"),
        ]
    return [
        (
            "system",
            "Answer the user question concisely in the exact same language as the user question. "
            "If insufficient context is available, state that clearly.",
        ),
        ("user", f"Question: {query}"),
    ]


def _resolve_node_settings(deps: GraphDependencies) -> Settings | None:
    if deps.settings is not None:
        return deps.settings
    router_settings = getattr(deps.router, "settings", None)
    if isinstance(router_settings, Settings):
        return router_settings
    try:
        return get_settings()
    except ValidationError:
        return None


def _chunk_to_citation(chunk: RetrievedChunk, *, lang: str = "en") -> Citation:
    raw_snippet = chunk.text[:240]
    section_list = list(chunk.metadata.section_path)
    if lang == "ko":
        from app.adapters.mock.router import DemoLLMRouter

        ko_label, ko_snippet = DemoLLMRouter.localize_chunk_for_ko(
            chunk.metadata.doc_id,
            chunk.metadata.source,
            " > ".join(section_list),
            raw_snippet,
        )
        raw_snippet = ko_snippet[:240]
        if ko_label:
            section_list = [ko_label]
    return Citation(
        doc_id=chunk.metadata.doc_id,
        source=chunk.metadata.source,
        section_path=section_list,
        chunk_index=chunk.metadata.chunk_index,
        score=float(chunk.score),
        snippet=raw_snippet,
    )


def _extract_citations(answer: str, docs: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Extract cited ``[doc_id]`` tags from ``answer`` and map to ``Citation`` models."""
    if not docs:
        return answer, []

    lang = detect_query_language(answer)
    by_doc_id: dict[str, RetrievedChunk] = {}
    for doc in docs:
        if doc.metadata.doc_id not in by_doc_id:
            by_doc_id[doc.metadata.doc_id] = doc

    matched_ids: list[str] = []
    for tag in _DOC_TAG_RE.findall(answer):
        if tag in by_doc_id and tag not in matched_ids:
            matched_ids.append(tag)

    if matched_ids:
        citations = [_chunk_to_citation(by_doc_id[doc_id], lang=lang) for doc_id in matched_ids]
        return answer, citations

    # Ensure provenance is never lost when LLM omits explicit bracket tags (e.g. mock mode).
    top_doc = docs[0]
    annotated_answer = f"{answer.rstrip()} [{top_doc.metadata.doc_id}]".strip()
    return annotated_answer, [_chunk_to_citation(top_doc, lang=lang)]


def _parse_cached_citations(raw_citations: Any) -> list[Citation]:
    if not isinstance(raw_citations, list):
        return []
    parsed: list[Citation] = []
    for item in raw_citations:
        try:
            parsed.append(coerce_citation(item))
        except Exception:
            continue
    return parsed


# ---------------------------------------------------------------------------
# Async Node Functions
# ---------------------------------------------------------------------------


async def cache_lookup_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Query the Redis semantic cache and short-circuit on similarity hit."""
    query = state.get("query", "").strip()
    session_id = state.get("session_id", "")
    pipeline_version = state.get("pipeline_version", "v3")

    if deps.cache is None or not query:
        update_request_metrics(cache_hit=False, session_id=session_id)
        return {"cache_hit": False}

    async with log_stage_latency("cache_lookup"):
        try:
            hit = await deps.cache.get(query, namespace=pipeline_version)
        except Exception as exc:
            logger.warning(
                "Cache lookup node failed (fail-open)",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            hit = None

    if hit is None:
        update_request_metrics(cache_hit=False, session_id=session_id)
        return {"cache_hit": False}

    meta = getattr(hit, "metadata", {}) or {}
    cached_text = str(getattr(hit, "response", "") or "")
    if not is_cacheable_response(cached_text, metadata=meta):
        update_request_metrics(cache_hit=False, session_id=session_id)
        logger.warning(
            "Skipped polluted or fallback semantic cache entry",
            session_id=session_id,
        )
        return {"cache_hit": False}

    route_tier = str(meta.get("route_tier", "fast"))
    cached_gen_mode = str(meta.get("generation_mode") or "live_llm")
    citations = _parse_cached_citations(meta.get("citations"))
    update_request_metrics(
        cache_hit=True,
        session_id=session_id,
        route_tier=route_tier,
        generation_mode=cached_gen_mode,
    )
    return {
        "cache_hit": True,
        "answer": cached_text,
        "route_tier": route_tier,
        "citations": citations,
        "hallucination_passed": True,
        "fallback_triggered": False,
        "is_fallback": False,
        "generation_mode": cached_gen_mode,
        "error": None,
    }


async def router_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Select Fast vs Strong LLM tier based on query complexity and retry count."""
    active_query = (state.get("rewritten_query") or state.get("query", "")).strip()
    retry_count = int(state.get("retry_count", 0))

    async with log_stage_latency("router"):
        decision = deps.router.select_route(query=active_query, retry_count=retry_count)

    tier_val = decision.tier.value if hasattr(decision.tier, "value") else str(decision.tier)
    update_request_metrics(
        route_tier=tier_val,
        model_name=decision.model_name,
        retry_count=retry_count,
    )
    return {
        "route_tier": tier_val,
        "rewritten_query": active_query,
    }


async def retrieve_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Execute hybrid dense+sparse retrieval and local CUDA reranking."""
    active_query = (state.get("rewritten_query") or state.get("query", "")).strip()
    pipeline_version = state.get("pipeline_version", "v3")

    async with log_stage_latency("retrieve", pipeline_version=pipeline_version):
        try:
            raw_docs = await deps.retriever.search(
                active_query,
                pipeline_version=pipeline_version,
                top_k=deps.top_k,
                top_n=deps.top_n,
            )
            docs = [coerce_retrieved_chunk(doc) for doc in raw_docs]
            return {"documents": docs, "error": None}
        except Exception as exc:
            logger.warning(
                "Retrieval node failed",
                query=active_query[:80],
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return {"documents": [], "error": str(exc)}


async def grade_documents_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Filter retrieved chunks using the fast LLM route (or injected grader)."""
    active_query = (state.get("rewritten_query") or state.get("query", "")).strip()
    docs = [coerce_retrieved_chunk(d) for d in (state.get("documents") or [])]

    if not docs:
        return {"graded_documents": []}

    async def _grade_chunk(chunk: RetrievedChunk) -> bool:
        if deps.document_grader is not None:
            return bool(await deps.document_grader(active_query, chunk))

        messages = [
            (
                "system",
                "You are a retrieval grader. Determine whether the document chunk contains "
                "information relevant to answering the user query.",
            ),
            ("user", f"Query: {active_query}\n\nDocument:\n{chunk.text}"),
        ]
        try:
            verdict = await deps.router.ainvoke_structured(
                DocumentGradeOutput,
                messages,
                task_type=TaskType.DOC_GRADING,
            )
            return bool(verdict.relevant)
        except Exception as exc:
            logger.warning(
                "Structured document grading failed, using lexical fallback",
                doc_id=chunk.metadata.doc_id,
                error_type=type(exc).__name__,
            )
            return _lexical_relevance_fallback(active_query, chunk.text)

    async with log_stage_latency("grade_documents", candidate_count=len(docs)):
        verdicts = await asyncio.gather(*(_grade_chunk(chunk) for chunk in docs))

    graded = [chunk for chunk, relevant in zip(docs, verdicts, strict=True) if relevant]
    return {"graded_documents": graded}


async def rewrite_query_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Rewrite the retrieval query on relevance/hallucination retry (capped at max_retries)."""
    current_query = (state.get("rewritten_query") or state.get("query", "")).strip()
    next_retry = int(state.get("retry_count", 0)) + 1

    async with log_stage_latency("rewrite_query", retry_count=next_retry):
        if deps.query_rewriter is not None:
            rewritten = (await deps.query_rewriter(current_query, next_retry)).strip()
        else:
            messages = [
                (
                    "system",
                    "Rewrite the user search query into a clearer, more specific retrieval "
                    "query that surfaces key technical terms. Return only the rewritten query.",
                ),
                ("user", f"Original query: {current_query}"),
            ]
            try:
                result = await deps.router.ainvoke(
                    messages,
                    task_type=TaskType.QUERY_REWRITE,
                )
                rewritten = result.content.strip().strip('"').strip("'")
            except Exception as exc:
                logger.warning(
                    "Query rewrite failed, keeping current query",
                    retry_count=next_retry,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                rewritten = current_query

    final_query = rewritten or current_query
    update_request_metrics(retry_count=next_retry)
    return {
        "rewritten_query": final_query,
        "retry_count": next_retry,
    }


async def _try_vertex_generation(
    query: str,
    docs: list[RetrievedChunk],
    deps: GraphDependencies,
    settings: Settings | None,
) -> tuple[str, str] | None:
    """Route generation to VertexAIRouter only when explicitly enabled with valid credentials."""
    if settings is None or not settings.use_vertex_ai:
        return None

    vertex_router = deps.vertex_router or VertexAIRouter(settings=settings)
    if not vertex_router.has_valid_credentials():
        logger.warning(
            "USE_VERTEX_AI is enabled but Vertex AI credentials or project ID are unavailable; "
            "falling back to standard router",
            project_id=settings.vertex_project_id,
        )
        return None

    lang = detect_query_language(query)
    context_entries = _format_context_entries(docs)
    try:
        answer = await vertex_router.generate_response(query, context_entries, lang)
    except VertexAIRouterError as exc:
        logger.warning(
            "VertexAIRouter failed; falling back to standard router",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None
    except Exception as exc:
        logger.warning(
            "Unexpected VertexAIRouter error; falling back to standard router",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None

    model_name = str(getattr(vertex_router, "model_name", "") or settings.strong_llm_model)
    return answer, model_name


async def generate_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Generate a grounded answer citing source ``[doc_id]`` identifiers."""
    query = (state.get("query") or state.get("rewritten_query") or "").strip()
    docs = [coerce_retrieved_chunk(d) for d in (state.get("graded_documents") or [])]
    retry_count = int(state.get("retry_count", 0))

    if not docs:
        return {
            "answer": "",
            "citations": [],
            "error": "No graded documents available for generation.",
        }

    messages = _build_rag_messages(query, docs)
    node_settings = _resolve_node_settings(deps)

    async with log_stage_latency("generate", context_chunks=len(docs)):
        vertex_result = await _try_vertex_generation(query, docs, deps, node_settings)
        if vertex_result is not None:
            raw_answer, vertex_model = vertex_result
            annotated_answer, citations = _extract_citations(raw_answer, docs)
            update_request_metrics(
                route_tier=RouteTier.STRONG.value,
                model_name=vertex_model,
                generation_mode="vertex_ai",
            )
            return {
                "answer": annotated_answer,
                "citations": citations,
                "route_tier": RouteTier.STRONG.value,
                "fallback_triggered": False,
                "is_fallback": False,
                "generation_mode": "vertex_ai",
                "error": None,
            }

        try:
            gen_result = await deps.router.ainvoke(
                messages,
                task_type=TaskType.GENERATION,
                force_tier=RouteTier.STRONG,
                retry_count=retry_count,
            )
            annotated_answer, citations = _extract_citations(gen_result.content, docs)
            gen_mode = str(
                getattr(gen_result, "generation_mode", "")
                or ("demo_synthesis" if getattr(gen_result, "mocked", False) else "live_llm")
            )
            is_fb = bool(getattr(gen_result, "is_fallback", False))
            if is_fb:
                gen_mode = "fallback"
            update_request_metrics(generation_mode=gen_mode)
            return {
                "answer": annotated_answer,
                "citations": citations,
                "route_tier": RouteTier.STRONG.value,
                "fallback_triggered": is_fb,
                "is_fallback": is_fb,
                "generation_mode": gen_mode,
                "error": None,
            }
        except Exception as exc:
            logger.error(
                "Generation node failed; synthesizing degraded fallback answer",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            from app.adapters.mock.router import DemoLLMRouter

            fallback_text = DemoLLMRouter._synthesize_answer(
                messages,
                RouteTier.STRONG,
                pipeline_version=state.get("pipeline_version", "v3"),
            )
            annotated_answer, citations = _extract_citations(fallback_text, docs)
            update_request_metrics(
                route_tier=RouteTier.STRONG.value,
                generation_mode="fallback",
            )
            return {
                "answer": annotated_answer,
                "citations": citations,
                "route_tier": RouteTier.STRONG.value,
                "fallback_triggered": True,
                "is_fallback": True,
                "generation_mode": "fallback",
                "error": str(exc),
            }


async def hallucination_check_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Verify that the generated answer is strictly grounded in ``graded_documents``."""
    answer = (state.get("answer") or "").strip()
    docs = [coerce_retrieved_chunk(d) for d in (state.get("graded_documents") or [])]
    current_h_retries = int(state.get("hallucination_retries", 0))

    if not answer or not docs:
        return {
            "hallucination_passed": False,
            "hallucination_retries": current_h_retries + 1,
        }

    async with log_stage_latency("hallucination_check"):
        if deps.hallucination_checker is not None:
            grounded = bool(await deps.hallucination_checker(answer, docs))
        else:
            context_block = _format_context_block(docs)
            messages = [
                (
                    "system",
                    "You are a faithfulness judge. Return grounded=True only if the answer "
                    "is fully supported by the provided context documents without fabrication.",
                ),
                ("user", f"Context:\n{context_block}\n\nAnswer:\n{answer}"),
            ]
            try:
                verdict = await deps.router.ainvoke_structured(
                    HallucinationGradeOutput,
                    messages,
                    task_type=TaskType.HALLUCINATION_CHECK,
                    force_tier=RouteTier.STRONG,
                )
                grounded = bool(verdict.grounded)
            except Exception as exc:
                logger.warning(
                    "Hallucination check failed, checking citation provenance fallback",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                grounded = bool(state.get("citations"))

    if grounded:
        return {"hallucination_passed": True}

    return {
        "hallucination_passed": False,
        "hallucination_retries": current_h_retries + 1,
    }


async def cache_write_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Persist verified, grounded responses into the Redis semantic cache."""
    query = (state.get("query") or "").strip()
    answer = (state.get("answer") or "").strip()
    pipeline_version = state.get("pipeline_version", "v3")
    is_fallback = bool(state.get("is_fallback")) or bool(state.get("fallback_triggered"))
    gen_mode = str(state.get("generation_mode") or "")

    should_write = (
        deps.cache is not None
        and bool(query)
        and bool(answer)
        and bool(state.get("hallucination_passed"))
        and not bool(state.get("cache_hit"))
        and not is_fallback
        and gen_mode.lower() != "fallback"
        and is_cacheable_response(
            answer,
            metadata={"is_fallback": is_fallback, "generation_mode": gen_mode},
            is_fallback=is_fallback,
        )
    )

    if should_write:
        citations = [coerce_citation(c).model_dump() for c in (state.get("citations") or [])]
        metadata = {
            "route_tier": state.get("route_tier", RouteTier.STRONG.value),
            "generation_mode": gen_mode or "live_llm",
            "is_fallback": False,
            "citations": citations,
        }
        async with log_stage_latency("cache_write"):
            try:
                await deps.cache.set(
                    query,
                    answer,
                    namespace=pipeline_version,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Cache write node failed (fail-open)",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    return {"cache_hit": False}


async def fallback_node(state: RAGGraphState, deps: GraphDependencies) -> dict[str, Any]:
    """Emit a safe deterministic response when retries are exhausted or context is empty."""
    retry_count = int(state.get("retry_count", 0))
    logger.warning(
        "Entering LangGraph fallback node",
        retry_count=retry_count,
        max_retries=deps.max_retries,
        error=state.get("error"),
    )
    update_request_metrics(generation_mode="fallback")
    return {
        "answer": FALLBACK_ANSWER,
        "citations": [],
        "hallucination_passed": False,
        "fallback_triggered": True,
        "is_fallback": True,
        "generation_mode": "fallback",
    }


# ---------------------------------------------------------------------------
# Node Binding Helper
# ---------------------------------------------------------------------------


def bind_nodes(
    deps: GraphDependencies,
) -> dict[str, Callable[[RAGGraphState], Awaitable[dict[str, Any]]]]:
    """Bind ``GraphDependencies`` to all 9 LangGraph node callables."""

    async def _cache_lookup(state: RAGGraphState) -> dict[str, Any]:
        return await cache_lookup_node(state, deps)

    async def _router(state: RAGGraphState) -> dict[str, Any]:
        return await router_node(state, deps)

    async def _retrieve(state: RAGGraphState) -> dict[str, Any]:
        return await retrieve_node(state, deps)

    async def _grade_documents(state: RAGGraphState) -> dict[str, Any]:
        return await grade_documents_node(state, deps)

    async def _rewrite_query(state: RAGGraphState) -> dict[str, Any]:
        return await rewrite_query_node(state, deps)

    async def _generate(state: RAGGraphState) -> dict[str, Any]:
        return await generate_node(state, deps)

    async def _hallucination_check(state: RAGGraphState) -> dict[str, Any]:
        return await hallucination_check_node(state, deps)

    async def _cache_write(state: RAGGraphState) -> dict[str, Any]:
        return await cache_write_node(state, deps)

    async def _fallback(state: RAGGraphState) -> dict[str, Any]:
        return await fallback_node(state, deps)

    return {
        "cache_lookup": _cache_lookup,
        "router": _router,
        "retrieve": _retrieve,
        "grade_documents": _grade_documents,
        "rewrite_query": _rewrite_query,
        "generate": _generate,
        "hallucination_check": _hallucination_check,
        "cache_write": _cache_write,
        "fallback": _fallback,
    }


__all__: Sequence[str] = [
    "FALLBACK_ANSWER",
    "DocumentGradeOutput",
    "GraphDependencies",
    "HallucinationGradeOutput",
    "QueryRewriteOutput",
    "_build_rag_messages",
    "bind_nodes",
    "build_multilingual_system_prompt",
    "cache_lookup_node",
    "cache_write_node",
    "detect_query_language",
    "fallback_node",
    "generate_node",
    "grade_documents_node",
    "hallucination_check_node",
    "retrieve_node",
    "rewrite_query_node",
    "router_node",
]
