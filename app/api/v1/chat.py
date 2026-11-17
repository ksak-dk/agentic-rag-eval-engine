"""
app/api/v1/chat.py
==================
FastAPI v1 endpoints for dual-channel SSE streaming and synchronous RAG chat.

Endpoints
---------
- ``POST /api/v1/chat``   — Routes requests across ``v1`` / ``v2`` / ``v3`` pipelines
                            with dual-channel SSE (``token``, ``citations``,
                            ``metrics``, ``done``, ``error``) or JSON responses.
- ``GET  /api/v1/health`` — Reports active ``pipeline_version``, CUDA availability,
                            VRAM allocation, and component readiness.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sse_starlette.sse import EventSourceResponse

from app.core.config import Settings, get_settings
from app.core.logging import (
    get_request_metrics,
    log_stage_latency,
    request_metrics_context,
    sample_cuda_vram_mb,
    update_request_metrics,
)
from app.core.router import RouteTier, TaskType
from app.graph.nodes import (
    FALLBACK_ANSWER,
    GraphDependencies,
    _extract_citations,
    _format_context_block,
)
from app.graph.state import (
    Citation,
    RAGGraphState,
    coerce_citation,
    coerce_retrieved_chunk,
    make_initial_state,
)
from app.graph.workflow import AgenticRAGWorkflow, compile_workflow
from app.retrieval.hybrid_store import RetrievedChunk

try:
    import torch
except ImportError:  # pragma: no cover - defensive fallback for minimal envs
    torch = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

PipelineLiteral = Literal["v1", "v2", "v3"]
_STREAM_TOKEN_RE: re.Pattern[str] = re.compile(r"\S+\s*|\s+")

router = APIRouter(prefix="/api/v1", tags=["chat"])


# ---------------------------------------------------------------------------
# Pydantic V2 Request & Response Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Validated chat request payload supporting v1/v2/v3 pipeline selection."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User natural-language query.")
    session_id: str | None = Field(
        default=None,
        description="Optional conversation or LangGraph thread identifier.",
    )
    pipeline_version: PipelineLiteral | None = Field(
        default=None,
        description="Optional pipeline override ('v1', 'v2', or 'v3').",
    )
    stream: bool = Field(
        default=True,
        description="When True, returns dual-channel SSE; otherwise returns JSON.",
    )

    @field_validator("query")
    @classmethod
    def validate_non_blank_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty or whitespace-only.")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ChatResponse(BaseModel):
    """Synchronous JSON chat response with citations and RequestMetrics telemetry."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(description="Generated assistant answer.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Structured provenance citations attached to the answer.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-request telemetry serialized from RequestMetrics.to_dict().",
    )
    session_id: str = Field(description="Active session / LangGraph thread identifier.")
    pipeline_version: PipelineLiteral = Field(
        default="v3",
        description="Pipeline variant that handled the request.",
    )


class HealthResponse(BaseModel):
    """Service readiness, CUDA VRAM status, and active pipeline reporting."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"] = Field(
        default="ok",
        description="Overall service readiness status.",
    )
    pipeline_version: PipelineLiteral = Field(
        default="v3",
        description="Configured default pipeline version.",
    )
    cuda_available: bool = Field(
        default=False,
        description="Whether CUDA GPU acceleration is available.",
    )
    vram_allocated_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Currently allocated CUDA memory in megabytes.",
    )
    components: dict[str, bool] = Field(
        default_factory=dict,
        description="Readiness status of retriever, router, cache, and checkpointer.",
    )


# ---------------------------------------------------------------------------
# Dependency Injection Providers (app.state & app.dependency_overrides)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChatServiceDependencies:
    """Runtime services resolved per request via FastAPI dependency injection."""

    settings: Settings | None = None
    retriever: Any = None
    router: Any = None
    cache: Any | None = None
    checkpointer: Any | None = None
    workflow: Any | None = None
    checkpoint_db: Path | None = None


def _try_get_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


def get_api_settings(request: Request) -> Settings | None:
    """Resolve ``Settings`` from ``app.state`` or fall back to cached settings."""
    state_settings = getattr(request.app.state, "settings", None)
    if state_settings is not None:
        return cast(Settings, state_settings)
    return _try_get_settings()


def get_retriever(request: Request) -> Any:
    """Resolve the hybrid/dense retriever from ``app.state``."""
    return getattr(request.app.state, "retriever", None)


def get_llm_router(request: Request) -> Any:
    """Resolve the cloud ``LLMRouter`` from ``app.state``."""
    return getattr(request.app.state, "router", None)


def get_semantic_cache(request: Request) -> Any:
    """Resolve the optional Redis ``SemanticCache`` from ``app.state``."""
    return getattr(request.app.state, "cache", None)


def get_checkpointer(request: Request) -> Any:
    """Resolve the optional ``AsyncSqliteSaver`` checkpointer from ``app.state``."""
    return getattr(request.app.state, "checkpointer", None)


def get_workflow(request: Request) -> Any:
    """Resolve the optional preconfigured ``AgenticRAGWorkflow`` from ``app.state``."""
    return getattr(request.app.state, "workflow", None)


def get_checkpoint_db(request: Request) -> Path | None:
    """Resolve an optional checkpoint SQLite path override from ``app.state``."""
    raw = getattr(request.app.state, "checkpoint_db", None)
    return Path(raw) if raw is not None else None


def get_chat_dependencies(
    settings: Settings | None = Depends(get_api_settings),
    retriever: Any = Depends(get_retriever),
    llm_router: Any = Depends(get_llm_router),
    cache: Any = Depends(get_semantic_cache),
    checkpointer: Any = Depends(get_checkpointer),
    workflow: Any = Depends(get_workflow),
    checkpoint_db: Path | None = Depends(get_checkpoint_db),
) -> ChatServiceDependencies:
    """Aggregate service dependencies so tests may override individually or together."""
    return ChatServiceDependencies(
        settings=settings,
        retriever=retriever,
        router=llm_router,
        cache=cache,
        checkpointer=checkpointer,
        workflow=workflow,
        checkpoint_db=checkpoint_db,
    )


# ---------------------------------------------------------------------------
# Private Pipeline Helpers
# ---------------------------------------------------------------------------


def _is_cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_pipeline_version(
    requested: PipelineLiteral | None,
    settings: Settings | None,
) -> PipelineLiteral:
    if requested is not None:
        return requested
    if settings is not None:
        value = settings.pipeline_version.value
        if value in ("v1", "v2", "v3"):
            return cast(PipelineLiteral, value)
    return "v3"


def _validate_pipeline_dependencies(
    version: PipelineLiteral,
    deps: ChatServiceDependencies,
) -> None:
    if version in ("v1", "v2"):
        if deps.retriever is None or deps.router is None:
            raise RuntimeError(
                f"Pipeline '{version}' requires both 'retriever' and 'router' services."
            )
        return

    if deps.workflow is None and (deps.retriever is None or deps.router is None):
        raise RuntimeError(
            "Pipeline 'v3' requires either a configured 'workflow' or both "
            "'retriever' and 'router' services."
        )


def _resolve_model_name_for_tier(route_tier: str, deps: ChatServiceDependencies) -> str:
    settings = deps.settings or getattr(deps.router, "settings", None)
    if route_tier == RouteTier.STRONG.value:
        return settings.strong_llm_model if settings is not None else "gemini-2.5-flash"
    return settings.fast_llm_model if settings is not None else "llama-3.1-8b-instant"


def _build_rag_messages(query: str, docs: list[RetrievedChunk]) -> list[tuple[str, str]]:
    context_block = _format_context_block(docs)
    if context_block:
        return [
            (
                "system",
                "Answer the user question using ONLY the provided context snippets. "
                "Cite supporting sources inline using their exact [doc_id] tag.",
            ),
            ("user", f"Question: {query}\n\nContext:\n{context_block}"),
        ]
    return [
        (
            "system",
            "Answer the user question concisely. If insufficient context is available, "
            "state that clearly.",
        ),
        ("user", f"Question: {query}"),
    ]


def _sync_metrics_from_generation(gen_result: Any) -> None:
    raw_tier = getattr(gen_result, "route_tier", RouteTier.FAST)
    tier_val = raw_tier.value if hasattr(raw_tier, "value") else str(raw_tier)
    updates: dict[str, Any] = {
        "route_tier": tier_val,
        "model_name": str(getattr(gen_result, "model_name", "")),
        "cache_hit": False,
    }
    ttft_ms = getattr(gen_result, "ttft_ms", None)
    if ttft_ms is not None:
        updates["ttft_ms"] = float(ttft_ms)
    prompt_tokens = getattr(gen_result, "prompt_tokens", 0)
    if prompt_tokens:
        updates["prompt_tokens"] = int(prompt_tokens)
    completion_tokens = getattr(gen_result, "completion_tokens", 0)
    if completion_tokens:
        updates["completion_tokens"] = int(completion_tokens)
    update_request_metrics(**updates)


def _sync_metrics_from_v3_state(
    state: RAGGraphState,
    session_id: str,
    deps: ChatServiceDependencies,
) -> None:
    route_tier = str(state.get("route_tier") or RouteTier.FAST.value)
    cache_hit = bool(state.get("cache_hit", False))
    retry_count = int(state.get("retry_count", 0))
    explicit_model = state.get("model_name")
    if isinstance(explicit_model, str) and explicit_model:
        model_name = explicit_model
    elif cache_hit:
        model_name = ""
    else:
        model_name = _resolve_model_name_for_tier(route_tier, deps)

    update_request_metrics(
        session_id=session_id,
        route_tier=route_tier,
        model_name=model_name,
        cache_hit=cache_hit,
        retry_count=retry_count,
    )


def _tokenize_for_stream(text: str) -> list[str]:
    if not text:
        return [""]
    tokens = _STREAM_TOKEN_RE.findall(text)
    return tokens if tokens else [text]


def _supports_astream(llm_router: Any) -> bool:
    astream_fn = getattr(llm_router, "astream", None)
    return callable(astream_fn)


def _format_sse_event(event: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "event": event,
        "data": json.dumps(payload, ensure_ascii=False),
    }


async def _retrieve_chunks(
    query: str,
    version: Literal["v1", "v2"],
    deps: ChatServiceDependencies,
) -> list[RetrievedChunk]:
    top_k = deps.settings.retrieval_top_k if deps.settings is not None else 5
    top_n = deps.settings.retrieval_top_n if deps.settings is not None else 20
    async with log_stage_latency("retrieve", pipeline_version=version):
        raw_docs = await deps.retriever.search(
            query,
            pipeline_version=version,
            top_k=top_k,
            top_n=top_n,
        )
        return [coerce_retrieved_chunk(doc) for doc in raw_docs]


async def _run_v1_or_v2_sync(
    query: str,
    version: Literal["v1", "v2"],
    deps: ChatServiceDependencies,
) -> tuple[str, list[Citation]]:
    docs = await _retrieve_chunks(query, version, deps)
    messages = _build_rag_messages(query, docs)
    force_tier = RouteTier.FAST if version == "v1" else RouteTier.STRONG

    async with log_stage_latency("generate", pipeline_version=version, context_chunks=len(docs)):
        gen_result = await deps.router.ainvoke(
            messages,
            task_type=TaskType.GENERATION,
            force_tier=force_tier,
        )
        _sync_metrics_from_generation(gen_result)

    annotated_answer, citations = _extract_citations(str(gen_result.content), docs)
    return annotated_answer, citations


async def _run_v3_sync(
    query: str,
    session_id: str,
    deps: ChatServiceDependencies,
) -> tuple[str, list[Citation]]:
    top_k = deps.settings.retrieval_top_k if deps.settings is not None else 5
    top_n = deps.settings.retrieval_top_n if deps.settings is not None else 20
    max_retries = deps.settings.max_retries if deps.settings is not None else 2
    checkpoint_db = deps.checkpoint_db or (
        deps.settings.langgraph_checkpoint_db if deps.settings is not None else None
    )

    async with log_stage_latency("workflow_v3", pipeline_version="v3"):
        if deps.workflow is not None:
            state = await deps.workflow.ainvoke(
                query,
                session_id=session_id,
                pipeline_version="v3",
            )
        elif deps.checkpointer is not None:
            graph_deps = GraphDependencies(
                retriever=deps.retriever,
                router=deps.router,
                cache=deps.cache,
                top_k=top_k,
                top_n=top_n,
                max_retries=max_retries,
            )
            compiled = compile_workflow(
                graph_deps,
                checkpointer=deps.checkpointer,
                max_retries=max_retries,
            )
            initial_state = make_initial_state(
                query,
                session_id=session_id,
                pipeline_version="v3",
            )
            state = await compiled.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
            )
        else:
            graph_deps = GraphDependencies(
                retriever=deps.retriever,
                router=deps.router,
                cache=deps.cache,
                top_k=top_k,
                top_n=top_n,
                max_retries=max_retries,
            )
            workflow = AgenticRAGWorkflow(
                graph_deps,
                db_path=checkpoint_db,
                max_retries=max_retries,
            )
            state = await workflow.ainvoke(
                query,
                session_id=session_id,
                pipeline_version="v3",
            )

    _sync_metrics_from_v3_state(state, session_id, deps)
    answer = str(state.get("answer") or FALLBACK_ANSWER)
    citations = [coerce_citation(c) for c in (state.get("citations") or [])]
    return answer, citations


async def _stream_v1_or_v2_live(
    query: str,
    version: Literal["v1", "v2"],
    session_id: str,
    deps: ChatServiceDependencies,
    request_start: float,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ``("token", text)`` events followed by a final ``("citations", citations)``."""
    docs = await _retrieve_chunks(query, version, deps)
    messages = _build_rag_messages(query, docs)
    force_tier = RouteTier.FAST if version == "v1" else RouteTier.STRONG
    raw_parts: list[str] = []
    first_token_ms: float | None = None

    async with log_stage_latency("generate", pipeline_version=version, context_chunks=len(docs)):
        async for chunk in deps.router.astream(
            messages,
            task_type=TaskType.GENERATION,
            force_tier=force_tier,
        ):
            piece = str(chunk)
            if not piece:
                continue
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
            raw_parts.append(piece)
            yield ("token", piece)

        raw_answer = "".join(raw_parts)
        annotated_answer, citations = _extract_citations(raw_answer, docs)
        if annotated_answer != raw_answer and annotated_answer.startswith(raw_answer):
            suffix = annotated_answer[len(raw_answer) :]
            if suffix:
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
                yield ("token", suffix)

        current = get_request_metrics()
        metric_updates: dict[str, Any] = {"cache_hit": False}
        if not current.route_tier:
            metric_updates["route_tier"] = force_tier.value
        if not current.model_name:
            metric_updates["model_name"] = _resolve_model_name_for_tier(force_tier.value, deps)
        if current.ttft_ms is None and first_token_ms is not None:
            metric_updates["ttft_ms"] = first_token_ms
        update_request_metrics(**metric_updates)

    yield ("citations", citations)


# ---------------------------------------------------------------------------
# FastAPI Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(
    deps: ChatServiceDependencies = Depends(get_chat_dependencies),
) -> HealthResponse:
    """Return service health, CUDA/VRAM telemetry, and dependency readiness."""
    default_version = _resolve_pipeline_version(None, deps.settings)
    cuda_available = _is_cuda_available()
    vram_mb = sample_cuda_vram_mb()
    components: dict[str, bool] = {
        "retriever": deps.retriever is not None,
        "router": deps.router is not None,
        "cache": deps.cache is not None,
        "checkpointer": deps.checkpointer is not None,
    }
    overall_status: Literal["ok", "degraded"] = (
        "ok" if components["retriever"] and components["router"] else "degraded"
    )
    return HealthResponse(
        status=overall_status,
        pipeline_version=default_version,
        cuda_available=cuda_available,
        vram_allocated_mb=vram_mb,
        components=components,
    )


@router.post("/chat", response_model=None)
async def chat_endpoint(
    payload: ChatRequest,
    deps: ChatServiceDependencies = Depends(get_chat_dependencies),
) -> ChatResponse | EventSourceResponse:
    """Execute v1/v2/v3 RAG pipeline and emit dual-channel SSE or JSON response."""
    version = _resolve_pipeline_version(payload.pipeline_version, deps.settings)
    session_id = payload.session_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    if not payload.stream:
        try:
            _validate_pipeline_dependencies(version, deps)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        async with request_metrics_context(request_id=request_id, session_id=session_id):
            try:
                async with log_stage_latency(
                    "chat_request",
                    update_total_latency=True,
                    pipeline_version=version,
                ):
                    if version in ("v1", "v2"):
                        answer, citations = await _run_v1_or_v2_sync(
                            payload.query,
                            cast(Literal["v1", "v2"], version),
                            deps,
                        )
                    else:
                        answer, citations = await _run_v3_sync(
                            payload.query,
                            session_id,
                            deps,
                        )
            except Exception as exc:
                logger.error(
                    "Chat request failed",
                    pipeline_version=version,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Chat execution failed: {exc}",
                ) from exc

            metrics_dict = get_request_metrics().to_dict()
            return ChatResponse(
                answer=answer,
                citations=citations,
                metrics=metrics_dict,
                session_id=session_id,
                pipeline_version=version,
            )

    async def _event_generator() -> AsyncIterator[dict[str, str]]:
        request_start = time.perf_counter()
        citations: list[Citation] = []

        async with request_metrics_context(request_id=request_id, session_id=session_id):
            try:
                async with log_stage_latency(
                    "chat_request",
                    update_total_latency=True,
                    pipeline_version=version,
                ):
                    _validate_pipeline_dependencies(version, deps)

                    if version in ("v1", "v2") and _supports_astream(deps.router):
                        async for item_kind, item_value in _stream_v1_or_v2_live(
                            payload.query,
                            cast(Literal["v1", "v2"], version),
                            session_id,
                            deps,
                            request_start,
                        ):
                            if item_kind == "token":
                                yield _format_sse_event(
                                    "token",
                                    {"token": str(item_value), "session_id": session_id},
                                )
                            elif item_kind == "citations":
                                citations = cast(list[Citation], item_value)
                    else:
                        if version in ("v1", "v2"):
                            answer, citations = await _run_v1_or_v2_sync(
                                payload.query,
                                cast(Literal["v1", "v2"], version),
                                deps,
                            )
                        else:
                            answer, citations = await _run_v3_sync(
                                payload.query,
                                session_id,
                                deps,
                            )

                        if get_request_metrics().ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - request_start) * 1000.0, 2)
                            update_request_metrics(ttft_ms=ttft_ms)

                        for token in _tokenize_for_stream(answer):
                            yield _format_sse_event(
                                "token",
                                {"token": token, "session_id": session_id},
                            )

                yield _format_sse_event(
                    "citations",
                    {
                        "citations": [c.model_dump() for c in citations],
                        "session_id": session_id,
                    },
                )
                yield _format_sse_event("metrics", get_request_metrics().to_dict())
                yield _format_sse_event(
                    "done",
                    {
                        "status": "completed",
                        "session_id": session_id,
                        "pipeline_version": version,
                    },
                )
            except Exception as exc:
                logger.error(
                    "SSE chat stream failed",
                    pipeline_version=version,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                yield _format_sse_event(
                    "error",
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "session_id": session_id,
                    },
                )

    return EventSourceResponse(_event_generator())


__all__: Sequence[str] = [
    "ChatRequest",
    "ChatResponse",
    "ChatServiceDependencies",
    "HealthResponse",
    "PipelineLiteral",
    "chat_endpoint",
    "get_api_settings",
    "get_chat_dependencies",
    "get_checkpoint_db",
    "get_checkpointer",
    "get_llm_router",
    "get_retriever",
    "get_semantic_cache",
    "get_workflow",
    "health_check",
    "router",
]
