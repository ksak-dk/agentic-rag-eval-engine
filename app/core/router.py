"""
app/core/router.py
==================
Zero-cost dual-engine LLM router bridging Groq (fast route) and Google AI
Studio Gemini (fallback fast route & strong route).

Routing policy (aligned with GEMINI.md)
---------------------------------------
- Fast Route (primary)  : ``llama-3.1-8b-instant`` via Groq OpenAI-compatible API
                          (intent routing, query rewrite, simple QA, doc grading)
- Fast Route (fallback) : ``gemini-2.5-flash-lite`` via Google AI Studio
                          (automatic failover when Groq rate-limits or errors)
- Strong Route          : ``gemini-2.5-flash`` via Google AI Studio
                          (multi-hop generation, hallucination check, Ragas eval)

Hardware contract
-----------------
Never loads any local LLM on CPU or CUDA. RTX 3060 6 GB VRAM remains reserved
exclusively for local FP16 embedding and reranking models.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Any, TypeVar

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.core.config import Settings, get_settings
from app.core.logging import update_request_metrics

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class RouteTier(StrEnum):
    """Supported routing tiers for cloud LLM execution."""

    FAST = "fast"
    STRONG = "strong"


class TaskType(StrEnum):
    """Pipeline task categories mapped to fast or strong model tiers."""

    INTENT_ROUTING = "intent_routing"
    QUERY_REWRITE = "query_rewrite"
    SIMPLE_QA = "simple_qa"
    DOC_GRADING = "doc_grading"
    GENERATION = "generation"
    MULTI_HOP_QA = "multi_hop_qa"
    HALLUCINATION_CHECK = "hallucination_check"
    EVAL_JUDGE = "eval_judge"


_FAST_TASKS: frozenset[TaskType] = frozenset(
    {
        TaskType.INTENT_ROUTING,
        TaskType.QUERY_REWRITE,
        TaskType.SIMPLE_QA,
        TaskType.DOC_GRADING,
    }
)

_STRONG_TASKS: frozenset[TaskType] = frozenset(
    {
        TaskType.GENERATION,
        TaskType.MULTI_HOP_QA,
        TaskType.HALLUCINATION_CHECK,
        TaskType.EVAL_JUDGE,
    }
)

_COMPLEX_QUERY_MARKERS: tuple[str, ...] = (
    "compare",
    "contrast",
    "difference between",
    "tradeoff",
    "trade-off",
    "architecture",
    "step by step",
    "multi-hop",
    "synthesize",
    "evaluate",
    "analyze",
    "why does",
    "how does",
    "对比",
    "比较",
    "区别",
    "架构",
    "为什么",
    "分析",
    "综合",
)


class LLMRouterError(RuntimeError):
    """Raised when primary and fallback cloud LLM routes both fail."""


class RouteDecision(BaseModel):
    """Immutable routing decision emitted by ``LLMRouter.select_route``."""

    model_config = ConfigDict(frozen=True)

    tier: RouteTier = Field(description="Selected route tier (fast | strong).")
    model_name: str = Field(description="Primary model identifier for this route.")
    fallback_model_name: str | None = Field(
        default=None,
        description="Fallback model identifier used if primary route fails.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Task type that informed the routing decision.",
    )
    reason: str = Field(description="Human-readable rationale for route selection.")


class LLMGenerationResult(BaseModel):
    """Structured output returned by ``LLMRouter.ainvoke``."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="Generated assistant text.")
    route_tier: RouteTier = Field(description="Route tier actually executed.")
    model_name: str = Field(description="Concrete model identifier that produced output.")
    used_fallback: bool = Field(
        default=False,
        description="True when fast primary failed and fallback model answered.",
    )
    mocked: bool = Field(
        default=False,
        description="True when produced under MOCK_LLM_STREAM=true.",
    )
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time to first token in milliseconds.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total end-to-end LLM call latency in milliseconds.",
    )
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MessageInput = str | Sequence[BaseMessage | tuple[str, str] | dict[str, str]]


def _coerce_route_tier(tier: RouteTier | str | None) -> RouteTier | None:
    if tier is None or isinstance(tier, RouteTier):
        return tier
    return RouteTier(tier.strip().lower())


def _coerce_task_type(task_type: TaskType | str | None) -> TaskType | None:
    if task_type is None or isinstance(task_type, TaskType):
        return task_type
    return TaskType(task_type.strip().lower())


def _is_complex_query(query: str) -> bool:
    """Apply lightweight zero-cost heuristics to detect multi-hop/complex queries."""
    cleaned = query.strip()
    if not cleaned:
        return False
    if len(cleaned) > 160:
        return True
    if cleaned.count("?") + cleaned.count("\uff1f") >= 2:
        return True
    lowered = cleaned.lower()
    return any(marker in lowered for marker in _COMPLEX_QUERY_MARKERS)


def _normalize_messages(messages: MessageInput) -> list[BaseMessage]:
    """Convert string, tuple, dict, or BaseMessage inputs into LangChain messages."""
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]

    normalized: list[BaseMessage] = []
    for item in messages:
        if isinstance(item, BaseMessage):
            normalized.append(item)
            continue

        if isinstance(item, tuple) and len(item) == 2:
            role, content = item[0].strip().lower(), str(item[1])
        elif isinstance(item, dict):
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", ""))
        else:
            raise TypeError(f"Unsupported message item type: {type(item).__name__}")

        if role in {"system", "developer"}:
            normalized.append(SystemMessage(content=content))
        elif role in {"assistant", "ai"}:
            normalized.append(AIMessage(content=content))
        else:
            normalized.append(HumanMessage(content=content))

    return normalized


def _extract_last_query_text(messages: list[BaseMessage]) -> str | None:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _extract_text(msg)
    if messages:
        return _extract_text(messages[-1])
    return None


def _extract_text(message_or_chunk: Any) -> str:
    """Safely extract string content from LangChain messages, chunks, or parts."""
    if message_or_chunk is None:
        return ""
    if isinstance(message_or_chunk, str):
        return message_or_chunk

    content = getattr(message_or_chunk, "content", message_or_chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "".join(parts)
    return str(content) if content is not None else ""


def _extract_token_usage(message: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) across OpenAI and Gemini metadata."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        return max(0, prompt), max(0, completion)

    resp_meta = getattr(message, "response_metadata", None)
    if isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage_metadata")
        if isinstance(token_usage, dict):
            prompt = int(
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
                or token_usage.get("prompt_token_count")
                or 0
            )
            completion = int(
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or token_usage.get("candidates_token_count")
                or 0
            )
            return max(0, prompt), max(0, completion)

    return 0, 0


def _build_mock_structured_instance(schema: type[T]) -> T:
    """Construct a valid mock instance of a Pydantic V2 model for CI mock mode."""
    try:
        return schema.model_validate({})
    except Exception:
        synthetic: dict[str, Any] = {}
        for field_name, field_info in schema.model_fields.items():
            if not field_info.is_required():
                continue
            ann = field_info.annotation
            if ann is bool:
                synthetic[field_name] = True
            elif ann is int:
                synthetic[field_name] = 0
            elif ann is float:
                synthetic[field_name] = 0.0
            elif ann is str:
                synthetic[field_name] = "mock"
            elif ann is list:
                synthetic[field_name] = []
            elif ann is dict:
                synthetic[field_name] = {}
            else:
                synthetic[field_name] = None
        return schema.model_validate(synthetic)


# ---------------------------------------------------------------------------
# Dual-Engine LLM Router
# ---------------------------------------------------------------------------


class LLMRouter:
    """Zero-cost dual-engine cloud LLM router with automatic Groq->Gemini fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        mock_ttft_seconds: float = 0.3,
        request_timeout: float = 30.0,
        fast_model: Runnable[Any, Any] | None = None,
        fallback_model: Runnable[Any, Any] | None = None,
        strong_model: Runnable[Any, Any] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._mock_ttft_seconds = max(0.0, mock_ttft_seconds)
        self._request_timeout = max(1.0, request_timeout)
        self._fast_override = fast_model
        self._fallback_override = fallback_model
        self._strong_override = strong_model

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
        """Choose between ``RouteTier.FAST`` and ``RouteTier.STRONG``."""
        resolved_force = _coerce_route_tier(force_tier)
        resolved_task = _coerce_task_type(task_type)

        if resolved_force == RouteTier.STRONG:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name=self._settings.strong_llm_model,
                fallback_model_name=None,
                task_type=resolved_task,
                reason="Explicitly forced strong route",
            )
        if resolved_force == RouteTier.FAST:
            return RouteDecision(
                tier=RouteTier.FAST,
                model_name=self._settings.fast_llm_model,
                fallback_model_name=self._settings.fast_llm_fallback_model,
                task_type=resolved_task,
                reason="Explicitly forced fast route",
            )

        if retry_count > 0:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name=self._settings.strong_llm_model,
                fallback_model_name=None,
                task_type=resolved_task,
                reason=f"Escalated to strong route on retry_count={retry_count}",
            )

        if resolved_task in _STRONG_TASKS:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name=self._settings.strong_llm_model,
                fallback_model_name=None,
                task_type=resolved_task,
                reason=f"Task '{resolved_task.value}' requires strong model",
            )

        if resolved_task in _FAST_TASKS:
            return RouteDecision(
                tier=RouteTier.FAST,
                model_name=self._settings.fast_llm_model,
                fallback_model_name=self._settings.fast_llm_fallback_model,
                task_type=resolved_task,
                reason=f"Task '{resolved_task.value}' assigned to fast route",
            )

        if query and _is_complex_query(query):
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name=self._settings.strong_llm_model,
                fallback_model_name=None,
                task_type=resolved_task,
                reason="Query complexity heuristic matched strong route",
            )

        return RouteDecision(
            tier=RouteTier.FAST,
            model_name=self._settings.fast_llm_model,
            fallback_model_name=self._settings.fast_llm_fallback_model,
            task_type=resolved_task,
            reason="Default fast route",
        )

    def _build_fast_primary(self, temperature: float) -> Runnable[Any, Any]:
        if self._fast_override is not None:
            return self._fast_override
        return ChatOpenAI(
            model=self._settings.fast_llm_model,
            api_key=SecretStr(self._settings.groq_api_key),
            base_url=self._settings.groq_base_url,
            temperature=temperature,
            max_retries=1,
            request_timeout=self._request_timeout,
        )

    def _build_fast_fallback(self, temperature: float) -> Runnable[Any, Any]:
        if self._fallback_override is not None:
            return self._fallback_override
        return ChatGoogleGenerativeAI(
            model=self._settings.fast_llm_fallback_model,
            google_api_key=self._settings.google_api_key,
            temperature=temperature,
            max_retries=1,
            timeout=self._request_timeout,
        )

    def _build_strong_model(self, temperature: float) -> Runnable[Any, Any]:
        if self._strong_override is not None:
            return self._strong_override
        return ChatGoogleGenerativeAI(
            model=self._settings.strong_llm_model,
            google_api_key=self._settings.google_api_key,
            temperature=temperature,
            max_retries=2,
            timeout=self._request_timeout,
        )

    def get_chat_model(
        self,
        tier: RouteTier | str = RouteTier.FAST,
        *,
        temperature: float = 0.0,
        with_fallback: bool = True,
    ) -> BaseChatModel | Runnable[Any, BaseMessage]:
        """Return a configured LangChain chat model (optionally wrapped with fallback)."""
        resolved_tier = _coerce_route_tier(tier) or RouteTier.FAST
        if resolved_tier == RouteTier.STRONG:
            return self._build_strong_model(temperature)

        primary = self._build_fast_primary(temperature)
        if with_fallback:
            fallback = self._build_fast_fallback(temperature)
            return primary.with_fallbacks([fallback])
        return primary

    def _resolve_decision_for_call(
        self,
        normalized_messages: list[BaseMessage],
        *,
        task_type: TaskType | str | None,
        force_tier: RouteTier | str | None,
        retry_count: int,
    ) -> RouteDecision:
        query = _extract_last_query_text(normalized_messages) if task_type is None else None
        return self.select_route(
            task_type=task_type,
            query=query,
            retry_count=retry_count,
            force_tier=force_tier,
        )

    async def ainvoke(
        self,
        messages: MessageInput,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> LLMGenerationResult:
        """Invoke the routed cloud LLM asynchronously with Groq->Gemini fallback."""
        normalized = _normalize_messages(messages)
        decision = self._resolve_decision_for_call(
            normalized,
            task_type=task_type,
            force_tier=force_tier,
            retry_count=retry_count,
        )
        start_time = time.perf_counter()

        if self._settings.mock_llm_stream:
            await asyncio.sleep(self._mock_ttft_seconds)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            preview = (_extract_last_query_text(normalized) or "query")[:60]
            content = f"[MOCK:{decision.model_name}] Simulated response for: {preview}"
            prompt_tokens = max(1, sum(len(_extract_text(m)) for m in normalized) // 4)
            completion_tokens = max(1, len(content) // 4)
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

        used_fallback = False
        active_model_name = decision.model_name
        response: Any

        if decision.tier == RouteTier.STRONG:
            strong_model = self._build_strong_model(temperature)
            try:
                response = await strong_model.ainvoke(normalized)
            except Exception as exc:
                logger.error(
                    "Strong LLM route failed",
                    model=active_model_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise LLMRouterError(
                    f"Strong LLM route '{active_model_name}' failed: {exc}"
                ) from exc
        else:
            primary_model = self._build_fast_primary(temperature)
            try:
                response = await primary_model.ainvoke(normalized)
            except Exception as primary_exc:
                fallback_name = self._settings.fast_llm_fallback_model
                logger.warning(
                    "Fast primary LLM failed, falling back to Gemini Flash-Lite",
                    primary_model=active_model_name,
                    fallback_model=fallback_name,
                    error_type=type(primary_exc).__name__,
                    error=str(primary_exc),
                )
                used_fallback = True
                active_model_name = fallback_name
                fallback_model = self._build_fast_fallback(temperature)
                try:
                    response = await fallback_model.ainvoke(normalized)
                except Exception as fallback_exc:
                    logger.error(
                        "Fast fallback LLM failed",
                        fallback_model=fallback_name,
                        error_type=type(fallback_exc).__name__,
                        error=str(fallback_exc),
                    )
                    raise LLMRouterError(
                        f"Fast route failed on both '{decision.model_name}' "
                        f"and fallback '{fallback_name}': {fallback_exc}"
                    ) from fallback_exc

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        content = _extract_text(response)
        prompt_tokens, completion_tokens = _extract_token_usage(response)
        update_request_metrics(
            route_tier=decision.tier.value,
            model_name=active_model_name,
            ttft_ms=elapsed_ms,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return LLMGenerationResult(
            content=content,
            route_tier=decision.tier,
            model_name=active_model_name,
            used_fallback=used_fallback,
            mocked=False,
            ttft_ms=elapsed_ms,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def astream(
        self,
        messages: MessageInput,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream text chunks asynchronously while tracking TTFT and total latency."""
        normalized = _normalize_messages(messages)
        decision = self._resolve_decision_for_call(
            normalized,
            task_type=task_type,
            force_tier=force_tier,
            retry_count=retry_count,
        )
        start_time = time.perf_counter()
        ttft_ms: float | None = None
        active_model_name = decision.model_name
        total_chars = 0

        if self._settings.mock_llm_stream:
            try:
                await asyncio.sleep(self._mock_ttft_seconds)
                preview = (_extract_last_query_text(normalized) or "query")[:40]
                mock_chunks = [
                    f"[MOCK:{decision.model_name}] ",
                    "Simulated streaming response ",
                    f"for: {preview}",
                ]
                for idx, piece in enumerate(mock_chunks):
                    if idx == 0:
                        ttft_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                    else:
                        await asyncio.sleep(0.005)
                    total_chars += len(piece)
                    yield piece
            finally:
                total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
                prompt_tokens = max(1, sum(len(_extract_text(m)) for m in normalized) // 4)
                completion_tokens = max(1, total_chars // 4)
                update_request_metrics(
                    route_tier=decision.tier.value,
                    model_name=active_model_name,
                    ttft_ms=ttft_ms,
                    latency_ms=total_latency,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            return

        async def _iter_model(model: Runnable[Any, Any]) -> AsyncIterator[str]:
            nonlocal ttft_ms, total_chars
            async for chunk in model.astream(normalized):
                text = _extract_text(chunk)
                if not text:
                    continue
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                total_chars += len(text)
                yield text

        try:
            if decision.tier == RouteTier.STRONG:
                strong_model = self._build_strong_model(temperature)
                try:
                    async for text in _iter_model(strong_model):
                        yield text
                except Exception as exc:
                    logger.error(
                        "Strong LLM stream failed",
                        model=active_model_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise LLMRouterError(
                        f"Strong LLM stream '{active_model_name}' failed: {exc}"
                    ) from exc
            else:
                primary_model = self._build_fast_primary(temperature)
                emitted_any = False
                try:
                    async for text in _iter_model(primary_model):
                        emitted_any = True
                        yield text
                except Exception as primary_exc:
                    if emitted_any:
                        logger.error(
                            "Fast primary LLM failed mid-stream",
                            primary_model=active_model_name,
                            error_type=type(primary_exc).__name__,
                            error=str(primary_exc),
                        )
                        raise LLMRouterError(
                            f"Fast primary stream '{active_model_name}' failed mid-stream: "
                            f"{primary_exc}"
                        ) from primary_exc

                    fallback_name = self._settings.fast_llm_fallback_model
                    logger.warning(
                        "Fast primary stream failed before first token, falling back",
                        primary_model=active_model_name,
                        fallback_model=fallback_name,
                        error_type=type(primary_exc).__name__,
                        error=str(primary_exc),
                    )
                    active_model_name = fallback_name
                    fallback_model = self._build_fast_fallback(temperature)
                    try:
                        async for text in _iter_model(fallback_model):
                            yield text
                    except Exception as fallback_exc:
                        logger.error(
                            "Fast fallback stream failed",
                            fallback_model=fallback_name,
                            error_type=type(fallback_exc).__name__,
                            error=str(fallback_exc),
                        )
                        raise LLMRouterError(
                            f"Fast stream failed on both '{decision.model_name}' "
                            f"and fallback '{fallback_name}': {fallback_exc}"
                        ) from fallback_exc
        finally:
            total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
            update_request_metrics(
                route_tier=decision.tier.value,
                model_name=active_model_name,
                ttft_ms=ttft_ms,
                latency_ms=total_latency,
            )

    async def ainvoke_structured(
        self,
        schema: type[T],
        messages: MessageInput,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> T:
        """Invoke the routed LLM and validate output against a Pydantic V2 schema."""
        normalized = _normalize_messages(messages)
        decision = self._resolve_decision_for_call(
            normalized,
            task_type=task_type,
            force_tier=force_tier,
            retry_count=retry_count,
        )
        start_time = time.perf_counter()

        if self._settings.mock_llm_stream:
            await asyncio.sleep(self._mock_ttft_seconds)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            update_request_metrics(
                route_tier=decision.tier.value,
                model_name=decision.model_name,
                ttft_ms=elapsed_ms,
                latency_ms=elapsed_ms,
            )
            return _build_mock_structured_instance(schema)

        active_model_name = decision.model_name
        result: Any

        if decision.tier == RouteTier.STRONG:
            strong_model = self._build_strong_model(temperature)
            structured_runnable = strong_model.with_structured_output(schema)  # type: ignore[union-attr]
            try:
                result = await structured_runnable.ainvoke(normalized)
            except Exception as exc:
                logger.error(
                    "Strong structured LLM call failed",
                    model=active_model_name,
                    schema=schema.__name__,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise LLMRouterError(
                    f"Strong structured route '{active_model_name}' failed: {exc}"
                ) from exc
        else:
            primary_model = self._build_fast_primary(temperature)
            structured_primary = primary_model.with_structured_output(schema)  # type: ignore[union-attr]
            try:
                result = await structured_primary.ainvoke(normalized)
            except Exception as primary_exc:
                fallback_name = self._settings.fast_llm_fallback_model
                logger.warning(
                    "Fast primary structured call failed, falling back to Gemini Flash-Lite",
                    primary_model=active_model_name,
                    fallback_model=fallback_name,
                    schema=schema.__name__,
                    error_type=type(primary_exc).__name__,
                    error=str(primary_exc),
                )
                active_model_name = fallback_name
                fallback_model = self._build_fast_fallback(temperature)
                structured_fallback = fallback_model.with_structured_output(schema)  # type: ignore[union-attr]
                try:
                    result = await structured_fallback.ainvoke(normalized)
                except Exception as fallback_exc:
                    logger.error(
                        "Fast fallback structured call failed",
                        fallback_model=fallback_name,
                        schema=schema.__name__,
                        error_type=type(fallback_exc).__name__,
                        error=str(fallback_exc),
                    )
                    raise LLMRouterError(
                        f"Fast structured route failed on both '{decision.model_name}' "
                        f"and fallback '{fallback_name}': {fallback_exc}"
                    ) from fallback_exc

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        update_request_metrics(
            route_tier=decision.tier.value,
            model_name=active_model_name,
            ttft_ms=elapsed_ms,
            latency_ms=elapsed_ms,
        )
        return result if isinstance(result, schema) else schema.model_validate(result)


ModelRouter = LLMRouter


def get_router(settings: Settings | None = None) -> LLMRouter:
    """Create an ``LLMRouter`` bound to the provided or singleton ``Settings``."""
    return LLMRouter(settings=settings)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__: Sequence[str] = [
    "LLMGenerationResult",
    "LLMRouter",
    "LLMRouterError",
    "ModelRouter",
    "RouteDecision",
    "RouteTier",
    "TaskType",
    "get_router",
]
