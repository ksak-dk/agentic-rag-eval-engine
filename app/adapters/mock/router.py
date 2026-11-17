"""
app/adapters/mock/router.py
===========================
Deterministic demo/offline LLM router adapter implementing ``LLMRouterPort``.

Handles route selection, grounded synthesis with ``[doc_id]`` inline citations,
async token streaming, and structured outputs for LangGraph ``grade_documents``,
``rewrite_query``, and ``hallucination_check`` nodes without cloud API calls.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.core.config import Settings
from app.core.logging import update_request_metrics
from app.core.router import LLMGenerationResult, RouteDecision, RouteTier, TaskType
from app.graph.nodes import DocumentGradeOutput, HallucinationGradeOutput, QueryRewriteOutput

_CONTEXT_BLOCK_RE: re.Pattern[str] = re.compile(
    r"\[([^\[\]\s]+)\]\s*\(source:\s*([^,\)]+),\s*section:\s*([^\)]+)\)\s*\n(.*?)(?=\n\n---\n\n|\Z)",
    re.DOTALL,
)


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
            if "original query:" in question.lower():
                raw = question.split(":", 1)[-1].strip()
                return f"{raw} architecture hybrid retrieval"
            return (
                "No supporting context chunks were retrieved from the demo knowledge base "
                f"for query: {question}."
            )

        q_lower = question.lower()
        q_words = set(re.findall(r"[a-z0-9_.-]+|[\u4e00-\u9fff]+", q_lower))
        doc_ids = [doc_id for doc_id, _ in contexts]
        primary_id = doc_ids[0]
        secondary_id = doc_ids[1] if len(doc_ids) > 1 else primary_id

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
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMGenerationResult:
        _ = (temperature, kwargs)
        decision = self.select_route(
            task_type=task_type,
            query=query,
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
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        _ = (temperature, kwargs)
        decision = self.select_route(
            task_type=task_type,
            query=query,
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
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        _ = (task_type, query, force_tier, retry_count, temperature, kwargs)
        await asyncio.sleep(0.003)
        if schema is DocumentGradeOutput:
            return DocumentGradeOutput(relevant=True, reason="Verified relevant demo corpus chunk")
        if schema is HallucinationGradeOutput:
            return HallucinationGradeOutput(
                grounded=True,
                reason="Answer grounded in retrieved demo corpus citations",
            )
        if schema is QueryRewriteOutput:
            raw_question, _ = self._extract_question_and_contexts(messages)
            cleaned = raw_question.split(":", 1)[-1].strip() or "hybrid rag retrieval"
            return QueryRewriteOutput(rewritten_query=f"{cleaned} architecture")
        return schema.model_validate({})


__all__: Sequence[str] = ["DemoLLMRouter"]
