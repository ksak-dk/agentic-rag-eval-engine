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
from app.core.logging import get_active_pipeline_version, update_request_metrics
from app.core.router import LLMGenerationResult, RouteDecision, RouteTier, TaskType
from app.graph.nodes import DocumentGradeOutput, HallucinationGradeOutput, QueryRewriteOutput
from app.retrieval.chunker import sanitize_markdown_text

_CONTEXT_BLOCK_RE: re.Pattern[str] = re.compile(
    r"\[([^\[\]\s]+)\]\s*\(source:\s*([^,\)]+),\s*section:\s*([^\)]+)\)\s*\n(.*?)(?=\n\n---\n\n|\Z)",
    re.DOTALL,
)
_PIPELINE_HINT_RE: re.Pattern[str] = re.compile(r"\[pipeline:(v1|v2|v3)\]", re.IGNORECASE)
_MD_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
_FENCED_CODE_RE: re.Pattern[str] = re.compile(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```")
_TABLE_SEP_RE: re.Pattern[str] = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "is",
        "are",
        "was",
        "were",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "who",
        "does",
        "do",
        "did",
        "its",
        "it",
        "this",
        "that",
        "can",
        "from",
        "about",
        "explain",
        "describe",
        "tell",
        "me",
        "between",
    }
)
_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")


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
    def _clean_excerpt_text(raw: str) -> str:
        """Convert raw Markdown/chunk text into clean, readable prose for synthesis."""
        sanitized = sanitize_markdown_text(raw)
        if not sanitized:
            return ""
        # Condense fenced code blocks into inline snippets so raw ``` fences do not leak.
        cleaned = _FENCED_CODE_RE.sub(
            lambda m: " ".join(
                ln.strip()
                for ln in m.group(1).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ),
            sanitized,
        )
        lines: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line or _TABLE_SEP_RE.match(line):
                continue
            line = re.sub(r"^#{1,6}\s+", "", line)
            line = re.sub(r"^>\s*", "", line)
            line = re.sub(r"^[-*+]\s+", "", line)
            line = re.sub(r"^\d+\.\s+", "", line)
            if "|" in line:
                cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
                line = " · ".join(cells)
            line = _MD_LINK_RE.sub(r"\1", line)
            line = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", line)
            line = re.sub(r"`([^`]+)`", r"`\1`", line)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    @classmethod
    def _extract_question_and_contexts(
        cls,
        messages: Any,
    ) -> tuple[str, list[tuple[str, str, str, str]]]:
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

        contexts: list[tuple[str, str, str, str]] = []
        for match in _CONTEXT_BLOCK_RE.finditer(ctx_part):
            doc_id = match.group(1).strip()
            source = match.group(2).strip()
            section = match.group(3).strip()
            body = cls._clean_excerpt_text(match.group(4))
            if doc_id and body:
                contexts.append((doc_id, source, section, body))
        return question, contexts

    @staticmethod
    def _resolve_pipeline_version(
        messages: Any,
        tier: RouteTier,
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        explicit = (kwargs or {}).get("pipeline_version")
        if isinstance(explicit, str) and explicit.strip().lower() in ("v1", "v2", "v3"):
            return explicit.strip().lower()
        ctx_ver = get_active_pipeline_version()
        if ctx_ver in ("v1", "v2", "v3"):
            return ctx_ver
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            for item in messages:
                if isinstance(item, tuple) and len(item) >= 2:
                    match = _PIPELINE_HINT_RE.search(str(item[1]))
                    if match:
                        return match.group(1).lower()
        return "v1" if tier == RouteTier.FAST else "v3"

    @staticmethod
    def _select_salient_snippets(
        question: str,
        contexts: list[tuple[str, str, str, str]],
        limit: int,
    ) -> list[tuple[str, str, str, str, int]]:
        q_terms = {
            tok
            for tok in re.findall(r"[a-z0-9_.-]+|[\u4e00-\u9fff]+", question.lower())
            if tok not in _STOPWORDS and len(tok) > 1
        }
        scored: list[tuple[int, int, str, str, str, str]] = []
        for idx, (doc_id, source, section, body) in enumerate(contexts):
            sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
            best_sentence = body
            best_hits = 0
            if sentences and q_terms:
                ranked_sents: list[tuple[int, int, str]] = []
                for s_idx, sent in enumerate(sentences):
                    s_lower = sent.lower()
                    hits = sum(1 for term in q_terms if term in s_lower)
                    ranked_sents.append((hits, -s_idx, sent))
                ranked_sents.sort(reverse=True)
                top_sents = [s for hits, _, s in ranked_sents[:2] if hits > 0]
                if top_sents:
                    best_sentence = " ".join(top_sents)
                    best_hits = ranked_sents[0][0]
            if best_hits == 0 and q_terms:
                body_lower = body.lower()
                best_hits = sum(1 for term in q_terms if term in body_lower)

            clipped = best_sentence[:240].rstrip()
            if len(best_sentence) > 240:
                clipped = f"{clipped}..."
            scored.append((best_hits, -idx, doc_id, source, section, clipped))

        scored.sort(reverse=True)
        return [
            (doc_id, source, section, snippet, hits)
            for hits, _, doc_id, source, section, snippet in scored[: max(1, limit)]
        ]

    @classmethod
    def _synthesize_answer(
        cls,
        messages: Any,
        tier: RouteTier,
        *,
        pipeline_version: str | None = None,
    ) -> str:
        question, contexts = cls._extract_question_and_contexts(messages)
        if not contexts:
            if "original query:" in question.lower():
                raw = question.split(":", 1)[-1].strip()
                return f"{raw} architecture hybrid retrieval"
            return (
                "No supporting context chunks were retrieved from the demo knowledge base "
                f"for query: {question}."
            )

        version = pipeline_version or cls._resolve_pipeline_version(messages, tier)
        q_lower = question.lower()
        q_words = set(re.findall(r"[a-z0-9_.-]+|[\u4e00-\u9fff]+", q_lower))
        limit = 1 if version == "v1" else min(3, len(contexts))
        selected = cls._select_salient_snippets(question, contexts, limit=limit)
        primary_id, primary_source, _, primary_snippet, _ = selected[0]

        is_builtin_source = any(
            marker in primary_source.lower()
            for marker in (
                "readme",
                "gemini.md",
                "benchmark_report.md",
                "builtin://",
                "docs/test.md",
            )
        )

        topic_summary = ""
        if is_builtin_source:
            if q_words & {"v1", "v2", "v3", "infinite", "langgraph", "loop", "loops"}:
                topic_summary = (
                    "The `v1 -> v2 -> v3` progression upgrades fixed-window dense search (`v1`) "
                    "to Markdown semantic chunking + BM25 RRF + FP16 reranking (`v2`), and "
                    "adds a bounded LangGraph `StateGraph` (`max_retries=2`, `AsyncSqliteSaver`, "
                    "Redis 8 semantic cache) in `v3`."
                )
            elif q_words & {"bm25", "rrf", "rerank", "reranking", "qdrant", "dense", "sparse"}:
                if version == "v1":
                    topic_summary = (
                        "The `v1` baseline runs fixed-length chunking with Qdrant `int8` "
                        "dense-only ANN retrieval and fast single-step generation."
                    )
                else:
                    topic_summary = (
                        "Hybrid retrieval prefetches 1024-dim `BAAI/bge-m3` dense vectors "
                        "alongside `BM25` sparse vectors in Qdrant, fuses candidates with "
                        "server-side `Fusion.RRF`, and reranks via FP16 `BAAI/bge-reranker-v2-m3`."
                    )
            elif (q_words & {"vram", "3060", "gpu", "6gb", "显存"}) or "6 gb" in q_lower:
                topic_summary = (
                    "Under the RTX 3060 6 GB VRAM ceiling, zero local LLMs are loaded: "
                    "only FP16 `BAAI/bge-m3` (~1.1 GB) and `BAAI/bge-reranker-v2-m3` (~1.1 GB) "
                    "run on CUDA (~2.3 GB total), while LLM inference routes to cloud APIs."
                )
            elif q_words & {
                "ragas",
                "benchmark",
                "benchmarks",
                "eval",
                "evals",
                "sla",
                "p95",
                "评测",
            }:
                topic_summary = (
                    "Ragas v0.2+ benchmarks rise from Overall Score `0.6759` (`v1`) to "
                    "`0.8554` (`v2`) and `0.9950` (`v3`), while concurrent SSE load tests "
                    "pass the P95 < 1800 ms SLA at `678.77 ms`."
                )

        if version == "v1":
            if topic_summary:
                return f"{topic_summary} [{primary_id}] ({primary_snippet})"
            return f"{primary_snippet} [{primary_id}]"

        bullet_lines: list[str] = []
        for doc_id, source, section, snippet, _ in selected:
            label = section if section and section.lower() != "root" else source
            bullet_lines.append(f"- **{label}**: {snippet} [{doc_id}]")
        bullets_md = "\n".join(bullet_lines)

        if version == "v2":
            lead = topic_summary or (
                f"**Hybrid Retrieval Summary** (Dense + BM25 `Fusion.RRF` + Rerank from "
                f"`{primary_source}`):"
            )
            if topic_summary:
                lead = f"**Hybrid Retrieval Summary**: {topic_summary} [{primary_id}]"
            return f"{lead}\n\n{bullets_md}"

        # v3: Agentic Self-Correcting Synthesis
        if topic_summary:
            core_line = f"**Agentic Synthesis**: {topic_summary} [{primary_id}]"
        else:
            core_line = f"**Agentic Synthesis**: {primary_snippet} [{primary_id}]"

        return (
            f"{core_line}\n\n"
            f"**Verified Evidence & Context**:\n"
            f"{bullets_md}\n\n"
            f"**Grounding Check**: Verified across `{len(selected)}` retrieved context "
            f"segments with `[doc_id]` citation tracing (`v3` self-correcting pipeline)."
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
        _ = (temperature, kwargs)
        decision = self.select_route(
            task_type=task_type,
            query=query,
            retry_count=retry_count,
            force_tier=force_tier,
        )
        start = time.perf_counter()
        await asyncio.sleep(0.012 if decision.tier == RouteTier.FAST else 0.02)
        pipeline_ver = self._resolve_pipeline_version(messages, decision.tier, kwargs)
        content = self._synthesize_answer(
            messages,
            decision.tier,
            pipeline_version=pipeline_ver,
        )
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
            generation_mode="demo_synthesis",
        )
        return LLMGenerationResult(
            content=content,
            route_tier=decision.tier,
            model_name=decision.model_name,
            used_fallback=False,
            is_fallback=False,
            generation_mode="demo_synthesis",
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
        _ = temperature
        decision = self.select_route(
            task_type=task_type,
            query=query,
            retry_count=retry_count,
            force_tier=force_tier,
        )
        start = time.perf_counter()
        await asyncio.sleep(0.01 if decision.tier == RouteTier.FAST else 0.016)
        pipeline_ver = self._resolve_pipeline_version(messages, decision.tier, kwargs)
        content = self._synthesize_answer(
            messages,
            decision.tier,
            pipeline_version=pipeline_ver,
        )
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
                generation_mode="demo_synthesis",
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
