"""
app/ports/interfaces.py
=======================
Hexagonal architecture port contracts (``typing.Protocol``) decoupling domain
orchestration and FastAPI/LangGraph wiring from concrete infrastructure or
offline mock/heuristic adapters.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.cache.semantic_cache import CachedResponse, SingleFlight
    from app.core.router import LLMGenerationResult, RouteDecision
    from app.retrieval.hybrid_store import RetrievedChunk


@runtime_checkable
class RetrieverPort(Protocol):
    """Port contract for dense, hybrid, or in-memory retrieval backends."""

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        """Retrieve up to ``top_k`` ranked chunks for ``query``."""
        ...

    async def add_chunks(self, chunks: Sequence[Any]) -> int:
        """Incrementally upsert chunks into the retrieval store and return added/updated count."""
        ...

    def count_chunks(self) -> int:
        """Return the total number of indexed chunks currently held by the store."""
        ...


@runtime_checkable
class LLMRouterPort(Protocol):
    """Port contract for dual-tier cloud or deterministic demo LLM routers."""

    def select_route(
        self,
        *,
        task_type: Any = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: Any = None,
    ) -> RouteDecision:
        """Select a fast or strong route decision."""
        ...

    async def ainvoke(
        self,
        messages: Any,
        *,
        task_type: Any = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: Any = None,
        **kwargs: Any,
    ) -> LLMGenerationResult:
        """Invoke the routed LLM asynchronously and return a complete response."""
        ...

    def astream(
        self,
        messages: Any,
        *,
        task_type: Any = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response text chunks asynchronously."""
        ...

    async def ainvoke_structured(
        self,
        schema: type[Any],
        messages: Any,
        *,
        task_type: Any = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the routed LLM and validate output against ``schema``."""
        ...


@runtime_checkable
class SemanticCachePort(Protocol):
    """Port contract for Redis 8 vector semantic cache and in-memory fallbacks."""

    singleflight: SingleFlight

    async def get(
        self,
        query: str,
        *,
        namespace: str | None = None,
        threshold: float | None = None,
    ) -> CachedResponse | None:
        """Look up a semantically similar cached response."""
        ...

    async def lookup(
        self,
        query: str,
        *,
        namespace: str | None = None,
        threshold: float | None = None,
    ) -> CachedResponse | None:
        """Alias for ``get``."""
        ...

    async def set(
        self,
        query: str,
        response: str,
        *,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Persist a query vector and response with TTL expiration."""
        ...

    async def store(
        self,
        query: str,
        response: str,
        *,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Alias for ``set``."""
        ...

    async def invalidate_namespace(self, namespace: str | None = None) -> int:
        """Delete all entries in ``namespace``."""
        ...

    async def aclose(self) -> None:
        """Gracefully close cache connections or clear in-memory state."""
        ...


@runtime_checkable
class RagasEvaluatorPort(Protocol):
    """Port contract for multi-pipeline Ragas v0.2+ evaluation runners."""

    async def evaluate_sample(self, sample: Any, pipeline_version: Any) -> Any:
        """Evaluate a single sample on one pipeline version."""
        ...

    async def evaluate_pipeline(self, samples: Sequence[Any], pipeline_version: Any) -> Any:
        """Evaluate a dataset on one pipeline version."""
        ...

    async def compare_pipelines(
        self,
        samples: Sequence[Any],
        versions: Sequence[Any] = ("v1", "v2", "v3"),
    ) -> Any:
        """Compare multiple pipeline versions over the same dataset."""
        ...


__all__: Sequence[str] = [
    "LLMRouterPort",
    "RagasEvaluatorPort",
    "RetrieverPort",
    "SemanticCachePort",
]
