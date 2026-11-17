"""
app/core/container.py
=====================
Lifespan dependency injection container with automatic ``live`` / ``demo`` / ``test``
mode detection and graceful degradation.

Design Contract
---------------
- Strictly orchestrates mode resolution, component wiring, and lifecycle shutdown
  against ``app.ports`` protocols (``RetrieverPort``, ``LLMRouterPort``,
  ``SemanticCachePort``).
- In ``demo`` and ``test`` modes, NEVER loads CUDA weights or contacts external
  Qdrant / Redis / Cloud LLM endpoints; wires ``app.adapters.mock`` adapters +
  ``InMemorySemanticCache`` + ``AsyncSqliteSaver``.
- In ``live`` mode, wires real ``HybridVectorStore`` (FP16 ``bge-m3`` +
  ``bge-reranker-v2-m3``, ~2.3 GB VRAM total, zero local LLMs), ``LLMRouter``,
  Redis 8 ``SemanticCache``, and ``AsyncSqliteSaver``. Any external outage
  gracefully falls back to the corresponding ``demo``/in-memory adapter.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.adapters.mock import DemoLLMRouter, InMemoryHybridRetriever
from app.cache.semantic_cache import InMemorySemanticCache, SemanticCache, SingleFlight
from app.core.config import Settings, get_settings
from app.core.router import LLMRouter
from app.graph.nodes import GraphDependencies
from app.graph.workflow import AgenticRAGWorkflow, open_async_sqlite_saver
from app.ports import LLMRouterPort, RetrieverPort, SemanticCachePort
from app.retrieval.hybrid_store import (
    BGEDenseEmbedder,
    BGEReranker,
    BM25SparseEmbedder,
    HybridVectorStore,
)

try:
    import torch
except ImportError:  # pragma: no cover - defensive fallback for minimal envs
    torch = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

ResolvedRuntimeMode = Literal["live", "demo", "test"]

_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your_",
    "placeholder",
    "changeme",
    "replace_me",
    "example",
    "gsk_your_",
    "aizasy_your_",
)


# ---------------------------------------------------------------------------
# Environment & Service Probing Helpers
# ---------------------------------------------------------------------------


def is_cuda_available() -> bool:
    """Safely check whether CUDA GPU acceleration is available."""
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_placeholder_api_key(value: str | None) -> bool:
    """Return True when an API key is empty, whitespace, or an unedited template value."""
    if value is None:
        return True
    cleaned = value.strip()
    if len(cleaned) < 8:
        return True
    lowered = cleaned.lower()
    if lowered.startswith("<") or lowered.endswith(">"):
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def has_valid_cloud_api_keys(settings: Settings) -> bool:
    """Check whether at least one real cloud LLM API key is configured."""
    candidates = (
        settings.groq_api_key,
        settings.google_api_key,
        settings.gemini_api_key,
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("GEMINI_API_KEY", ""),
    )
    return any(not is_placeholder_api_key(key) for key in candidates)


async def probe_qdrant_ready(qdrant_url: str, *, timeout_seconds: float = 1.0) -> bool:
    """Perform a lightweight non-blocking HTTP health probe against Qdrant."""
    base = qdrant_url.strip().rstrip("/")
    if not base:
        return False
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for endpoint in ("/readyz", "/collections"):
                try:
                    resp = await client.get(f"{base}{endpoint}")
                    if resp.status_code < 500:
                        return True
                except httpx.HTTPError:
                    continue
    except Exception:
        return False
    return False


async def probe_redis_ready(redis_url: str, *, timeout_seconds: float = 1.0) -> bool:
    """Perform a lightweight non-blocking Redis PING check."""
    cleaned = redis_url.strip()
    if not cleaned:
        return False
    client: Redis | None = None
    try:
        client = Redis.from_url(
            cleaned,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            decode_responses=False,
        )
        pong = await client.ping()
        return bool(pong)
    except Exception:
        return False
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ServiceContainer (Lifespan Dependency Injection & Dual-Mode Auto-Wiring)
# ---------------------------------------------------------------------------


class ServiceContainer:
    """Manages startup wiring and graceful shutdown for RAG infrastructure components."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        mode_override: ResolvedRuntimeMode | None = None,
        checkpoint_db: str | Path | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self._mode_override = mode_override
        self._checkpoint_db_override = checkpoint_db

        self.mode: ResolvedRuntimeMode = "demo"
        self.retriever: RetrieverPort | None = None
        self.router: LLMRouterPort | None = None
        self.semantic_cache: SemanticCachePort | None = None
        self.workflow: AgenticRAGWorkflow | None = None
        self.checkpointer: Any = None
        self.component_status: dict[str, str] = {}
        self.singleflight = SingleFlight()

        self._exit_stack: AsyncExitStack | None = None
        self._owned_qdrant_client: AsyncQdrantClient | None = None
        self._initialized: bool = False

    async def _resolve_mode(self) -> tuple[ResolvedRuntimeMode, str]:
        if self._mode_override is not None:
            return self._mode_override, "explicit container mode_override"

        configured = self.settings.runtime_mode
        if configured in ("live", "demo", "test"):
            return configured, f"explicit settings.runtime_mode={configured}"

        if self.settings.mock_llm_stream:
            return "demo", "MOCK_LLM_STREAM=true forces offline demo mode"

        if not has_valid_cloud_api_keys(self.settings):
            return "demo", "Cloud LLM API keys missing or template placeholders"

        if not await probe_qdrant_ready(self.settings.qdrant_url):
            return "demo", f"Qdrant unreachable at {self.settings.qdrant_url}"

        return "live", "Cloud API keys and Qdrant service are available"

    def _resolve_checkpoint_target(self, mode: ResolvedRuntimeMode) -> str | Path:
        if self._checkpoint_db_override is not None:
            return self._checkpoint_db_override
        if mode == "test":
            return ":memory:"
        return self.settings.langgraph_checkpoint_db

    async def _open_checkpointer(self, target: str | Path) -> Any:
        assert self._exit_stack is not None
        try:
            return await self._exit_stack.enter_async_context(open_async_sqlite_saver(target))
        except Exception as exc:
            logger.warning(
                "AsyncSqliteSaver failed to open target path, falling back to :memory:",
                checkpoint_db=str(target),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return await self._exit_stack.enter_async_context(open_async_sqlite_saver(":memory:"))

    def _compile_workflow(self, checkpoint_target: str | Path) -> AgenticRAGWorkflow:
        graph_deps = GraphDependencies(
            retriever=self.retriever,
            router=self.router,
            cache=self.semantic_cache,
            top_k=self.settings.retrieval_top_k,
            top_n=self.settings.retrieval_top_n,
            max_retries=self.settings.max_retries,
        )
        return AgenticRAGWorkflow(
            graph_deps,
            db_path=checkpoint_target,
            max_retries=self.settings.max_retries,
            checkpointer=self.checkpointer,
        )

    async def _init_demo_or_test(self, mode: ResolvedRuntimeMode) -> None:
        self.retriever = await InMemoryHybridRetriever.from_corpus_paths(
            self.settings.demo_corpus_paths
        )
        self.router = DemoLLMRouter(self.settings)
        self.semantic_cache = InMemorySemanticCache(
            settings=self.settings,
            similarity_threshold=self.settings.cache_similarity_threshold,
            ttl_seconds=self.settings.cache_ttl_seconds,
        )

        checkpoint_target = self._resolve_checkpoint_target(mode)
        self.checkpointer = await self._open_checkpointer(checkpoint_target)
        self.workflow = self._compile_workflow(checkpoint_target)
        self.component_status = {
            "runtime_mode": mode,
            "retriever": f"ready ({mode})",
            "router": f"ready ({mode})",
            "semantic_cache": f"ready ({mode})",
            "workflow": f"ready ({mode})",
            "checkpointer": f"ready ({mode})",
        }

    async def _init_live(self) -> None:
        dense_embedder: BGEDenseEmbedder | None = None
        retriever_status = "ready (live)"
        router_status = "ready (live)"
        cache_status = "ready (live)"

        if is_cuda_available() and await probe_qdrant_ready(self.settings.qdrant_url):
            qdrant_client: AsyncQdrantClient | None = None
            try:
                qdrant_client = AsyncQdrantClient(url=self.settings.qdrant_url)
                dense_embedder = BGEDenseEmbedder(model_name=self.settings.embed_model_name)
                sparse_embedder = BM25SparseEmbedder()
                reranker = BGEReranker(model_name=self.settings.reranker_model_name)
                store = HybridVectorStore(
                    client=qdrant_client,
                    dense_embedder=dense_embedder,
                    sparse_embedder=sparse_embedder,
                    reranker=reranker,
                    collection_name=self.settings.qdrant_collection_name,
                )
                await store.ensure_collection()
                self._owned_qdrant_client = qdrant_client
                self.retriever = store
            except Exception as exc:
                logger.warning(
                    "Live HybridVectorStore initialization failed; falling back to in-memory retriever",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                if qdrant_client is not None:
                    try:
                        await qdrant_client.close()
                    except Exception:
                        pass
                dense_embedder = None
                self.retriever = await InMemoryHybridRetriever.from_corpus_paths(
                    self.settings.demo_corpus_paths
                )
                retriever_status = "ready (demo-fallback)"
        else:
            logger.warning(
                "Live retrieval prerequisites unavailable (CUDA or Qdrant); using demo retriever fallback",
                cuda_available=is_cuda_available(),
                qdrant_url=self.settings.qdrant_url,
            )
            self.retriever = await InMemoryHybridRetriever.from_corpus_paths(
                self.settings.demo_corpus_paths
            )
            retriever_status = "ready (demo-fallback)"

        if has_valid_cloud_api_keys(self.settings) and not self.settings.mock_llm_stream:
            try:
                self.router = LLMRouter(settings=self.settings)
            except Exception as exc:
                logger.warning(
                    "Live LLMRouter initialization failed; falling back to DemoLLMRouter",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                self.router = DemoLLMRouter(self.settings)
                router_status = "ready (demo-fallback)"
        else:
            self.router = DemoLLMRouter(self.settings)
            router_status = "ready (demo-fallback)"

        if dense_embedder is not None and await probe_redis_ready(self.settings.redis_url):
            live_cache = SemanticCache.from_url(
                dense_embedder,
                redis_url=self.settings.redis_url,
                settings=self.settings,
            )
            try:
                index_ok = await live_cache.ensure_index()
                if index_ok:
                    self.semantic_cache = live_cache
                else:
                    await live_cache.aclose()
                    raise RuntimeError("RediSearch index initialization returned False")
            except Exception as exc:
                logger.warning(
                    "Live Redis SemanticCache unavailable; falling back to InMemorySemanticCache",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                try:
                    await live_cache.aclose()
                except Exception:
                    pass
                self.semantic_cache = InMemorySemanticCache(
                    settings=self.settings,
                    similarity_threshold=self.settings.cache_similarity_threshold,
                    ttl_seconds=self.settings.cache_ttl_seconds,
                )
                cache_status = "ready (in-memory-fallback)"
        else:
            self.semantic_cache = InMemorySemanticCache(
                settings=self.settings,
                similarity_threshold=self.settings.cache_similarity_threshold,
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
            cache_status = "ready (in-memory-fallback)"

        checkpoint_target = self._resolve_checkpoint_target("live")
        self.checkpointer = await self._open_checkpointer(checkpoint_target)
        self.workflow = self._compile_workflow(checkpoint_target)
        self.component_status = {
            "runtime_mode": "live",
            "retriever": retriever_status,
            "router": router_status,
            "semantic_cache": cache_status,
            "workflow": "ready (live)",
            "checkpointer": "ready (live)",
        }

    async def initialize(self) -> None:
        """Resolve runtime mode and wire all RAG components safely."""
        if self._initialized:
            return

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        try:
            resolved_mode, reason = await self._resolve_mode()
            self.mode = resolved_mode
            if resolved_mode == "live":
                logger.info(
                    "ServiceContainer activating live mode",
                    runtime_mode=resolved_mode,
                    reason=reason,
                )
                await self._init_live()
            else:
                logger.warning(
                    "ServiceContainer activating offline mode",
                    runtime_mode=resolved_mode,
                    reason=reason,
                )
                await self._init_demo_or_test(resolved_mode)
            self._initialized = True
        except Exception:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Close AsyncSqliteSaver, SemanticCache, and Qdrant client connections."""
        if self.semantic_cache is not None:
            try:
                await self.semantic_cache.aclose()
            except Exception as exc:
                logger.warning(
                    "Error closing semantic_cache",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            self.semantic_cache = None

        if self._owned_qdrant_client is not None:
            try:
                await self._owned_qdrant_client.close()
            except Exception as exc:
                logger.warning(
                    "Error closing AsyncQdrantClient",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            self._owned_qdrant_client = None

        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.warning(
                    "Error closing ServiceContainer AsyncExitStack",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            self._exit_stack = None

        self.checkpointer = None
        self.workflow = None
        self._initialized = False


__all__: Sequence[str] = [
    "DemoLLMRouter",
    "InMemoryHybridRetriever",
    "ResolvedRuntimeMode",
    "ServiceContainer",
    "has_valid_cloud_api_keys",
    "is_cuda_available",
    "is_placeholder_api_key",
    "probe_qdrant_ready",
    "probe_redis_ready",
]
