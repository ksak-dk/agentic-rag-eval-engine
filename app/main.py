"""
app/main.py
===========
FastAPI application factory and async lifespan resource management.

Hardware & Architecture Contract
--------------------------------
- Configures single-line JSON structured logging via ``configure_logging()``.
- In non-test, non-mock GPU runtime, loads ONLY FP16 ``BAAI/bge-m3`` and
  ``BAAI/bge-reranker-v2-m3`` (~2.3 GB VRAM total). Never loads any local LLM.
- Manages ``HybridVectorStore``, ``SemanticCache``, ``LLMRouter``, and
  ``AsyncSqliteSaver`` lifecycle inside ``lifespan``.
- Supports full dependency injection via ``create_app(...)``, ``app.state``,
  or FastAPI ``app.dependency_overrides`` for offline CPU/mock testing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from app.api.v1.chat import router as chat_v1_router
from app.cache.semantic_cache import SemanticCache
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, sample_cuda_vram_mb
from app.core.router import LLMRouter
from app.graph.nodes import GraphDependencies
from app.graph.workflow import AgenticRAGWorkflow, open_async_sqlite_saver
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


def _try_get_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


def _is_cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize and gracefully tear down async RAG infrastructure resources."""
    vram_before_mb = sample_cuda_vram_mb()

    settings: Settings | None = getattr(application.state, "settings", None)
    if settings is None:
        settings = _try_get_settings()
        application.state.settings = settings

    configure_logging(settings.log_level.value if settings is not None else None)

    testing = bool(getattr(application.state, "testing", False))
    mock_mode = bool(settings.mock_llm_stream) if settings is not None else True

    if getattr(application.state, "router", None) is None and settings is not None:
        application.state.router = LLMRouter(settings=settings)

    owned_qdrant_client: AsyncQdrantClient | None = None
    owned_cache: SemanticCache | None = None

    async with AsyncExitStack() as exit_stack:
        if (
            not testing
            and not mock_mode
            and settings is not None
            and getattr(application.state, "retriever", None) is None
        ):
            if _is_cuda_available():
                owned_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
                dense_embedder = BGEDenseEmbedder(model_name=settings.embed_model_name)
                sparse_embedder = BM25SparseEmbedder()
                reranker = BGEReranker(model_name=settings.reranker_model_name)
                retriever = HybridVectorStore(
                    client=owned_qdrant_client,
                    dense_embedder=dense_embedder,
                    sparse_embedder=sparse_embedder,
                    reranker=reranker,
                    collection_name=settings.qdrant_collection_name,
                )
                try:
                    await retriever.ensure_collection()
                except Exception as exc:
                    logger.warning(
                        "Qdrant collection initialization failed (deferred)",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                application.state.retriever = retriever

                if getattr(application.state, "cache", None) is None:
                    owned_cache = SemanticCache.from_url(
                        dense_embedder,
                        redis_url=settings.redis_url,
                        settings=settings,
                    )
                    try:
                        await owned_cache.ensure_index()
                    except Exception as exc:
                        logger.warning(
                            "Redis semantic cache initialization failed (fail-open)",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    application.state.cache = owned_cache
            else:
                logger.warning(
                    "CUDA is unavailable in non-mock startup; skipping local FP16 model load",
                )

        if getattr(application.state, "checkpointer", None) is None and not testing:
            db_path = getattr(application.state, "checkpoint_db", None) or (
                settings.langgraph_checkpoint_db if settings is not None else None
            )
            try:
                application.state.checkpointer = await exit_stack.enter_async_context(
                    open_async_sqlite_saver(db_path)
                )
            except Exception as exc:
                logger.warning(
                    "AsyncSqliteSaver startup initialization failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        retriever = getattr(application.state, "retriever", None)
        llm_router = getattr(application.state, "router", None)
        cache = getattr(application.state, "cache", None)
        if (
            getattr(application.state, "workflow", None) is None
            and retriever is not None
            and llm_router is not None
        ):
            top_k = settings.retrieval_top_k if settings is not None else 5
            top_n = settings.retrieval_top_n if settings is not None else 20
            max_retries = settings.max_retries if settings is not None else 2
            db_path = getattr(application.state, "checkpoint_db", None) or (
                settings.langgraph_checkpoint_db if settings is not None else None
            )
            graph_deps = GraphDependencies(
                retriever=retriever,
                router=llm_router,
                cache=cache,
                top_k=top_k,
                top_n=top_n,
                max_retries=max_retries,
            )
            application.state.workflow = AgenticRAGWorkflow(
                graph_deps,
                db_path=db_path,
                max_retries=max_retries,
            )

        vram_after_mb = sample_cuda_vram_mb()
        vram_delta_mb = round(max(0.0, vram_after_mb - vram_before_mb), 2)
        logger.info(
            "Application lifespan initialized",
            pipeline_version=settings.pipeline_version.value if settings is not None else "v3",
            testing=testing,
            mock_llm_stream=mock_mode,
            cuda_available=_is_cuda_available(),
            vram_before_mb=vram_before_mb,
            vram_after_mb=vram_after_mb,
            vram_delta_mb=vram_delta_mb,
            retriever_ready=retriever is not None,
            router_ready=llm_router is not None,
            cache_ready=cache is not None,
            checkpointer_ready=getattr(application.state, "checkpointer", None) is not None,
            workflow_ready=getattr(application.state, "workflow", None) is not None,
        )

        try:
            yield
        finally:
            if owned_cache is not None:
                try:
                    await owned_cache.aclose()
                except Exception as exc:
                    logger.warning(
                        "Error closing SemanticCache",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            if owned_qdrant_client is not None:
                try:
                    await owned_qdrant_client.close()
                except Exception as exc:
                    logger.warning(
                        "Error closing AsyncQdrantClient",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
            logger.info("Application lifespan shutdown complete")


def create_app(
    *,
    settings: Settings | None = None,
    retriever: Any = None,
    router: Any = None,
    cache: Any | None = None,
    checkpointer: Any | None = None,
    workflow: Any | None = None,
    checkpoint_db: str | Path | None = None,
    testing: bool = False,
) -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    application = FastAPI(
        title="agentic-rag-eval-engine",
        version="0.1.0",
        description="Dual-channel SSE Agentic RAG service supporting v1/v2/v3 pipelines.",
        lifespan=lifespan,
    )

    application.state.settings = settings
    application.state.retriever = retriever
    application.state.router = router
    application.state.cache = cache
    application.state.checkpointer = checkpointer
    application.state.workflow = workflow
    application.state.checkpoint_db = Path(checkpoint_db) if checkpoint_db is not None else None
    application.state.testing = testing

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(chat_v1_router)
    return application


app = create_app()


__all__: Sequence[str] = [
    "app",
    "create_app",
    "lifespan",
]
