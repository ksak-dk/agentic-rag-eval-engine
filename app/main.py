"""
app/main.py
===========
FastAPI application factory and async lifespan resource management.

Hardware & Architecture Contract
--------------------------------
- Configures single-line JSON structured logging via ``configure_logging()``.
- Respects explicit dependency injections on ``app.state`` (tests/overrides).
- Otherwise delegates auto-wiring to ``ServiceContainer`` for seamless
  ``live`` (Qdrant + Redis + CUDA FP16 + Cloud LLMs) or zero-dependency
  ``demo`` / ``test`` startup.
- Serves ``tests/sse_viewer.html`` at ``GET /`` and ``GET /viewer`` for
  one-click browser debugging while preserving local CORS access.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from app.api.v1.chat import router as chat_v1_router
from app.api.v1.ingest import router as ingest_v1_router
from app.core.config import Settings, get_settings
from app.core.container import ServiceContainer, is_cuda_available
from app.core.logging import configure_logging, sample_cuda_vram_mb
from app.core.router import LLMRouter
from app.graph.nodes import GraphDependencies
from app.graph.workflow import AgenticRAGWorkflow, open_async_sqlite_saver
from app.retrieval.ingest import DocumentIngestionService

logger = structlog.get_logger(__name__)

_SSE_VIEWER_PATH: Path = Path(__file__).resolve().parent.parent / "tests" / "sse_viewer.html"


def _try_get_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


def _has_preinjected_pipeline(application: FastAPI) -> bool:
    return any(
        getattr(application.state, attr, None) is not None
        for attr in ("workflow", "retriever", "router")
    )


async def _bootstrap_legacy_custom_injection(
    application: FastAPI,
    settings: Settings | None,
    exit_stack: AsyncExitStack,
    testing: bool,
) -> None:
    """Complete partial custom injections without overwriting caller-supplied services."""
    if getattr(application.state, "router", None) is None and settings is not None:
        application.state.router = LLMRouter(settings=settings)

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
    cache = getattr(application.state, "cache", None) or getattr(
        application.state, "semantic_cache", None
    )
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
    if getattr(application.state, "ingestion_service", None) is None and retriever is not None:
        application.state.ingestion_service = DocumentIngestionService(
            retriever=retriever,
            semantic_cache=cache,
        )


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
    auto_bootstrap = bool(settings.auto_bootstrap_container) if settings is not None else True
    owned_container: ServiceContainer | None = None

    async with AsyncExitStack() as exit_stack:
        if _has_preinjected_pipeline(application):
            await _bootstrap_legacy_custom_injection(application, settings, exit_stack, testing)
        elif auto_bootstrap:
            effective_settings = settings or Settings()
            if testing and effective_settings.runtime_mode == "auto":
                effective_settings = effective_settings.model_copy(update={"runtime_mode": "test"})
            checkpoint_override = getattr(application.state, "checkpoint_db", None)
            owned_container = ServiceContainer(
                effective_settings,
                checkpoint_db=checkpoint_override,
            )
            await owned_container.initialize()
            application.state.container = owned_container
            application.state.retriever = owned_container.retriever
            application.state.router = owned_container.router
            application.state.cache = owned_container.semantic_cache
            application.state.semantic_cache = owned_container.semantic_cache
            application.state.checkpointer = owned_container.checkpointer
            application.state.workflow = owned_container.workflow
            application.state.ingestion_service = owned_container.ingestion_service

        vram_after_mb = sample_cuda_vram_mb()
        vram_delta_mb = round(max(0.0, vram_after_mb - vram_before_mb), 2)
        logger.info(
            "Application lifespan initialized",
            runtime_mode=owned_container.mode if owned_container is not None else "custom",
            pipeline_version=settings.pipeline_version.value if settings is not None else "v3",
            testing=testing,
            cuda_available=is_cuda_available(),
            vram_before_mb=vram_before_mb,
            vram_after_mb=vram_after_mb,
            vram_delta_mb=vram_delta_mb,
            retriever_ready=getattr(application.state, "retriever", None) is not None,
            router_ready=getattr(application.state, "router", None) is not None,
            cache_ready=(
                getattr(application.state, "cache", None) is not None
                or getattr(application.state, "semantic_cache", None) is not None
            ),
            checkpointer_ready=getattr(application.state, "checkpointer", None) is not None,
            workflow_ready=getattr(application.state, "workflow", None) is not None,
        )

        try:
            yield
        finally:
            if owned_container is not None:
                await owned_container.shutdown()
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
        version="0.2.0",
        description="Dual-channel SSE Agentic RAG service supporting v1/v2/v3 pipelines.",
        lifespan=lifespan,
    )

    application.state.settings = settings
    application.state.retriever = retriever
    application.state.router = router
    application.state.cache = cache
    application.state.semantic_cache = cache
    application.state.checkpointer = checkpointer
    application.state.workflow = workflow
    application.state.checkpoint_db = Path(checkpoint_db) if checkpoint_db is not None else None
    application.state.testing = testing
    application.state.container = None
    application.state.ingestion_service = None

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/", include_in_schema=False, response_model=None)
    @application.get("/viewer", include_in_schema=False, response_model=None)
    async def sse_viewer() -> FileResponse | HTMLResponse:
        if await asyncio.to_thread(_SSE_VIEWER_PATH.is_file):
            return FileResponse(_SSE_VIEWER_PATH, media_type="text/html")
        return HTMLResponse(
            content=(
                "<html><body><h1>agentic-rag-eval-engine</h1>"
                "<p>SSE viewer file not found. Open <a href='/docs'>/docs</a>.</p>"
                "</body></html>"
            )
        )

    application.include_router(chat_v1_router)
    application.include_router(ingest_v1_router)
    return application


app = create_app()


__all__: Sequence[str] = [
    "app",
    "create_app",
    "lifespan",
]
