"""
app/api/v1/ingest.py
====================
FastAPI v1 endpoints for incremental document ingestion and knowledge base stats.

Endpoints
---------
- ``POST /api/v1/documents/ingest`` — Incrementally chunk, deduplicate (SHA-256),
  index Markdown/TXT documents, and invalidate stale semantic cache namespaces.
- ``GET  /api/v1/documents/stats``  — Report total indexed chunks, ingested
  document count, and active runtime mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.ingest import DocumentIngestionService, IngestRequest, IngestResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentStatsResponse(BaseModel):
    """Knowledge base index statistics reported by ``GET /api/v1/documents/stats``."""

    model_config = ConfigDict(frozen=True)

    total_chunks_in_store: int = Field(default=0, ge=0)
    ingested_documents_count: int = Field(default=0, ge=0)
    runtime_mode: str = Field(default="demo")
    pipeline_version: str = Field(default="v3")


def get_ingestion_service(request: Request) -> DocumentIngestionService:
    """Resolve ``DocumentIngestionService`` from ``app.state`` or lazy-wire from retriever."""
    service = getattr(request.app.state, "ingestion_service", None)
    if isinstance(service, DocumentIngestionService):
        return service

    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document ingestion requires an initialized retriever service.",
        )

    cache = getattr(request.app.state, "cache", None) or getattr(
        request.app.state, "semantic_cache", None
    )
    service = DocumentIngestionService(retriever=retriever, semantic_cache=cache)
    request.app.state.ingestion_service = service
    return service


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_documents(
    payload: IngestRequest,
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Incrementally ingest inline Markdown/TXT documents or local file paths."""
    if not payload.documents and not payload.file_paths:
        raise HTTPException(
            status_code=422,
            detail="Either 'documents' or 'file_paths' must contain at least one entry.",
        )

    try:
        return await service.ingest(payload)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(
            "Document ingestion failed",
            pipeline_version=payload.pipeline_version,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@router.get("/stats", response_model=DocumentStatsResponse, status_code=status.HTTP_200_OK)
async def get_document_stats(request: Request) -> DocumentStatsResponse:
    """Return total chunk count, ingested document count, and active runtime mode."""
    service = getattr(request.app.state, "ingestion_service", None)
    retriever = getattr(request.app.state, "retriever", None)
    container = getattr(request.app.state, "container", None)
    settings = getattr(request.app.state, "settings", None)

    if isinstance(service, DocumentIngestionService):
        total_chunks = service.total_chunks_in_store()
        ingested_docs = service.ingested_documents_count
    elif retriever is not None and hasattr(retriever, "count_chunks"):
        total_chunks = int(retriever.count_chunks())
        ingested_docs = 0
    else:
        total_chunks = 0
        ingested_docs = 0

    runtime_mode = str(getattr(container, "mode", "custom" if retriever is not None else "unready"))
    pipeline_version = settings.pipeline_version.value if settings is not None else "v3"
    return DocumentStatsResponse(
        total_chunks_in_store=total_chunks,
        ingested_documents_count=ingested_docs,
        runtime_mode=runtime_mode,
        pipeline_version=pipeline_version,
    )


__all__: Sequence[str] = [
    "DocumentStatsResponse",
    "get_document_stats",
    "get_ingestion_service",
    "ingest_documents",
    "router",
]
