"""
app/retrieval/ingest.py
=======================
Idempotent incremental document ingestion pipeline and CLI for Markdown / TXT
knowledge base updates.

Design Contract
---------------
- Computes normalized SHA-256 content fingerprints keyed by
  ``(pipeline_version, source, doc_sha256)`` to guarantee idempotent retries
  without index bloat.
- Uses ``make_chunker(pipeline_version)`` (v1 fixed-window vs v2/v3 Markdown
  header-aware semantic chunking) and assigns ``doc_id#chunk_index`` provenance
  identifiers compatible with inline citation extraction.
- Delegates storage to ``RetrieverPort.add_chunks(...)`` (supporting both
  ``InMemoryHybridRetriever`` and Qdrant-backed ``HybridVectorStore``) and
  invalidates stale ``SemanticCachePort`` entries when new chunks land.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.adapters.mock.retriever import InMemoryHybridRetriever
from app.ports import RetrieverPort, SemanticCachePort
from app.retrieval.chunker import ChunkMetadata, DocumentChunk, make_chunker

logger = structlog.get_logger(__name__)

PipelineLiteral = Literal["v1", "v2", "v3"]
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".markdown"})


# ---------------------------------------------------------------------------
# Pydantic V2 Schemas
# ---------------------------------------------------------------------------


class IngestDocumentItem(BaseModel):
    """Single Markdown or plain-text document submitted for ingestion."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., min_length=1, description="Logical document name or relative path.")
    content: str = Field(..., min_length=1, description="Raw Markdown or text body.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional caller metadata (e.g. category, author, doc_sha256).",
    )

    @field_validator("source", "content")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty or whitespace-only.")
        return cleaned


class IngestRequest(BaseModel):
    """Batch ingestion request supporting inline documents and/or local file paths."""

    model_config = ConfigDict(frozen=True)

    documents: list[IngestDocumentItem] = Field(
        default_factory=list,
        description="Inline Markdown/text documents to chunk and index.",
    )
    file_paths: list[str] = Field(
        default_factory=list,
        description="Local Markdown/text file or directory paths to ingest.",
    )
    pipeline_version: PipelineLiteral = Field(
        default="v3",
        description="Chunking strategy selector: v1 (fixed) or v2/v3 (Markdown semantic).",
    )
    invalidate_cache: bool = Field(
        default=True,
        description="When True and chunks are added, invalidate stale semantic cache entries.",
    )


class IngestDocumentSummary(BaseModel):
    """Per-document ingestion outcome with SHA-256 deduplication status."""

    model_config = ConfigDict(frozen=True)

    source: str
    doc_sha256: str
    chunks_created: int = Field(..., ge=0)
    skipped_duplicate: bool = False


class IngestResponse(BaseModel):
    """Aggregated ingestion telemetry returned by the API and CLI."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    pipeline_version: str
    documents_processed: int = Field(..., ge=0)
    documents_skipped: int = Field(..., ge=0)
    total_chunks_added: int = Field(..., ge=0)
    total_chunks_in_store: int = Field(..., ge=0)
    cache_entries_invalidated: int = Field(default=0, ge=0)
    items: list[IngestDocumentSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify_source(source: str) -> str:
    stem = Path(source).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "doc"


def compute_content_sha256(content: str) -> str:
    """Compute normalized SHA-256 hex digest for idempotent deduplication."""
    normalized = content.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_documents_from_paths_sync(file_paths: Sequence[str]) -> list[IngestDocumentItem]:
    items: list[IngestDocumentItem] = []
    resolved_files: list[Path] = []

    for raw_path in file_paths:
        candidate = Path(raw_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Ingestion path not found: {raw_path}")
        if candidate.is_dir():
            dir_files = sorted(
                p
                for p in candidate.rglob("*")
                if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
            )
            if not dir_files:
                raise ValueError(f"No .md/.txt files found in directory: {raw_path}")
            resolved_files.extend(dir_files)
        elif candidate.is_file():
            resolved_files.append(candidate)

    for file_path in resolved_files:
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        items.append(
            IngestDocumentItem(
                source=file_path.as_posix(),
                content=text,
                metadata={"loaded_from_disk": True},
            )
        )
    return items


# ---------------------------------------------------------------------------
# DocumentIngestionService
# ---------------------------------------------------------------------------


class DocumentIngestionService:
    """Idempotent incremental ingestion orchestrator bound to Retriever & Cache ports."""

    def __init__(
        self,
        *,
        retriever: RetrieverPort,
        semantic_cache: SemanticCachePort | None = None,
    ) -> None:
        self._retriever = retriever
        self._semantic_cache = semantic_cache
        self._ingested_fingerprints: set[tuple[str, str, str]] = set()
        self._lock = asyncio.Lock()

    @property
    def ingested_documents_count(self) -> int:
        """Number of unique ``(pipeline_version, source, sha256)`` documents ingested."""
        return len(self._ingested_fingerprints)

    def total_chunks_in_store(self) -> int:
        """Query the underlying retriever for its current indexed chunk count."""
        return int(self._retriever.count_chunks())

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Chunk, deduplicate, index documents, and invalidate stale semantic cache entries."""
        all_items: list[IngestDocumentItem] = list(request.documents)
        if request.file_paths:
            disk_items = await asyncio.to_thread(
                _load_documents_from_paths_sync,
                request.file_paths,
            )
            all_items.extend(disk_items)

        if not all_items:
            raise ValueError(
                "No non-empty documents or readable file_paths provided for ingestion."
            )

        chunker = make_chunker(request.pipeline_version)
        summaries: list[IngestDocumentSummary] = []
        documents_processed = 0
        documents_skipped = 0
        total_chunks_added = 0

        async with self._lock:
            for item in all_items:
                doc_sha256 = compute_content_sha256(item.content)
                fingerprint = (request.pipeline_version, item.source, doc_sha256)

                if fingerprint in self._ingested_fingerprints:
                    documents_skipped += 1
                    summaries.append(
                        IngestDocumentSummary(
                            source=item.source,
                            doc_sha256=doc_sha256,
                            chunks_created=0,
                            skipped_duplicate=True,
                        )
                    )
                    continue

                slug = _slugify_source(item.source)
                base_doc_id = f"doc-{slug}-{doc_sha256[:8]}"
                raw_chunks = chunker.chunk(
                    text=item.content,
                    source=item.source,
                    doc_id=base_doc_id,
                )
                prepared_chunks = [
                    DocumentChunk.build(
                        text=chunk.text,
                        metadata=ChunkMetadata(
                            doc_id=f"{base_doc_id}#{chunk.metadata.chunk_index}",
                            source=item.source,
                            section_path=list(chunk.metadata.section_path),
                            chunk_index=chunk.metadata.chunk_index,
                        ),
                    )
                    for chunk in raw_chunks
                ]

                added = await self._retriever.add_chunks(prepared_chunks)
                self._ingested_fingerprints.add(fingerprint)
                documents_processed += 1
                total_chunks_added += added
                summaries.append(
                    IngestDocumentSummary(
                        source=item.source,
                        doc_sha256=doc_sha256,
                        chunks_created=added,
                        skipped_duplicate=False,
                    )
                )

            cache_entries_invalidated = 0
            if (
                request.invalidate_cache
                and total_chunks_added > 0
                and self._semantic_cache is not None
            ):
                namespaces = [request.pipeline_version]
                if "v3" not in namespaces:
                    namespaces.append("v3")
                for ns in namespaces:
                    try:
                        cache_entries_invalidated += int(
                            await self._semantic_cache.invalidate_namespace(ns)
                        )
                    except Exception as exc:
                        logger.warning(
                            "Semantic cache invalidation failed during ingestion (fail-open)",
                            namespace=ns,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

        total_in_store = self._retriever.count_chunks()
        logger.info(
            "Document ingestion completed",
            pipeline_version=request.pipeline_version,
            documents_processed=documents_processed,
            documents_skipped=documents_skipped,
            total_chunks_added=total_chunks_added,
            total_chunks_in_store=total_in_store,
            cache_entries_invalidated=cache_entries_invalidated,
        )
        return IngestResponse(
            status="ok",
            pipeline_version=request.pipeline_version,
            documents_processed=documents_processed,
            documents_skipped=documents_skipped,
            total_chunks_added=total_chunks_added,
            total_chunks_in_store=total_in_store,
            cache_entries_invalidated=cache_entries_invalidated,
            items=summaries,
        )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally chunk and ingest Markdown/TXT files into the RAG retriever.",
    )
    parser.add_argument(
        "--path",
        nargs="+",
        required=True,
        help="One or more Markdown/TXT file or directory paths to ingest.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("v1", "v2", "v3"),
        default="v3",
        help="Chunking pipeline strategy (default: v3).",
    )
    parser.add_argument(
        "--no-invalidate-cache",
        action="store_true",
        help="Disable automatic semantic cache invalidation.",
    )
    return parser


async def _async_cli(args: argparse.Namespace) -> int:
    retriever = InMemoryHybridRetriever([])
    service = DocumentIngestionService(retriever=retriever)
    request = IngestRequest(
        file_paths=list(args.path),
        pipeline_version=args.pipeline,
        invalidate_cache=not bool(args.no_invalidate_cache),
    )
    response = await service.ingest(request)
    print(response.model_dump_json(indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m app.retrieval.ingest``."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return asyncio.run(_async_cli(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[INGEST ERROR] {exc}", file=sys.stderr)
        return 1


__all__: Sequence[str] = [
    "DocumentIngestionService",
    "IngestDocumentItem",
    "IngestDocumentSummary",
    "IngestRequest",
    "IngestResponse",
    "PipelineLiteral",
    "compute_content_sha256",
    "main",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
