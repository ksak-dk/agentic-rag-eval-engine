"""
app/retrieval/hybrid_store.py
==============================
Qdrant-backed hybrid vector store with two operating modes:

  v1 — Dense-only retrieval (Qdrant approximate nearest neighbour).
  v2/v3 — Prefetch Dense Top-N + Sparse (BM25) Top-N
           → server-side Fusion.RRF
           → local CUDA cross-encoder reranker (bge-reranker-v2-m3) Top-K.

Design philosophy
-----------------
- Three ``typing.Protocol`` interfaces (DenseEmbedder, SparseEmbedder, Reranker)
  decouple ``HybridVectorStore`` from concrete GPU/CPU models.  Tests inject
  lightweight Protocol-compatible stubs without loading any CUDA weights.
- All blocking CUDA / CPU inference runs in a thread executor so the asyncio
  event loop is never stalled.
- VRAM budget enforced by FP16 loading: bge-m3 ≈ 1.1 GB + reranker ≈ 1.1 GB
  = 2.3 GB total — safely within the RTX 3060 6 GB ceiling.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import structlog
import torch
from fastembed import SparseTextEmbedding
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    PointStruct,
    Prefetch,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.retrieval.chunker import ChunkMetadata, DocumentChunk

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSE_VECTOR_NAME: str = "dense"
SPARSE_VECTOR_NAME: str = "sparse"
DENSE_DIM: int = 1024  # BAAI/bge-m3 output dimensionality

# (indices, values) pair representing one sparse vector
SparseEmbedding = tuple[list[int], list[float]]


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A ranked retrieval result returned by HybridVectorStore.search()."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Chunk text content.")
    metadata: ChunkMetadata
    score: float = Field(
        description=(
            "Reranker cross-encoder score for v2/v3; Qdrant cosine similarity score for v1."
        )
    )
    rank: int = Field(ge=0, description="Zero-based rank after final scoring (0 = best).")


# ---------------------------------------------------------------------------
# Protocols  (dependency interfaces — inject stubs in tests)
# ---------------------------------------------------------------------------


@runtime_checkable
class DenseEmbedder(Protocol):
    """Interface for dense text embedding models (e.g. BAAI/bge-m3)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised dense vectors, shape (len(texts), dim)."""
        ...

    def get_vram_usage_mb(self) -> float:
        """Current GPU VRAM allocated by this model in megabytes."""
        ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Interface for sparse (lexical) text embedding models (e.g. BM25)."""

    def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        """Return (indices, values) sparse vector pairs for each input text."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Interface for cross-encoder reranking models."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Return a relevance score for each (query, text) pair (higher = more relevant)."""
        ...

    def get_vram_usage_mb(self) -> float:
        """Current GPU VRAM allocated by this model in megabytes."""
        ...


# ---------------------------------------------------------------------------
# Default CUDA / FastEmbed implementations
# ---------------------------------------------------------------------------


class BGEDenseEmbedder:
    """BAAI/bge-m3 dense embedder — FP16 on CUDA via sentence-transformers.

    VRAM footprint: ~1.1 GB.
    Loaded with ``model_kwargs={"torch_dtype": torch.float16}`` to prevent
    accidental FP32 materialisation that would exhaust the 6 GB VRAM budget.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        assert torch.cuda.is_available(), (
            "CUDA is required for BGEDenseEmbedder but is not available. "
            "Verify RTX 3060 drivers and CUDA 12.4 toolkit installation."
        )
        vram_before_mb = torch.cuda.memory_allocated() / 1024 / 1024

        self._model = SentenceTransformer(
            model_name,
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )

        vram_after_mb = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info(
            "BGEDenseEmbedder loaded",
            model=model_name,
            vram_delta_mb=round(vram_after_mb - vram_before_mb, 1),
            vram_total_mb=round(vram_after_mb, 1),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()  # type: ignore[union-attr]

    def get_vram_usage_mb(self) -> float:
        return torch.cuda.memory_allocated() / 1024 / 1024


class BM25SparseEmbedder:
    """Qdrant/bm25 sparse embedder via fastembed — CPU inference.

    Returns (indices, values) pairs compatible with Qdrant SparseVector payloads.
    No VRAM usage; runs entirely on the CPU thread pool.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model = SparseTextEmbedding(model_name=model_name)
        logger.info("BM25SparseEmbedder loaded", model=model_name)

    def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        results: list[SparseEmbedding] = []
        for emb in self._model.embed(texts):
            results.append((emb.indices.tolist(), emb.values.tolist()))
        return results


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder reranker — FP16 on CUDA.

    VRAM footprint: ~1.1 GB.
    Combined with BGEDenseEmbedder the total stays within the ~2.3 GB budget.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        assert torch.cuda.is_available(), (
            "CUDA is required for BGEReranker but is not available. "
            "Verify RTX 3060 drivers and CUDA 12.4 toolkit installation."
        )
        vram_before_mb = torch.cuda.memory_allocated() / 1024 / 1024

        self._model = CrossEncoder(
            model_name_or_path=model_name,
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )

        vram_after_mb = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info(
            "BGEReranker loaded",
            model=model_name,
            vram_delta_mb=round(vram_after_mb - vram_before_mb, 1),
            vram_total_mb=round(vram_after_mb, 1),
        )

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    def get_vram_usage_mb(self) -> float:
        return torch.cuda.memory_allocated() / 1024 / 1024


# ---------------------------------------------------------------------------
# HybridVectorStore
# ---------------------------------------------------------------------------


class HybridVectorStore:
    """Qdrant-backed hybrid retrieval store.

    All three component dependencies are injected via constructor — the class
    never instantiates models internally, making it trivially testable with
    Protocol-compatible stubs that never touch the GPU.

    Retrieval modes
    ---------------
    v1  Dense-only: single Qdrant ANN search, return Top-K by cosine score.
    v2/v3  Hybrid: Prefetch Dense Top-N + Sparse (BM25) Top-N
                   → server-side Fusion.RRF (Top-N candidates)
                   → local CUDA reranker (Top-K final results).

    Async safety
    ------------
    All blocking calls (CUDA inference, CPU BM25 tokenisation) run inside
    ``loop.run_in_executor(None, ...)`` so the event loop is never stalled.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        reranker: Reranker,
        collection_name: str = "rag_documents",
        upsert_batch_size: int = 32,
    ) -> None:
        self._client = client
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._reranker = reranker
        self._collection = collection_name
        self._batch_size = upsert_batch_size

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist.

        Collection schema
        -----------------
        - ``dense``  : 1024-dim Cosine HNSW, INT8 scalar quantisation,
                       ``always_ram=True`` (index hot in memory).
        - ``sparse`` : BM25 sparse vectors, in-memory index.

        Idempotent: calling this multiple times is safe.
        """
        if await self._client.collection_exists(self._collection):
            logger.info("Collection already exists, skipping creation", collection=self._collection)
            return

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10_000,
                    ),
                    quantization_config=ScalarQuantizationConfig(
                        scalar=ScalarQuantization(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        logger.info(
            "Qdrant collection created",
            collection=self._collection,
            dense_dim=DENSE_DIM,
            quantization="INT8",
            sparse_index="BM25/on-memory",
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and upsert DocumentChunks into Qdrant in batches.

        Both dense and sparse embeddings are computed in a thread executor
        to prevent event-loop blocking.  Points are upserted with
        ``wait=True`` to guarantee durability before returning.

        Args:
            chunks: Ordered list of DocumentChunk objects from the chunker.

        Returns:
            Total number of points successfully upserted.
        """
        if not chunks:
            return 0

        loop = asyncio.get_running_loop()
        total = 0

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            texts = [c.text for c in batch]

            # Embed dense and sparse concurrently (both blocking → executor)
            dense_fut = loop.run_in_executor(None, self._dense.embed, texts)
            sparse_fut = loop.run_in_executor(None, self._sparse.embed, texts)
            dense_vecs, sparse_vecs = await asyncio.gather(dense_fut, sparse_fut)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_VECTOR_NAME: dense_vecs[i],
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_vecs[i][0],
                            values=sparse_vecs[i][1],
                        ),
                    },
                    payload={
                        "text": chunk.text,
                        "doc_id": chunk.metadata.doc_id,
                        "source": chunk.metadata.source,
                        "section_path": chunk.metadata.section_path,
                        "chunk_index": chunk.metadata.chunk_index,
                    },
                )
                for i, chunk in enumerate(batch)
            ]

            await self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,
            )
            total += len(batch)
            logger.debug("Batch upserted", batch_size=len(batch), total=total)

        logger.info("Upsert complete", collection=self._collection, total_points=total)
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        """Retrieve and rerank documents for the given query.

        Args:
            query:            Natural-language query string.
            pipeline_version: ``"v1"`` → dense only;
                              ``"v2"`` / ``"v3"`` → hybrid RRF + rerank.
            top_k:            Final number of chunks returned (after reranking).
            top_n:            Candidate pool fetched from Qdrant before reranking.
                              Must be > top_k.

        Returns:
            Ordered list of RetrievedChunk (rank 0 = most relevant).
            Returns an empty list when the collection has no matching results.
        """
        if pipeline_version == "v1":
            return await self._search_dense_only(query, top_k)
        return await self._search_hybrid_rerank(query, top_k, top_n)

    # ------------------------------------------------------------------
    # Private: v1 — dense-only ANN
    # ------------------------------------------------------------------

    async def _search_dense_only(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        loop = asyncio.get_running_loop()
        dense_vecs: list[list[float]] = await loop.run_in_executor(None, self._dense.embed, [query])

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=(DENSE_VECTOR_NAME, dense_vecs[0]),
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                text=_safe_str(hit.payload, "text"),
                metadata=_payload_to_metadata(hit.payload),
                score=float(hit.score),
                rank=rank,
            )
            for rank, hit in enumerate(hits)
        ]

    # ------------------------------------------------------------------
    # Private: v2/v3 — hybrid Prefetch → RRF → reranker
    # ------------------------------------------------------------------

    async def _search_hybrid_rerank(
        self,
        query: str,
        top_k: int,
        top_n: int,
    ) -> list[RetrievedChunk]:
        loop = asyncio.get_running_loop()

        # 1. Embed query with both encoders concurrently
        dense_fut = loop.run_in_executor(None, self._dense.embed, [query])
        sparse_fut = loop.run_in_executor(None, self._sparse.embed, [query])
        dense_vecs, sparse_vecs = await asyncio.gather(dense_fut, sparse_fut)

        dense_vec = dense_vecs[0]
        sparse_indices, sparse_values = sparse_vecs[0]

        # 2. Qdrant server-side: Prefetch Dense + Sparse → Fusion RRF → Top-N
        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=top_n,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_n,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_n,
            with_payload=True,
        )

        candidates = response.points
        if not candidates:
            return []

        # 3. Local CUDA reranker: Top-N → Top-K
        candidate_texts = [_safe_str(p.payload, "text") for p in candidates]
        rerank_scores: list[float] = await loop.run_in_executor(
            None, self._reranker.rerank, query, candidate_texts
        )

        ranked = sorted(
            zip(rerank_scores, candidates, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]

        return [
            RetrievedChunk(
                text=_safe_str(point.payload, "text"),
                metadata=_payload_to_metadata(point.payload),
                score=score,
                rank=rank,
            )
            for rank, (score, point) in enumerate(ranked)
        ]


# ---------------------------------------------------------------------------
# Private payload helpers
# ---------------------------------------------------------------------------


def _safe_str(payload: dict | None, key: str, default: str = "") -> str:
    """Extract a string value from a Qdrant payload dict defensively."""
    if payload is None:
        return default
    value = payload.get(key, default)
    return str(value) if value is not None else default


def _payload_to_metadata(payload: dict | None) -> ChunkMetadata:
    """Reconstruct a ChunkMetadata from a Qdrant point payload dict."""
    p = payload or {}
    return ChunkMetadata(
        doc_id=_safe_str(p, "doc_id", "unknown"),
        source=_safe_str(p, "source", "unknown"),
        section_path=p.get("section_path") or [],
        chunk_index=int(p.get("chunk_index", 0)),
    )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__: Sequence[str] = [
    "DENSE_DIM",
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "BGEDenseEmbedder",
    "BGEReranker",
    "BM25SparseEmbedder",
    "DenseEmbedder",
    "HybridVectorStore",
    "Reranker",
    "RetrievedChunk",
    "SparseEmbedder",
    "SparseEmbedding",
]
