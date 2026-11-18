"""
app/adapters/mock/retriever.py
==============================
Zero-dependency in-memory hybrid retriever adapter implementing ``RetrieverPort``.

Simulates ``v1`` dense-only cosine retrieval and ``v2``/``v3`` dense + BM25 RRF
fusion + cross-encoder reranking without loading CUDA weights or contacting Qdrant.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.chunker import (
    ChunkerConfig,
    ChunkMetadata,
    ChunkMode,
    DocumentChunk,
    MarkdownHeaderChunker,
)
from app.retrieval.hybrid_store import RetrievedChunk

_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-zA-Z0-9_.-]+|[\u4e00-\u9fff]|[\uac00-\ud7af]+")

_BILINGUAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "显存": ("vram", "3060", "fp16", "2.3", "6gb", "bge-m3"),
    "缓存": ("cache", "redis", "cosine", "0.92", "semantic"),
    "重排": ("rerank", "reranker", "bge-reranker-v2-m3", "rrf"),
    "评测": ("ragas", "eval", "faithfulness", "recall", "precision", "benchmark"),
    "架构": ("architecture", "langgraph", "qdrant", "redis", "fastapi"),
    "循环": ("loop", "max_retries", "fallback", "retry"),
    "流式": ("sse", "stream", "token", "citations", "metrics"),
    "向量": ("vector", "dense", "sparse", "bm25", "int8"),
    "量化": ("quantization", "int8", "ragas", "benchmark"),
}

_KO_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "하이브리드": ("hybrid", "rrf", "bm25", "qdrant", "bge-reranker-v2-m3", "混合检索"),
    "검색": ("retrieval", "search", "qdrant", "bm25", "rrf", "检索"),
    "양자화": ("quantization", "int8", "qdrant", "量化"),
    "희소": ("sparse", "bm25", "rrf", "稀疏"),
    "융합": ("fusion", "rrf", "融合"),
    "자가": ("self-correcting", "langgraph", "stategraph", "rewrite", "自纠错"),
    "수정": ("self-correcting", "rewrite", "max_retries", "自纠错"),
    "재작성": ("rewrite", "query_rewrite", "max_retries", "重写"),
    "가드레일": ("guardrail", "is_cacheable_response", "fallback", "缓存"),
    "폴백": ("fallback", "is_cacheable_response", "cache", "降级"),
    "오염": ("pollution", "is_cacheable_response", "cache", "缓存"),
    "시맨틱": ("semantic", "cache", "redis", "cosine", "0.92", "语义缓存"),
    "캐시": ("cache", "redis", "cosine", "0.92", "singleflight", "缓存"),
    "동시성": ("concurrency", "singleflight", "load", "sla", "并发"),
    "무효화": ("invalidate", "invalidation", "ingest", "cache", "失效"),
    "수집": ("ingest", "ingestion", "sha256", "投喂"),
    "아키텍처": ("architecture", "v1", "v2", "v3", "langgraph", "架构"),
    "진화": ("evolution", "v1", "v2", "v3", "ragas", "演进"),
    "벤치마크": ("benchmark", "ragas", "p95", "1800", "sla", "评测"),
    "지표": ("metrics", "ragas", "faithfulness", "recall", "precision", "评测"),
    "지연": ("latency", "p95", "1800", "ttft", "sla", "延迟"),
}

DEFAULT_FALLBACK_CORPUS: str = """# agentic-rag-eval-engine Architecture Overview

## Hardware & 6GB VRAM Budget (RTX 3060)
The host runs Windows 11 + NVIDIA RTX 3060 (6 GB VRAM). Only two local models are allowed on CUDA in FP16:
- `BAAI/bge-m3` dense embedder (~1.1 GB VRAM, 1024-dim)
- `BAAI/bge-reranker-v2-m3` cross-encoder reranker (~1.1 GB VRAM)
Total VRAM budget is strictly ~2.3 GB (well under 6 GB). Zero local LLMs are ever loaded.

## v1 vs v2 vs v3 Pipeline Evolution
- `v1` Baseline: Fixed-length chunking + Qdrant INT8 dense-only retrieval + single-step fast generation (`llama-3.1-8b-instant`).
- `v2` Hybrid + CUDA Rerank: Markdown header semantic chunking (`H1/H2/H3`) + Qdrant Dense (1024-dim INT8) and BM25 Sparse prefetch + server-side `Fusion.RRF` + local FP16 `BAAI/bge-reranker-v2-m3` top-k reranking + strong generation (`gemini-2.5-flash`).
- `v3` Agentic Self-Correcting Loop: Adds Redis 8 native vector semantic cache (cosine >= 0.92), LangGraph `StateGraph` with document relevance grading, bounded query rewriting (`max_retries=2`), hallucination verification, `[doc_id]` citation tracing, and `AsyncSqliteSaver` persistence.

## Ragas v0.2+ & Async Load Benchmark
- Ragas Overall Score improves from `0.6759` (`v1`, Context Recall `0.3884`, Faithfulness `0.6392`) to `0.8554` (`v2`, Context Recall `0.7679`, Faithfulness `0.7984`) and `0.9950` (`v3`, Context Recall `1.0000`, Faithfulness `0.9851`).
- Under concurrency=10 load testing, `v3` achieves 30.0% semantic cache hit rate, P50 latency 445.69 ms, and P95 latency 678.77 ms (< 1800 ms SLA).

## Dual-Channel SSE & Cloud LLM Router
FastAPI `POST /api/v1/chat` streams dual-channel SSE events (`token`, `citations`, `metrics`, `done`). Cloud routing uses Groq `llama-3.1-8b-instant` (fast route with `gemini-2.5-flash-lite` failover) and Google AI Studio `gemini-2.5-flash` (strong route).
"""


def _tokenize_text(text: str) -> list[str]:
    """Tokenize multilingual text into normalized terms with bilingual/trilingual expansion."""
    raw_tokens = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    cjk_chars: list[str] = []
    for tok in raw_tokens:
        if "\u4e00" <= tok <= "\u9fff":
            cjk_chars.append(tok)
            tokens.append(tok)
        elif "\uac00" <= tok[0] <= "\ud7af":
            tokens.append(tok)
            for ko_key, ko_exp in _KO_EXPANSIONS.items():
                if ko_key in tok:
                    tokens.extend(ko_exp)
        else:
            tokens.append(tok)
            if len(tok) >= 4:
                tokens.append(tok[:4])
    for idx in range(len(cjk_chars) - 1):
        bigram = f"{cjk_chars[idx]}{cjk_chars[idx + 1]}"
        tokens.append(bigram)
        expansions = _BILINGUAL_EXPANSIONS.get(bigram)
        if expansions:
            tokens.extend(expansions)
    return tokens


def _slugify_source(source: str) -> str:
    stem = Path(source).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "corpus"


class _IndexedEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: RetrievedChunk
    tf: dict[str, float] = Field(default_factory=dict)
    norm: float = 1.0
    header_text: str = ""


class InMemoryHybridRetriever:
    """Zero-dependency in-memory hybrid retriever simulating dense + BM25 + RRF + rerank."""

    def __init__(self, chunks: Sequence[RetrievedChunk | DocumentChunk] = ()) -> None:
        self._chunks_by_key: dict[tuple[str, int], RetrievedChunk] = {}
        self._entries: list[_IndexedEntry] = []
        self._idf: dict[str, float] = {}
        self._lock = asyncio.Lock()
        for raw_chunk in chunks:
            coerced = self._coerce_chunk(raw_chunk, default_rank=len(self._chunks_by_key))
            key = self._chunk_key(coerced)
            self._chunks_by_key[key] = coerced
        self._build_index(list(self._chunks_by_key.values()))

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
        return (chunk.metadata.doc_id, chunk.metadata.chunk_index)

    @staticmethod
    def _coerce_chunk(
        chunk: RetrievedChunk | DocumentChunk,
        *,
        default_rank: int = 0,
    ) -> RetrievedChunk:
        if isinstance(chunk, RetrievedChunk):
            return chunk
        return RetrievedChunk(
            text=chunk.text,
            metadata=chunk.metadata,
            score=0.5,
            rank=default_rank,
        )

    def _build_index(self, chunks: Sequence[RetrievedChunk]) -> None:
        doc_freq: Counter[str] = Counter()
        tokenized_docs: list[tuple[RetrievedChunk, Counter[str], str]] = []
        for chunk in chunks:
            header_text = " ".join(chunk.metadata.section_path)
            metadata_text = f"{chunk.metadata.doc_id} {chunk.metadata.source} {header_text}".strip()
            tokens = _tokenize_text(f"{metadata_text} {chunk.text}")
            counts = Counter(tokens)
            for term in counts:
                doc_freq[term] += 1
            tokenized_docs.append((chunk, counts, metadata_text.lower()))

        total_docs = max(1, len(tokenized_docs))
        self._idf = {
            term: math.log((total_docs + 1.0) / (df + 0.5)) + 1.0 for term, df in doc_freq.items()
        }

        entries: list[_IndexedEntry] = []
        for chunk, counts, header_lower in tokenized_docs:
            total_terms = max(1, sum(counts.values()))
            tf = {term: cnt / total_terms for term, cnt in counts.items()}
            norm = math.sqrt(sum((w * self._idf.get(t, 1.0)) ** 2 for t, w in tf.items())) or 1.0
            entries.append(
                _IndexedEntry(
                    chunk=chunk,
                    tf=tf,
                    norm=norm,
                    header_text=header_lower,
                )
            )
        self._entries = entries

    @classmethod
    def from_markdown(
        cls,
        markdown_text: str,
        *,
        source: str = "builtin://architecture.md",
        doc_id_prefix: str = "doc-custom",
    ) -> InMemoryHybridRetriever:
        """Synchronously build an in-memory retriever from raw Markdown text."""
        chunker = MarkdownHeaderChunker(
            ChunkerConfig(
                mode=ChunkMode.SEMANTIC,
                max_tokens_per_chunk=320,
                min_chunk_chars=40,
            )
        )
        raw_chunks = chunker.chunk(markdown_text, source=source, doc_id=doc_id_prefix)
        chunks = [
            RetrievedChunk(
                text=c.text,
                metadata=ChunkMetadata(
                    doc_id=f"{doc_id_prefix}-{c.metadata.chunk_index}",
                    source=c.metadata.source,
                    section_path=list(c.metadata.section_path),
                    chunk_index=c.metadata.chunk_index,
                ),
                score=0.5,
                rank=idx,
            )
            for idx, c in enumerate(raw_chunks)
        ]
        return cls(chunks)

    @classmethod
    async def from_corpus_paths(
        cls,
        corpus_paths: Sequence[str],
        *,
        base_dir: Path | None = None,
    ) -> InMemoryHybridRetriever:
        """Load and semantically chunk Markdown files off the event loop."""
        root = base_dir or Path.cwd()

        def _load_sync() -> list[RetrievedChunk]:
            chunker = MarkdownHeaderChunker(
                ChunkerConfig(
                    mode=ChunkMode.SEMANTIC,
                    max_tokens_per_chunk=320,
                    min_chunk_chars=40,
                )
            )
            collected: list[RetrievedChunk] = []
            for raw_path in corpus_paths:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if not candidate.is_file():
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8")
                except OSError:
                    continue
                if not content.strip():
                    continue
                slug = _slugify_source(raw_path)
                doc_chunks = chunker.chunk(
                    content,
                    source=Path(raw_path).as_posix(),
                    doc_id=f"doc-{slug}",
                )
                for c in doc_chunks:
                    section_doc_id = f"doc-{slug}-{c.metadata.chunk_index}"
                    collected.append(
                        RetrievedChunk(
                            text=c.text,
                            metadata=ChunkMetadata(
                                doc_id=section_doc_id,
                                source=c.metadata.source,
                                section_path=list(c.metadata.section_path),
                                chunk_index=c.metadata.chunk_index,
                            ),
                            score=0.5,
                            rank=len(collected),
                        )
                    )

            if not collected:
                fallback_chunks = chunker.chunk(
                    DEFAULT_FALLBACK_CORPUS,
                    source="builtin://architecture.md",
                    doc_id="doc-builtin",
                )
                for c in fallback_chunks:
                    collected.append(
                        RetrievedChunk(
                            text=c.text,
                            metadata=ChunkMetadata(
                                doc_id=f"doc-builtin-{c.metadata.chunk_index}",
                                source=c.metadata.source,
                                section_path=list(c.metadata.section_path),
                                chunk_index=c.metadata.chunk_index,
                            ),
                            score=0.5,
                            rank=len(collected),
                        )
                    )
            return collected

        chunks = await asyncio.to_thread(_load_sync)
        return cls(chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._entries)

    def count_chunks(self) -> int:
        """Return the total number of indexed chunks currently in memory."""
        return len(self._entries)

    async def add_chunks(self, chunks: Sequence[Any]) -> int:
        """Incrementally upsert chunks by ``(doc_id, chunk_index)`` and rebuild index."""
        if not chunks:
            return 0
        async with self._lock:
            changed = 0
            for raw_chunk in chunks:
                coerced = self._coerce_chunk(raw_chunk, default_rank=len(self._chunks_by_key))
                key = self._chunk_key(coerced)
                existing = self._chunks_by_key.get(key)
                if existing is None or existing.text != coerced.text:
                    self._chunks_by_key[key] = coerced
                    changed += 1
            if changed > 0:
                self._build_index(list(self._chunks_by_key.values()))
            return changed

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        """Search indexed chunks using v1 dense-like or v2/v3 RRF + rerank scoring."""
        if not self._entries:
            return []

        await asyncio.sleep(0.004 if pipeline_version == "v1" else 0.008)
        query_tokens = _tokenize_text(query)
        query_counts = Counter(query_tokens)
        q_total = max(1, sum(query_counts.values()))
        q_tf = {term: cnt / q_total for term, cnt in query_counts.items()}
        q_norm = math.sqrt(sum((w * self._idf.get(t, 1.0)) ** 2 for t, w in q_tf.items())) or 1.0
        q_unique = set(query_counts.keys())

        dense_scores: list[tuple[int, float]] = []
        bm25_scores: list[tuple[int, float]] = []

        for idx, entry in enumerate(self._entries):
            dot = sum(
                q_tf[t] * entry.tf[t] * (self._idf.get(t, 1.0) ** 2) for t in q_tf if t in entry.tf
            )
            cosine = dot / (q_norm * entry.norm)
            dense_scores.append((idx, cosine))

            bm25 = 0.0
            for term in q_unique:
                tf_val = entry.tf.get(term, 0.0)
                if tf_val > 0.0:
                    idf = self._idf.get(term, 1.0)
                    bm25 += idf * (tf_val * 2.5) / (tf_val + 1.5)
            bm25_scores.append((idx, bm25))

        if pipeline_version == "v1":
            dense_ranked = sorted(dense_scores, key=lambda pair: pair[1], reverse=True)[:top_k]
            return [
                RetrievedChunk(
                    text=self._entries[idx].chunk.text,
                    metadata=self._entries[idx].chunk.metadata,
                    score=round(min(0.99, max(0.35, 0.55 + score * 0.45)), 4),
                    rank=rank,
                )
                for rank, (idx, score) in enumerate(dense_ranked)
            ]

        dense_top = sorted(dense_scores, key=lambda pair: pair[1], reverse=True)[:top_n]
        bm25_top = sorted(bm25_scores, key=lambda pair: pair[1], reverse=True)[:top_n]

        rrf_scores: dict[int, float] = {}
        for rank_pos, (idx, _) in enumerate(dense_top, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60.0 + rank_pos)
        for rank_pos, (idx, _) in enumerate(bm25_top, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60.0 + rank_pos)

        reranked: list[tuple[int, float]] = []
        for idx, rrf in rrf_scores.items():
            entry = self._entries[idx]
            overlap_ratio = (
                len(q_unique & set(entry.tf.keys())) / len(q_unique) if q_unique else 0.0
            )
            header_hits = sum(1 for t in q_unique if len(t) > 1 and t in entry.header_text)
            header_boost = min(0.25, 0.08 * header_hits)
            combined = min(0.99, 0.45 + rrf * 6.0 + overlap_ratio * 0.35 + header_boost)
            reranked.append((idx, round(combined, 4)))

        reranked.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(
                text=self._entries[idx].chunk.text,
                metadata=self._entries[idx].chunk.metadata,
                score=score,
                rank=rank,
            )
            for rank, (idx, score) in enumerate(reranked[:top_k])
        ]


__all__: Sequence[str] = [
    "DEFAULT_FALLBACK_CORPUS",
    "InMemoryHybridRetriever",
]
