from types import SimpleNamespace
from unittest.mock import AsyncMock

from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from app.retrieval.chunker import ChunkMode, DocumentChunker, make_chunker, sanitize_markdown_text
from app.retrieval.hybrid_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    HybridVectorStore,
)


class MockDenseEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_vram_usage_mb(self) -> float:
        return 0.0


class MockSparseEmbedder:
    def embed(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [([1, 2], [0.5, 0.25]) for _ in texts]


class MockReranker:
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [0.2, 0.9, 0.5]

    def get_vram_usage_mb(self) -> float:
        return 0.0


def _point(text: str, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        payload={
            "text": text,
            "doc_id": "doc-1",
            "source": "test.md",
            "section_path": ["Search"],
            "chunk_index": index,
        },
        score=0.1,
    )


def test_markdown_chunks_keep_full_header_hierarchy() -> None:
    text = (
        "Preamble content before headings.\n\n"
        "# Architecture\n"
        "Architecture overview with enough detail to remain a standalone chunk.\n\n"
        "## Retrieval\n"
        "Retrieval details explain dense and sparse search behavior in this section.\n\n"
        "### Reranking\n"
        "Reranking details explain how candidates are ordered after retrieval."
    )

    chunks = DocumentChunker().chunk(text, source="guide.md", doc_id="doc-1")

    assert [chunk.metadata.section_path for chunk in chunks] == [
        [],
        ["Architecture"],
        ["Architecture", "Retrieval"],
        ["Architecture", "Retrieval", "Reranking"],
    ]
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[1].metadata.source == "guide.md"


def test_v1_chunker_uses_fixed_windows_and_overlap() -> None:
    chunker = make_chunker("v1")
    text = "0123456789" * 60

    chunks = chunker.chunk(text, source="legacy.txt", doc_id="doc-1")

    assert chunker.config.mode == ChunkMode.FIXED
    assert len(chunks) == 2
    assert chunks[0].text == text[:512]
    assert chunks[1].text == text[448:]
    assert chunks[1].text[:64] == chunks[0].text[-64:]
    assert all(chunk.metadata.section_path == [] for chunk in chunks)


async def test_hybrid_search_uses_dense_sparse_prefetch_and_rrf() -> None:
    client = AsyncMock()
    client.query_points.return_value = SimpleNamespace(
        points=[_point("first", 0), _point("second", 1), _point("third", 2)]
    )
    store = HybridVectorStore(
        client=client,
        dense_embedder=MockDenseEmbedder(),
        sparse_embedder=MockSparseEmbedder(),
        reranker=MockReranker(),
        collection_name="test_collection",
    )

    results = await store.search("hybrid query", pipeline_version="v3", top_k=2, top_n=3)

    call = client.query_points.await_args
    assert call.kwargs["collection_name"] == "test_collection"
    assert call.kwargs["limit"] == 3
    assert call.kwargs["with_payload"] is True
    assert isinstance(call.kwargs["query"], FusionQuery)
    assert call.kwargs["query"].fusion == Fusion.RRF

    prefetch = call.kwargs["prefetch"]
    assert len(prefetch) == 2
    assert all(isinstance(item, Prefetch) for item in prefetch)
    assert [item.using for item in prefetch] == [DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME]
    assert [item.limit for item in prefetch] == [3, 3]
    assert prefetch[0].query == [0.1, 0.2, 0.3]
    assert isinstance(prefetch[1].query, SparseVector)
    assert prefetch[1].query.indices == [1, 2]
    assert prefetch[1].query.values == [0.5, 0.25]
    assert [result.text for result in results] == ["second", "third"]


async def test_hybrid_search_returns_reranker_order_with_zero_based_ranks() -> None:
    client = AsyncMock()
    client.query_points.return_value = SimpleNamespace(
        points=[_point("candidate-a", 0), _point("candidate-b", 1), _point("candidate-c", 2)]
    )
    store = HybridVectorStore(
        client=client,
        dense_embedder=MockDenseEmbedder(),
        sparse_embedder=MockSparseEmbedder(),
        reranker=MockReranker(),
    )

    results = await store.search("query", top_k=3, top_n=3)

    assert [(result.text, result.score, result.rank) for result in results] == [
        ("candidate-b", 0.9, 0),
        ("candidate-c", 0.5, 1),
        ("candidate-a", 0.2, 2),
    ]


def test_markdown_sanitizer_strips_html_badges_and_nav_links() -> None:
    noisy_doc = (
        "# agentic-rag-eval-engine\n\n"
        '<div align="center">\n\n'
        "[**简体中文**](README.md) | [English](README.en.md) | [한국어](README.ko.md)\n\n"
        "</div>\n\n"
        "[![CI](https://github.com/org/repo/actions/workflows/ci.yml/badge.svg)](https://github.com/org/repo)\n"
        "![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)\n\n"
        "## Overview\n\n"
        "Production-grade Agentic RAG engine for 6 GB VRAM edge GPUs.\n\n"
        "```bash\nuv run pytest -v\n```\n"
    )

    sanitized = sanitize_markdown_text(noisy_doc)
    assert "<div" not in sanitized
    assert "img.shields.io" not in sanitized
    assert "README.en.md" not in sanitized
    assert "uv run pytest -v" in sanitized

    for version in ("v1", "v3"):
        chunks = make_chunker(version).chunk(noisy_doc, source="README.md", doc_id="doc-readme")
        assert len(chunks) >= 1
        assert "<div" not in chunks[0].text
        assert "img.shields.io" not in chunks[0].text
        assert "Production-grade Agentic RAG engine" in chunks[0].text
        assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
