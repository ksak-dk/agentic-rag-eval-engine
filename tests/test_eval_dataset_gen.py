from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.router import RouteTier, TaskType
from app.retrieval.chunker import DocumentChunk, make_chunker
from evals.dataset_gen import (
    EvalSample,
    GeneratedQAPair,
    SyntheticDatasetGenerator,
    aload_dataset_jsonl,
    asave_dataset_jsonl,
    build_offline_dataset,
    load_dataset_jsonl,
    main,
    save_dataset_jsonl,
)


def _build_sample_chunks() -> list[DocumentChunk]:
    chunker = make_chunker("v3")
    doc_a = (
        "# Vector Storage\n\n"
        "Qdrant stores 1024-dimensional bge-m3 dense vectors with INT8 scalar quantization "
        "and in-memory BM25 sparse indices fused via server-side RRF.\n\n"
        "## Reranking\n\n"
        "The top 20 candidates from Qdrant are reranked locally on CUDA in FP16 using "
        "bge-reranker-v2-m3 down to top 5 chunks."
    )
    doc_b = (
        "# Agentic Orchestration\n\n"
        "LangGraph coordinates cache lookup, routing, retrieval, grading, generation, and "
        "hallucination checking with AsyncSqliteSaver checkpoints.\n\n"
        "## Retry Guard\n\n"
        "Query rewriting is capped at max_retries=2 to prevent infinite graph cycles."
    )
    chunks_a = chunker.chunk(doc_a, source="docs/retrieval.md", doc_id="doc-retrieval")
    chunks_b = chunker.chunk(doc_b, source="docs/langgraph.md", doc_id="doc-langgraph")
    return [*chunks_a, *chunks_b]


class MockDatasetRouter:
    """Offline mock LLMRouter tracking concurrency, route tiers, and structured outputs."""

    def __init__(
        self,
        responses: list[GeneratedQAPair | Exception] | None = None,
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        self._responses = list(responses or [])
        self._delay_seconds = delay_seconds
        self._call_index = 0
        self.active_calls = 0
        self.max_in_flight = 0
        self.calls: list[dict[str, Any]] = []

    async def ainvoke_structured(
        self,
        schema: type[Any],
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> GeneratedQAPair:
        self.active_calls += 1
        self.max_in_flight = max(self.max_in_flight, self.active_calls)
        try:
            if self._delay_seconds > 0:
                await asyncio.sleep(self._delay_seconds)

            idx = self._call_index
            self._call_index += 1
            self.calls.append(
                {
                    "schema": schema,
                    "messages": messages,
                    "task_type": task_type,
                    "force_tier": force_tier,
                }
            )
            if idx < len(self._responses):
                item = self._responses[idx]
                if isinstance(item, Exception):
                    raise item
                return item
            return GeneratedQAPair(
                question=f"Synthetic question {idx} about architecture?",
                ground_truth=f"Synthetic grounded answer {idx}.",
                is_valid=True,
            )
        finally:
            self.active_calls -= 1


async def test_generate_single_and_multi_hop_samples_via_strong_route() -> None:
    chunks = _build_sample_chunks()
    router = MockDatasetRouter(
        responses=[
            GeneratedQAPair(
                question="What quantization does Qdrant use for dense vectors?",
                ground_truth="Qdrant uses INT8 scalar quantization for dense vectors.",
                is_valid=True,
            ),
            GeneratedQAPair(
                question="What caps LangGraph query rewrite loops?",
                ground_truth="Query rewriting is capped at max_retries=2.",
                is_valid=True,
            ),
            GeneratedQAPair(
                question="How do Qdrant reranking and LangGraph retries cooperate?",
                ground_truth=(
                    "Qdrant reranks top 20 candidates to top 5 chunks, and LangGraph caps "
                    "query rewrites at max_retries=2."
                ),
                is_valid=True,
            ),
        ]
    )
    generator = SyntheticDatasetGenerator(
        router=router,
        max_concurrency=2,
        max_retries=1,
        retry_base_delay=0.0,
    )

    samples = await generator.generate(chunks, single_hop_count=2, multi_hop_count=1)

    assert len(samples) == 3
    single_samples = [s for s in samples if s.hop_type == "single_hop"]
    multi_samples = [s for s in samples if s.hop_type == "multi_hop"]
    assert len(single_samples) == 2
    assert len(multi_samples) == 1

    assert all(len(s.reference_contexts) == 1 for s in single_samples)
    assert all(len(s.source_doc_ids) == 1 for s in single_samples)
    assert len(multi_samples[0].reference_contexts) == 2
    assert len(multi_samples[0].source_doc_ids) >= 1

    assert all(call["task_type"] == TaskType.EVAL_JUDGE for call in router.calls)
    assert all(call["force_tier"] == RouteTier.STRONG for call in router.calls)
    assert all(call["schema"] is GeneratedQAPair for call in router.calls)


async def test_concurrency_semaphore_and_rate_limit_retry() -> None:
    chunks = _build_sample_chunks()
    responses: list[GeneratedQAPair | Exception] = [
        RuntimeError("429 Too Many Requests"),
        GeneratedQAPair(
            question="Recovered after 429 rate limit?",
            ground_truth="Yes, exponential backoff retry succeeded.",
            is_valid=True,
        ),
        RuntimeError("Persistent 503 error"),
        RuntimeError("Persistent 503 error"),
    ]
    router = MockDatasetRouter(responses=responses, delay_seconds=0.015)
    generator = SyntheticDatasetGenerator(
        router=router,
        max_concurrency=2,
        max_retries=1,
        retry_base_delay=0.005,
    )

    samples = await generator.generate(chunks, single_hop_count=3, multi_hop_count=3)

    assert router.max_in_flight <= 2
    assert any("Recovered after 429" in s.question for s in samples)
    # 1 recovered + 4 default responses (1 candidate exhausted 2 attempts and failed soft)
    assert len(samples) == 5


async def test_filters_invalid_and_duplicate_responses() -> None:
    chunks = _build_sample_chunks()
    router = MockDatasetRouter(
        responses=[
            GeneratedQAPair(
                question="Valid first question?", ground_truth="Answer 1.", is_valid=True
            ),
            GeneratedQAPair(question="   ", ground_truth="Blank question", is_valid=True),
            GeneratedQAPair(question="N/A", ground_truth="Placeholder", is_valid=True),
            GeneratedQAPair(question="Low factuality?", ground_truth="N/A", is_valid=True),
            GeneratedQAPair(
                question="Flagged invalid?",
                ground_truth="Should be dropped.",
                is_valid=False,
            ),
            GeneratedQAPair(
                question="valid first question?",
                ground_truth="Duplicate question.",
                is_valid=True,
            ),
        ]
    )
    generator = SyntheticDatasetGenerator(
        router=router,
        max_concurrency=2,
        max_retries=0,
        retry_base_delay=0.0,
    )

    samples = await generator.generate(chunks, single_hop_count=4, multi_hop_count=2)

    assert len(samples) == 1
    assert samples[0].question == "Valid first question?"


async def test_jsonl_roundtrip_and_validation_errors(tmp_path: Path) -> None:
    sample = EvalSample(
        sample_id="sample-1",
        question="How does hybrid search fuse dense and sparse rankings?",
        ground_truth="Qdrant applies server-side Fusion.RRF over dense and BM25 sparse prefetches.",
        reference_contexts=[
            "Dense bge-m3 and sparse BM25 candidates are fused via Fusion.RRF.",
            "Reranking narrows top 20 candidates to top 5.",
        ],
        hop_type="multi_hop",
        source_doc_ids=["doc-a", "doc-b", "doc-a"],
    )
    assert sample.source_doc_ids == ["doc-a", "doc-b"]

    target = tmp_path / "nested" / "eval_dataset.jsonl"
    saved_path = save_dataset_jsonl([sample], target)
    assert saved_path.exists()

    loaded = load_dataset_jsonl(target)
    assert loaded == [sample]

    async_target = tmp_path / "async_eval.jsonl"
    await asave_dataset_jsonl([sample], async_target)
    async_loaded = await aload_dataset_jsonl(async_target)
    assert async_loaded == [sample]

    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"question": "only question"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid EvalSample"):
        load_dataset_jsonl(bad_file)

    with pytest.raises(FileNotFoundError):
        load_dataset_jsonl(tmp_path / "missing.jsonl")

    with pytest.raises(ValidationError):
        EvalSample(
            question="   ",
            ground_truth="Valid truth",
            reference_contexts=["ctx"],
            hop_type="single_hop",
        )


def test_offline_dataset_builder_and_cli(tmp_path: Path) -> None:
    chunks = _build_sample_chunks()
    offline_samples = build_offline_dataset(
        chunks,
        single_hop_count=2,
        multi_hop_count=2,
        seed=7,
    )
    assert len(offline_samples) == 4
    assert [s.hop_type for s in offline_samples].count("single_hop") == 2
    assert [s.hop_type for s in offline_samples].count("multi_hop") == 2
    assert all(len(s.reference_contexts) == 2 for s in offline_samples if s.hop_type == "multi_hop")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "retrieval.md").write_text(
        "# Retrieval\n\nQdrant uses INT8 quantization and RRF fusion for hybrid search.",
        encoding="utf-8",
    )
    (docs_dir / "orchestration.md").write_text(
        "# Orchestration\n\nLangGraph persists state with AsyncSqliteSaver checkpoints.",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "out" / "eval_dataset.jsonl"

    exit_code = main(
        [
            str(docs_dir),
            "--output",
            str(output_jsonl),
            "--single-hop",
            "2",
            "--multi-hop",
            "1",
            "--offline",
        ]
    )
    assert exit_code == 0
    cli_samples = load_dataset_jsonl(output_jsonl)
    assert len(cli_samples) == 3
    assert sum(1 for s in cli_samples if s.hop_type == "single_hop") == 2
    assert sum(1 for s in cli_samples if s.hop_type == "multi_hop") == 1
