from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.retrieval.chunker import make_chunker
from evals.dataset_gen import build_offline_dataset
from evals.load_test import (
    AsyncLoadTester,
    LoadTestSummary,
    generate_benchmark_report_markdown,
    save_benchmark_report,
    save_load_test_results,
)
from evals.run_ragas import RagasPipelineEvaluator


async def test_single_sse_request_collects_ttft_and_total_latency() -> None:
    tester = AsyncLoadTester(concurrency=4, mock_llm_stream=True, cache_ratio=0.0)
    async with tester.open_client() as client:
        stream_record = await tester.send_single_request(
            client,
            "How does Qdrant INT8 quantization work?",
            pipeline_version="v3",
            stream=True,
        )
        sync_record = await tester.send_single_request(
            client,
            "How does RRF fusion work?",
            pipeline_version="v2",
            stream=False,
        )

    assert stream_record.is_success is True
    assert stream_record.status_code == 200
    assert stream_record.stream_mode is True
    assert 0.0 < stream_record.ttft_ms <= stream_record.total_latency_ms
    assert stream_record.request_id

    assert sync_record.is_success is True
    assert sync_record.status_code == 200
    assert sync_record.stream_mode is False
    assert 0.0 < sync_record.ttft_ms <= sync_record.total_latency_ms


async def test_concurrent_benchmark_meets_p95_sla() -> None:
    tester = AsyncLoadTester(concurrency=5, mock_llm_stream=True, cache_ratio=0.3)
    summary, records = await tester.run_benchmark(
        total_requests=20,
        pipeline_version="v3",
        stream=True,
    )

    assert len(records) == 20
    assert summary.pipeline_version == "v3"
    assert summary.total_requests == 20
    assert summary.concurrency == 5
    assert summary.mock_llm_stream is True
    assert summary.success_rate == 1.0
    assert summary.p95_latency_ms < 1800.0
    assert summary.p95_sla_passed is True
    assert summary.rps > 0.0
    assert 0.0 < summary.mean_ttft_ms <= summary.p95_latency_ms


async def test_semantic_cache_hits_reduce_average_latency() -> None:
    tester = AsyncLoadTester(concurrency=4, mock_llm_stream=True, cache_ratio=0.4)
    summary, records = await tester.run_benchmark(
        queries=[
            "Explain Qdrant INT8 quantization and RRF fusion",
            "How does LangGraph persist checkpoints?",
            "How does Redis semantic cache work?",
        ],
        total_requests=20,
        pipeline_version="v3",
        stream=True,
    )

    hit_latencies = [r.total_latency_ms for r in records if r.cache_hit]
    miss_latencies = [r.total_latency_ms for r in records if not r.cache_hit]

    assert summary.cache_hit_rate > 0.0
    assert len(hit_latencies) >= 6
    assert len(miss_latencies) >= 6
    mean_hit = sum(hit_latencies) / len(hit_latencies)
    mean_miss = sum(miss_latencies) / len(miss_latencies)
    assert mean_hit < mean_miss


async def test_compare_versions_load_and_mock_env_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_LLM_STREAM", "false")
    env_false_tester = AsyncLoadTester(base_url="http://localhost:8000")
    assert env_false_tester.mock_llm_stream is False

    monkeypatch.setenv("MOCK_LLM_STREAM", "true")
    env_true_tester = AsyncLoadTester(concurrency=4, cache_ratio=0.3)
    assert env_true_tester.mock_llm_stream is True

    summaries = await env_true_tester.compare_versions_load(
        total_requests_per_version=6,
        versions=("v1", "v2", "v3"),
    )

    assert set(summaries.keys()) == {"v1", "v2", "v3"}
    for version in ("v1", "v2", "v3"):
        summary = summaries[version]
        assert isinstance(summary, LoadTestSummary)
        assert summary.pipeline_version == version
        assert summary.total_requests == 6
        assert summary.p95_sla_passed is True
    assert summaries["v3"].cache_hit_rate > 0.0


async def test_generate_and_persist_benchmark_report_markdown(tmp_path: Path) -> None:
    chunker = make_chunker("v3")
    chunks = [
        *chunker.chunk(
            "# Hybrid Retrieval\n\nQdrant stores INT8 dense vectors and BM25 sparse vectors.",
            source="docs/retrieval.md",
            doc_id="doc-qdrant",
        ),
        *chunker.chunk(
            "# LangGraph\n\nLangGraph enforces max_retries=2 with AsyncSqliteSaver.",
            source="docs/graph.md",
            doc_id="doc-graph",
        ),
    ]
    samples = build_offline_dataset(chunks, single_hop_count=2, multi_hop_count=2, seed=11)

    ragas_report = await RagasPipelineEvaluator(offline_mode=True).compare_pipelines(samples)
    tester = AsyncLoadTester(concurrency=4, mock_llm_stream=True, cache_ratio=0.3)
    load_summaries = await tester.compare_versions_load(total_requests_per_version=6)

    markdown = generate_benchmark_report_markdown(ragas_report, load_summaries)
    for expected in (
        "RTX 3060",
        "Context Recall",
        "Faithfulness",
        "Context Precision",
        "P95",
        "| v1 |",
        "| v2 |",
        "| v3 |",
        "PASS",
    ):
        assert expected in markdown

    report_path = save_benchmark_report(markdown, tmp_path / "benchmark_report.md")
    json_path = save_load_test_results(load_summaries, tmp_path / "load_test_results.json")

    assert report_path.exists()
    assert json_path.exists()
    assert "RTX 3060" in report_path.read_text(encoding="utf-8")
    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(parsed_json.keys()) == {"v1", "v2", "v3"}
