from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from evals.dataset_gen import EvalSample, save_dataset_jsonl
from evals.run_ragas import (
    PipelineVersion,
    RagasComparisonReport,
    RagasPipelineEvaluator,
    build_evaluation_dataset,
    build_ragas_metrics,
    build_ragas_run_config,
    load_comparison_report,
    main,
    save_comparison_report,
)


def _make_eval_samples() -> list[EvalSample]:
    return [
        EvalSample(
            sample_id="single-1",
            question="How does Qdrant compress dense vectors in memory?",
            ground_truth="Qdrant applies INT8 scalar quantization with always_ram=True.",
            reference_contexts=[
                "Qdrant stores 1024-dimensional bge-m3 dense vectors using INT8 scalar "
                "quantization with always_ram=True for low-memory HNSW search."
            ],
            hop_type="single_hop",
            source_doc_ids=["doc-qdrant"],
        ),
        EvalSample(
            sample_id="multi-1",
            question="How do Qdrant RRF fusion and LangGraph retries ensure answer quality?",
            ground_truth=(
                "Qdrant fuses dense and BM25 sparse candidates via RRF before FP16 reranking, "
                "and LangGraph caps query rewrite retries at max_retries=2."
            ),
            reference_contexts=[
                "Qdrant fuses dense and BM25 sparse prefetches using server-side Fusion.RRF "
                "before local FP16 bge-reranker-v2-m3 reranking.",
                "LangGraph enforces max_retries=2 across document grading and hallucination "
                "checks before triggering deterministic fallback.",
            ],
            hop_type="multi_hop",
            source_doc_ids=["doc-qdrant", "doc-langgraph"],
        ),
    ]


def test_ragas_run_config_and_oop_metrics() -> None:
    run_config = build_ragas_run_config()
    assert isinstance(run_config, RunConfig)
    assert run_config.max_workers == 2
    assert run_config.max_retries == 5
    assert run_config.max_wait == 60

    fake_judge = LangchainLLMWrapper(FakeListChatModel(responses=["{}"]))
    metrics = build_ragas_metrics(fake_judge)
    assert len(metrics) == 3
    assert isinstance(metrics[0], LLMContextRecall)
    assert isinstance(metrics[1], Faithfulness)
    assert isinstance(metrics[2], LLMContextPrecisionWithReference)
    assert all(metric.llm is fake_judge for metric in metrics)

    dataset = build_evaluation_dataset(
        [
            SingleTurnSample(
                user_input="q",
                response="a",
                retrieved_contexts=["ctx"],
                reference="a",
                reference_contexts=["ctx"],
            )
        ]
    )
    assert len(dataset.samples) == 1


async def test_evaluate_sample_builds_single_turn_sample_and_record() -> None:
    sample = _make_eval_samples()[1]
    captured_turns: list[tuple[SingleTurnSample, PipelineVersion]] = []

    async def mock_runner(
        eval_sample: EvalSample,
        version: PipelineVersion,
    ) -> dict[str, Any]:
        return {
            "answer": f"[{version}] {eval_sample.ground_truth}",
            "retrieved_contexts": list(eval_sample.reference_contexts),
            "latency_ms": 812.5,
            "ttft_ms": 195.25,
            "rewrite_count": 1,
            "cache_hit": False,
        }

    async def mock_scorer(
        turn_sample: SingleTurnSample,
        version: PipelineVersion,
    ) -> dict[str, float]:
        captured_turns.append((turn_sample, version))
        return {
            "context_recall": 0.92,
            "faithfulness": 0.95,
            "context_precision": 0.88,
        }

    evaluator = RagasPipelineEvaluator(runner=mock_runner, scorer=mock_scorer)
    record = await evaluator.evaluate_sample(sample, "v3")

    assert len(captured_turns) == 1
    turn, ver = captured_turns[0]
    assert ver == "v3"
    assert turn.user_input == sample.question
    assert turn.response == f"[v3] {sample.ground_truth}"
    assert turn.retrieved_contexts == sample.reference_contexts
    assert turn.reference == sample.ground_truth
    assert turn.reference_contexts == sample.reference_contexts

    assert record.sample_id == "multi-1"
    assert record.pipeline_version == "v3"
    assert record.hop_type == "multi_hop"
    assert record.context_recall == 0.92
    assert record.faithfulness == 0.95
    assert record.context_precision == 0.88
    assert record.latency_ms == 812.5
    assert record.ttft_ms == 195.25
    assert record.rewrite_count == 1
    assert record.cache_hit is False


async def test_compare_pipelines_evolution_and_p95_aggregation() -> None:
    samples = _make_eval_samples()
    offline_evaluator = RagasPipelineEvaluator(offline_mode=True)
    report = await offline_evaluator.compare_pipelines(samples, versions=("v1", "v2", "v3"))

    assert set(report.summaries.keys()) == {"v1", "v2", "v3"}
    assert len(report.records) == 6
    v1 = report.summaries["v1"]
    v2 = report.summaries["v2"]
    v3 = report.summaries["v3"]
    assert v3.overall_score > v2.overall_score > v1.overall_score

    # Verify exact P50 / P95 percentile math on controlled latencies
    five_samples = [
        EvalSample(
            sample_id=f"s-{i}",
            question=f"Question {i}?",
            ground_truth=f"Answer {i}.",
            reference_contexts=[f"Context {i}."],
            hop_type="single_hop",
            source_doc_ids=[f"doc-{i}"],
        )
        for i in range(5)
    ]
    latencies = {"s-0": 100.0, "s-1": 200.0, "s-2": 300.0, "s-3": 400.0, "s-4": 500.0}

    async def latency_runner(s: EvalSample, version: PipelineVersion) -> dict[str, Any]:
        _ = version
        return {
            "answer": s.ground_truth,
            "retrieved_contexts": s.reference_contexts,
            "latency_ms": latencies[s.sample_id],
            "ttft_ms": 50.0,
        }

    async def fixed_scorer(
        turn: SingleTurnSample,
        version: PipelineVersion,
    ) -> dict[str, float]:
        _ = (turn, version)
        return {"context_recall": 0.9, "faithfulness": 0.6, "context_precision": 0.9}

    math_evaluator = RagasPipelineEvaluator(runner=latency_runner, scorer=fixed_scorer)
    summary, _ = await math_evaluator.evaluate_pipeline(five_samples, "v2")
    assert summary.sample_count == 5
    assert summary.mean_context_recall == 0.9
    assert summary.mean_faithfulness == 0.6
    assert summary.mean_context_precision == 0.9
    assert summary.overall_score == 0.8
    assert summary.p50_latency_ms == 300.0
    assert summary.p95_latency_ms == 480.0
    assert summary.mean_ttft_ms == 50.0


async def test_scorer_retry_clamping_and_concurrency_limit() -> None:
    samples = [
        EvalSample(
            sample_id=f"conc-{i}",
            question=f"Concurrency question {i}?",
            ground_truth=f"Concurrency answer {i}.",
            reference_contexts=[f"Concurrency context {i}."],
            hop_type="single_hop",
            source_doc_ids=[f"doc-{i}"],
        )
        for i in range(4)
    ]

    active_calls = 0
    max_in_flight = 0
    attempts_by_question: dict[str, int] = {}

    async def simple_runner(sample: EvalSample, version: PipelineVersion) -> dict[str, Any]:
        _ = version
        return {
            "answer": sample.ground_truth,
            "retrieved_contexts": sample.reference_contexts,
            "latency_ms": 120.0,
            "ttft_ms": 40.0,
        }

    async def flaky_clamping_scorer(
        turn_sample: SingleTurnSample,
        version: PipelineVersion,
    ) -> dict[str, float]:
        nonlocal active_calls, max_in_flight
        _ = version
        active_calls += 1
        max_in_flight = max(max_in_flight, active_calls)
        try:
            await asyncio.sleep(0.015)
            q = turn_sample.user_input or ""
            attempts_by_question[q] = attempts_by_question.get(q, 0) + 1
            if q == "Concurrency question 0?" and attempts_by_question[q] == 1:
                raise RuntimeError("429 RateLimit exceeded")
            return {
                "context_recall": 1.45,
                "faithfulness": -0.25,
                "context_precision": float("nan"),
            }
        finally:
            active_calls -= 1

    evaluator = RagasPipelineEvaluator(
        runner=simple_runner,
        scorer=flaky_clamping_scorer,
        run_config=build_ragas_run_config(max_workers=2, max_retries=2, max_wait=1),
        max_concurrency=2,
        retry_base_delay=0.005,
    )
    summary, records = await evaluator.evaluate_pipeline(samples, "v3")

    assert max_in_flight <= 2
    assert attempts_by_question["Concurrency question 0?"] == 2
    assert len(records) == 4
    assert all(r.context_recall == 1.0 for r in records)
    assert all(r.faithfulness == 0.0 for r in records)
    assert all(r.context_precision == 0.0 for r in records)
    assert summary.mean_context_recall == 1.0
    assert summary.mean_faithfulness == 0.0


async def test_report_persistence_markdown_and_cli(tmp_path: Path) -> None:
    samples = _make_eval_samples()
    evaluator = RagasPipelineEvaluator(offline_mode=True)
    report = await evaluator.compare_pipelines(samples)

    markdown = report.to_markdown_table()
    for token in (
        "Context Recall",
        "Faithfulness",
        "Context Precision",
        "Overall Score",
        "P50 Latency (ms)",
        "P95 Latency (ms)",
        "| v1 |",
        "| v2 |",
        "| v3 |",
    ):
        assert token in markdown

    output_json = tmp_path / "reports" / "ragas_eval_results.json"
    save_comparison_report(report, output_json)
    loaded = load_comparison_report(output_json)
    assert loaded == report

    with pytest.raises(FileNotFoundError):
        load_comparison_report(tmp_path / "missing.json")

    bad_json = tmp_path / "bad_report.json"
    bad_json.write_text('{"summaries": "invalid"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid RagasComparisonReport"):
        load_comparison_report(bad_json)

    # CLI offline execution smoke test
    dataset_path = tmp_path / "eval_dataset.jsonl"
    cli_output = tmp_path / "cli_ragas_results.json"
    save_dataset_jsonl(samples, dataset_path)

    exit_code = await asyncio.to_thread(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(cli_output),
            "--offline",
        ],
    )
    assert exit_code == 0
    cli_report = RagasComparisonReport.model_validate(
        load_comparison_report(cli_output).model_dump()
    )
    assert set(cli_report.summaries.keys()) == {"v1", "v2", "v3"}
