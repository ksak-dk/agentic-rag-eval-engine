"""
evals/run_ragas.py
==================
Automated Ragas v0.2+ multi-pipeline comparison evaluator (v1 vs v2 vs v3).

Design & Hardware Contract
--------------------------
- Uses Ragas v0.2+ OOP metrics exclusively:
  ``LLMContextRecall``, ``Faithfulness``, ``LLMContextPrecisionWithReference``.
- Enforces ``RunConfig(max_workers=2, max_retries=5, max_wait=60)`` and bounded
  async concurrency (``asyncio.Semaphore(2)``) to respect cloud API limits.
- Uses ``LLMRouter`` strong route (``gemini-2.5-flash``) wrapped via
  ``LangchainLLMWrapper`` for Judge evaluation. Never loads any local LLM.
- Supports offline deterministic simulation and heuristic scoring for CI and
  unit tests without external API calls or CUDA dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from app.adapters.mock.eval_heuristics import (
    clamp_score,
    compute_percentile,
    heuristic_ragas_scorer,
    offline_simulated_runner,
)
from app.core.config import Settings, get_settings
from app.core.router import LLMRouter, RouteTier, TaskType
from app.graph.nodes import GraphDependencies, _format_context_block
from app.graph.workflow import AgenticRAGWorkflow
from evals.dataset_gen import EvalSample, HopType, load_dataset_jsonl

logger = structlog.get_logger(__name__)

PipelineVersion = Literal["v1", "v2", "v3"]
PIPELINE_VERSIONS: tuple[PipelineVersion, ...] = ("v1", "v2", "v3")
DEFAULT_DATASET_PATH: Path = Path("./data/eval_dataset.jsonl")
DEFAULT_RESULTS_PATH: Path = Path("./data/ragas_eval_results.json")

PipelineRunner = Callable[[EvalSample, PipelineVersion], Awaitable[dict[str, Any]]]
SampleScorer = Callable[[SingleTurnSample, PipelineVersion], Awaitable[dict[str, float]]]


# ---------------------------------------------------------------------------
# Pydantic V2 Models
# ---------------------------------------------------------------------------


class SampleEvalRecord(BaseModel):
    """Per-sample Ragas evaluation scores and runtime telemetry."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(..., min_length=1)
    pipeline_version: PipelineVersion
    hop_type: HopType
    question: str
    ground_truth: str
    answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    reference_contexts: list[str] = Field(default_factory=list)
    context_recall: float = Field(..., ge=0.0, le=1.0)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0)
    ttft_ms: float = Field(..., ge=0.0)
    rewrite_count: int = Field(default=0, ge=0)
    cache_hit: bool = False


class PipelineEvalSummary(BaseModel):
    """Aggregated Ragas and latency summary for one pipeline version."""

    model_config = ConfigDict(frozen=True)

    pipeline_version: PipelineVersion
    sample_count: int = Field(..., ge=0)
    mean_context_recall: float = Field(..., ge=0.0, le=1.0)
    mean_faithfulness: float = Field(..., ge=0.0, le=1.0)
    mean_context_precision: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    p50_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    mean_ttft_ms: float = Field(..., ge=0.0)


class RagasComparisonReport(BaseModel):
    """Full v1/v2/v3 benchmark report with Markdown rendering."""

    model_config = ConfigDict(frozen=True)

    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    judge_model: str = Field(default="gemini-2.5-flash")
    summaries: dict[PipelineVersion, PipelineEvalSummary] = Field(default_factory=dict)
    records: list[SampleEvalRecord] = Field(default_factory=list)

    @field_validator("judge_model")
    @classmethod
    def _validate_judge_model(cls, value: str) -> str:
        cleaned = value.strip()
        return cleaned or "gemini-2.5-flash"

    def to_markdown_table(self) -> str:
        """Render a Markdown comparison table across evaluated pipeline versions."""
        lines = [
            "| Pipeline | Samples | Context Recall | Faithfulness | Context Precision "
            "| Overall Score | P50 Latency (ms) | P95 Latency (ms) | Mean TTFT (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        ordered_versions: list[PipelineVersion] = [
            version for version in PIPELINE_VERSIONS if version in self.summaries
        ]
        for version in self.summaries:
            if version not in ordered_versions:
                ordered_versions.append(version)

        for version in ordered_versions:
            s = self.summaries[version]
            lines.append(
                f"| {s.pipeline_version} | {s.sample_count} "
                f"| {s.mean_context_recall:.4f} | {s.mean_faithfulness:.4f} "
                f"| {s.mean_context_precision:.4f} | {s.overall_score:.4f} "
                f"| {s.p50_latency_ms:.2f} | {s.p95_latency_ms:.2f} | {s.mean_ttft_ms:.2f} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ragas v0.2+ Factories
# ---------------------------------------------------------------------------


def build_ragas_run_config(
    max_workers: int = 2,
    max_retries: int = 5,
    max_wait: int = 60,
) -> RunConfig:
    """Build rate-limit-safe Ragas ``RunConfig`` defaults for Google AI Studio."""
    return RunConfig(
        max_workers=max_workers,
        max_retries=max_retries,
        max_wait=max_wait,
    )


def build_ragas_metrics(judge_llm: Any | None = None) -> list[Any]:
    """Instantiate Ragas v0.2+ OOP metric objects (no deprecated singletons)."""
    return [
        LLMContextRecall(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        LLMContextPrecisionWithReference(llm=judge_llm),
    ]


def build_evaluation_dataset(samples: Sequence[SingleTurnSample]) -> EvaluationDataset:
    """Construct a Ragas ``EvaluationDataset`` from ``SingleTurnSample`` instances."""
    return EvaluationDataset(samples=list(samples))


def build_judge_llm(
    *,
    router: Any | None = None,
    settings: Settings | None = None,
) -> LangchainLLMWrapper:
    """Wrap the cloud strong-route chat model (`gemini-2.5-flash`) for Ragas."""
    resolved_router = router
    if resolved_router is None:
        resolved_settings = settings or _try_get_settings()
        if resolved_settings is None:
            raise ValueError("Valid 'settings' or 'router' is required to construct Judge LLM.")
        resolved_router = LLMRouter(settings=resolved_settings)

    chat_model = resolved_router.get_chat_model(
        RouteTier.STRONG,
        temperature=0.0,
        with_fallback=False,
    )
    return LangchainLLMWrapper(chat_model)


# ---------------------------------------------------------------------------
# Math, Scoring & Simulation Helpers
# ---------------------------------------------------------------------------


def _try_get_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


_clamp_score = clamp_score
_compute_percentile = compute_percentile
_offline_simulated_runner = offline_simulated_runner


def _aggregate_pipeline_summary(
    pipeline_version: PipelineVersion,
    records: Sequence[SampleEvalRecord],
) -> PipelineEvalSummary:
    if not records:
        return PipelineEvalSummary(
            pipeline_version=pipeline_version,
            sample_count=0,
            mean_context_recall=0.0,
            mean_faithfulness=0.0,
            mean_context_precision=0.0,
            overall_score=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            mean_ttft_ms=0.0,
        )

    count = len(records)
    mean_recall = round(sum(r.context_recall for r in records) / count, 4)
    mean_faith = round(sum(r.faithfulness for r in records) / count, 4)
    mean_prec = round(sum(r.context_precision for r in records) / count, 4)
    overall = round((mean_recall + mean_faith + mean_prec) / 3.0, 4)
    latencies = [r.latency_ms for r in records]
    mean_ttft = round(sum(r.ttft_ms for r in records) / count, 2)

    return PipelineEvalSummary(
        pipeline_version=pipeline_version,
        sample_count=count,
        mean_context_recall=_clamp_score(mean_recall),
        mean_faithfulness=_clamp_score(mean_faith),
        mean_context_precision=_clamp_score(mean_prec),
        overall_score=_clamp_score(overall),
        p50_latency_ms=_compute_percentile(latencies, 50.0),
        p95_latency_ms=_compute_percentile(latencies, 95.0),
        mean_ttft_ms=mean_ttft,
    )


# ---------------------------------------------------------------------------
# Ragas Pipeline Evaluator
# ---------------------------------------------------------------------------


class RagasPipelineEvaluator:
    """Async multi-pipeline evaluator computing Ragas v0.2+ metrics and latencies."""

    def __init__(
        self,
        *,
        runner: PipelineRunner | None = None,
        scorer: SampleScorer | None = None,
        router: Any | None = None,
        settings: Settings | None = None,
        run_config: RunConfig | None = None,
        max_concurrency: int = 2,
        judge_llm: Any | None = None,
        offline_mode: bool = False,
        retriever: Any | None = None,
        cache: Any | None = None,
        checkpoint_db: Path | None = None,
        retry_base_delay: float = 0.05,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}.")
        if retry_base_delay < 0.0:
            raise ValueError(f"retry_base_delay must be >= 0.0, got {retry_base_delay}.")

        self._settings = settings if settings is not None else _try_get_settings()
        self._router = router
        self._retriever = retriever
        self._cache = cache
        self._checkpoint_db = checkpoint_db
        self._offline_mode = offline_mode
        self._max_concurrency = max_concurrency
        self._retry_base_delay = retry_base_delay
        self._run_config = run_config or build_ragas_run_config(max_workers=max_concurrency)
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._custom_runner = runner
        self._custom_scorer = scorer
        self._judge_llm = judge_llm
        self._ragas_metrics: list[Any] | None = None

        self._judge_model_name = (
            self._settings.strong_llm_model if self._settings is not None else "gemini-2.5-flash"
        )

    @property
    def run_config(self) -> RunConfig:
        return self._run_config

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def _get_or_init_ragas_metrics(self) -> list[Any]:
        if self._ragas_metrics is None:
            judge = self._judge_llm
            if judge is None:
                judge = build_judge_llm(router=self._router, settings=self._settings)
                self._judge_llm = judge
            self._ragas_metrics = build_ragas_metrics(judge)
        return self._ragas_metrics

    async def _run_real_pipeline(
        self,
        sample: EvalSample,
        pipeline_version: PipelineVersion,
    ) -> dict[str, Any]:
        if self._retriever is None or self._router is None:
            raise ValueError(
                "Either custom 'runner', 'offline_mode=True', or both 'retriever' and "
                "'router' must be provided."
            )

        top_k = self._settings.retrieval_top_k if self._settings is not None else 5
        top_n = self._settings.retrieval_top_n if self._settings is not None else 20
        max_retries = self._settings.max_retries if self._settings is not None else 2
        start = time.perf_counter()

        if pipeline_version in ("v1", "v2"):
            docs = await self._retriever.search(
                sample.question,
                pipeline_version=pipeline_version,
                top_k=top_k,
                top_n=top_n,
            )
            context_block = _format_context_block(docs)
            messages = [
                (
                    "system",
                    "Answer the user question using ONLY the provided context snippets.",
                ),
                ("user", f"Question: {sample.question}\n\nContext:\n{context_block}"),
            ]
            force_tier = RouteTier.FAST if pipeline_version == "v1" else RouteTier.STRONG
            gen_result = await self._router.ainvoke(
                messages,
                task_type=TaskType.GENERATION,
                force_tier=force_tier,
            )
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            return {
                "answer": str(gen_result.content),
                "retrieved_contexts": [d.text for d in docs],
                "latency_ms": float(gen_result.latency_ms or elapsed_ms),
                "ttft_ms": float(gen_result.ttft_ms or elapsed_ms),
                "rewrite_count": 0,
                "cache_hit": False,
            }

        graph_deps = GraphDependencies(
            retriever=self._retriever,
            router=self._router,
            cache=self._cache,
            top_k=top_k,
            top_n=top_n,
            max_retries=max_retries,
        )
        workflow = AgenticRAGWorkflow(
            graph_deps,
            db_path=self._checkpoint_db,
            max_retries=max_retries,
        )
        state = await workflow.ainvoke(
            sample.question,
            session_id=f"eval-{sample.sample_id}",
            pipeline_version="v3",
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        docs = list(state.get("graded_documents") or state.get("documents") or [])
        return {
            "answer": str(state.get("answer") or ""),
            "retrieved_contexts": [d.text if hasattr(d, "text") else str(d) for d in docs],
            "latency_ms": elapsed_ms,
            "ttft_ms": elapsed_ms,
            "rewrite_count": int(state.get("retry_count", 0)),
            "cache_hit": bool(state.get("cache_hit", False)),
        }

    async def _execute_runner(
        self,
        sample: EvalSample,
        pipeline_version: PipelineVersion,
    ) -> dict[str, Any]:
        if self._custom_runner is not None:
            return await self._custom_runner(sample, pipeline_version)
        if self._offline_mode:
            return await _offline_simulated_runner(sample, pipeline_version)
        return await self._run_real_pipeline(sample, pipeline_version)

    async def _score_with_ragas_oop(self, sample: SingleTurnSample) -> dict[str, float]:
        metrics = self._get_or_init_ragas_metrics()
        context_recall, faithfulness, context_precision = await asyncio.gather(
            *(metric.single_turn_ascore(sample) for metric in metrics)
        )
        return {
            "context_recall": _clamp_score(context_recall),
            "faithfulness": _clamp_score(faithfulness),
            "context_precision": _clamp_score(context_precision),
        }

    async def _score_with_retry(
        self,
        turn_sample: SingleTurnSample,
        pipeline_version: PipelineVersion,
    ) -> dict[str, float]:
        max_retries = max(0, int(self._run_config.max_retries))
        max_wait = max(0.0, float(self._run_config.max_wait))

        async with self._semaphore:
            for attempt in range(max_retries + 1):
                try:
                    if self._custom_scorer is not None:
                        raw_scores = await self._custom_scorer(turn_sample, pipeline_version)
                    elif self._offline_mode:
                        raw_scores = await heuristic_ragas_scorer(turn_sample, pipeline_version)
                    else:
                        raw_scores = await self._score_with_ragas_oop(turn_sample)

                    return {
                        "context_recall": _clamp_score(raw_scores.get("context_recall")),
                        "faithfulness": _clamp_score(raw_scores.get("faithfulness")),
                        "context_precision": _clamp_score(raw_scores.get("context_precision")),
                    }
                except Exception as exc:
                    is_last = attempt >= max_retries
                    logger.warning(
                        "Ragas scoring attempt failed",
                        pipeline_version=pipeline_version,
                        attempt=attempt + 1,
                        max_attempts=max_retries + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if is_last:
                        logger.error(
                            "Ragas scoring retries exhausted; recording 0.0 fallback scores",
                            pipeline_version=pipeline_version,
                        )
                        return {
                            "context_recall": 0.0,
                            "faithfulness": 0.0,
                            "context_precision": 0.0,
                        }
                    delay = min(self._retry_base_delay * (2**attempt), max_wait)
                    if delay > 0:
                        await asyncio.sleep(delay)

        return {"context_recall": 0.0, "faithfulness": 0.0, "context_precision": 0.0}

    async def evaluate_sample(
        self,
        sample: EvalSample,
        pipeline_version: PipelineVersion,
    ) -> SampleEvalRecord:
        """Execute one sample on ``pipeline_version`` and score with Ragas."""
        run_output = await self._execute_runner(sample, pipeline_version)
        answer = str(run_output.get("answer") or "")
        retrieved_contexts = [str(ctx) for ctx in (run_output.get("retrieved_contexts") or [])]

        turn_sample = SingleTurnSample(
            user_input=sample.question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=sample.ground_truth,
            reference_contexts=list(sample.reference_contexts),
        )
        scores = await self._score_with_retry(turn_sample, pipeline_version)

        return SampleEvalRecord(
            sample_id=sample.sample_id,
            pipeline_version=pipeline_version,
            hop_type=sample.hop_type,
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            reference_contexts=list(sample.reference_contexts),
            context_recall=scores["context_recall"],
            faithfulness=scores["faithfulness"],
            context_precision=scores["context_precision"],
            latency_ms=max(0.0, float(run_output.get("latency_ms", 0.0))),
            ttft_ms=max(0.0, float(run_output.get("ttft_ms", 0.0))),
            rewrite_count=max(0, int(run_output.get("rewrite_count", 0))),
            cache_hit=bool(run_output.get("cache_hit", False)),
        )

    async def evaluate_pipeline(
        self,
        samples: Sequence[EvalSample],
        pipeline_version: PipelineVersion,
    ) -> tuple[PipelineEvalSummary, list[SampleEvalRecord]]:
        """Evaluate all samples on ``pipeline_version`` and compute summary stats."""
        records = list(
            await asyncio.gather(
                *(self.evaluate_sample(sample, pipeline_version) for sample in samples)
            )
        )
        summary = _aggregate_pipeline_summary(pipeline_version, records)
        return summary, records

    async def compare_pipelines(
        self,
        samples: Sequence[EvalSample],
        versions: Sequence[PipelineVersion] = PIPELINE_VERSIONS,
    ) -> RagasComparisonReport:
        """Evaluate and compare multiple pipeline versions on the same dataset."""
        summaries: dict[PipelineVersion, PipelineEvalSummary] = {}
        all_records: list[SampleEvalRecord] = []

        for version in versions:
            summary, records = await self.evaluate_pipeline(samples, version)
            summaries[version] = summary
            all_records.extend(records)

        logger.info(
            "Completed Ragas pipeline comparison",
            versions=list(versions),
            sample_count=len(samples),
            total_records=len(all_records),
        )
        return RagasComparisonReport(
            judge_model="offline-heuristic" if self._offline_mode else self._judge_model_name,
            summaries=summaries,
            records=all_records,
        )


# ---------------------------------------------------------------------------
# Persistence & CLI Entrypoint
# ---------------------------------------------------------------------------


def save_comparison_report(
    report: RagasComparisonReport,
    path: Path | str = DEFAULT_RESULTS_PATH,
) -> Path:
    """Persist a ``RagasComparisonReport`` as formatted UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Saved Ragas comparison report", path=str(target), records=len(report.records))
    return target


def load_comparison_report(
    path: Path | str = DEFAULT_RESULTS_PATH,
) -> RagasComparisonReport:
    """Load and validate a persisted ``RagasComparisonReport`` JSON file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Ragas comparison report not found: {source}")
    try:
        return RagasComparisonReport.model_validate_json(source.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"Invalid RagasComparisonReport in {source}: {exc}") from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Ragas v0.2+ multi-pipeline comparison evaluation across v1/v2/v3."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Input JSONL evaluation dataset path (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Output JSON comparison report path (default: {DEFAULT_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=PIPELINE_VERSIONS,
        default=list(PIPELINE_VERSIONS),
        help="Pipeline versions to evaluate (default: v1 v2 v3).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrent Ragas Judge evaluations (default: 2).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run deterministic offline simulation and heuristic Ragas scoring.",
    )
    return parser


async def _async_cli(args: argparse.Namespace) -> int:
    samples = await asyncio.to_thread(load_dataset_jsonl, args.dataset)
    if not samples:
        logger.warning("Evaluation dataset is empty", dataset=str(args.dataset))
        return 1

    evaluator = RagasPipelineEvaluator(
        max_concurrency=args.concurrency,
        offline_mode=bool(args.offline),
    )
    report = await evaluator.compare_pipelines(samples, versions=tuple(args.versions))
    await asyncio.to_thread(save_comparison_report, report, args.output)
    print(report.to_markdown_table())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m evals.run_ragas``."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_async_cli(args))


__all__: Sequence[str] = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_RESULTS_PATH",
    "PIPELINE_VERSIONS",
    "PipelineEvalSummary",
    "PipelineRunner",
    "PipelineVersion",
    "RagasComparisonReport",
    "RagasPipelineEvaluator",
    "SampleEvalRecord",
    "SampleScorer",
    "build_evaluation_dataset",
    "build_judge_llm",
    "build_ragas_metrics",
    "build_ragas_run_config",
    "heuristic_ragas_scorer",
    "load_comparison_report",
    "main",
    "save_comparison_report",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
