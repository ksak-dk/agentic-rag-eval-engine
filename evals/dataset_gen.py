"""
evals/dataset_gen.py
====================
Synthetic single-hop and multi-hop RAG evaluation dataset generator.

Design & Hardware Contract
--------------------------
- Never loads any local LLM on CPU or CUDA. All LLM synthesis goes through
  ``LLMRouter`` cloud strong route (``gemini-2.5-flash``) via
  ``ainvoke_structured``.
- Enforces bounded async concurrency via ``asyncio.Semaphore(2)`` and
  exponential-backoff retries to respect Google AI Studio rate limits (429).
- Provides deterministic offline dataset construction (``build_offline_dataset``)
  and JSONL persistence (``./data/eval_dataset.jsonl``) for CI and unit tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.adapters.mock.eval_heuristics import (
    build_offline_sample_dicts,
    select_multi_hop_candidates,
    select_single_hop_candidates,
)
from app.core.config import Settings, get_settings
from app.core.router import LLMRouter, RouteTier, TaskType
from app.retrieval.chunker import DocumentChunk, make_chunker

logger = structlog.get_logger(__name__)

HopType = Literal["single_hop", "multi_hop"]

DEFAULT_DATASET_PATH: Path = Path("./data/eval_dataset.jsonl")
DEFAULT_CONCURRENCY: int = 2
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_BASE_DELAY: float = 0.75

_INVALID_PLACEHOLDERS: frozenset[str] = frozenset(
    {"n/a", "na", "none", "null", "unknown", "tbd", "todo", "error"}
)


# ---------------------------------------------------------------------------
# Pydantic V2 Schemas
# ---------------------------------------------------------------------------


class EvalSample(BaseModel):
    """Single validated evaluation sample persisted in JSONL and consumed by Ragas."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the evaluation sample.",
    )
    question: str = Field(..., min_length=1, description="Evaluation question text.")
    ground_truth: str = Field(
        ...,
        min_length=1,
        description="Reference answer strictly grounded in reference_contexts.",
    )
    reference_contexts: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of ground-truth context passages.",
    )
    hop_type: HopType = Field(
        ...,
        description="Reasoning complexity category: 'single_hop' or 'multi_hop'.",
    )
    source_doc_ids: list[str] = Field(
        default_factory=list,
        description="Deduplicated document IDs from which reference_contexts originate.",
    )

    @field_validator("question", "ground_truth")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty or whitespace-only.")
        return cleaned

    @field_validator("reference_contexts")
    @classmethod
    def _validate_contexts(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("reference_contexts must contain at least one non-empty passage.")
        return cleaned

    @field_validator("source_doc_ids")
    @classmethod
    def _dedupe_doc_ids(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in value:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped


class GeneratedQAPair(BaseModel):
    """Structured LLM output schema returned by ``LLMRouter.ainvoke_structured``."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., description="Synthesized evaluation question.")
    ground_truth: str = Field(..., description="Concise reference answer grounded in context.")
    is_valid: bool = Field(
        default=True,
        description="False when the passages lack coherent facts for a reliable question.",
    )


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _try_get_settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


def _is_valid_qa_pair(pair: GeneratedQAPair | None) -> bool:
    if pair is None or not pair.is_valid:
        return False
    question = pair.question.strip()
    ground_truth = pair.ground_truth.strip()
    if len(question) < 5 or len(ground_truth) < 3:
        return False
    if question.lower() in _INVALID_PLACEHOLDERS:
        return False
    return ground_truth.lower() not in _INVALID_PLACEHOLDERS


def _format_chunk_for_prompt(chunk: DocumentChunk, index: int) -> str:
    section = " > ".join(chunk.metadata.section_path) or "root"
    return (
        f"[Passage {index + 1} | doc_id={chunk.metadata.doc_id} | "
        f"source={chunk.metadata.source} | section={section}]\n{chunk.text}"
    )


def _build_single_hop_messages(chunk: DocumentChunk) -> list[tuple[str, str]]:
    passage = _format_chunk_for_prompt(chunk, 0)
    return [
        (
            "system",
            "You are a RAG benchmark dataset author. Given one technical passage, generate ONE "
            "factual, self-contained single-hop question and a concise ground-truth answer "
            "strictly supported by that passage. If the passage lacks concrete technical facts, "
            "set is_valid=false.",
        ),
        ("user", f"Passage:\n{passage}"),
    ]


def _build_multi_hop_messages(chunks: Sequence[DocumentChunk]) -> list[tuple[str, str]]:
    passages = "\n\n---\n\n".join(
        _format_chunk_for_prompt(chunk, idx) for idx, chunk in enumerate(chunks)
    )
    return [
        (
            "system",
            "You are a RAG benchmark dataset author. Given multiple distinct technical passages, "
            "generate ONE multi-hop reasoning question whose answer MUST synthesize facts across "
            "at least two of the passages, together with a concise ground-truth answer strictly "
            "supported by those passages. Do not reference passage numbers directly (e.g. avoid "
            "'in Passage 1'). If the passages cannot support a meaningful multi-hop question, "
            "set is_valid=false.",
        ),
        ("user", f"Passages:\n{passages}"),
    ]


_select_single_hop_candidates = select_single_hop_candidates
_select_multi_hop_candidates = select_multi_hop_candidates


# ---------------------------------------------------------------------------
# JSONL Persistence & Offline Builders
# ---------------------------------------------------------------------------


def save_dataset_jsonl(
    samples: Sequence[EvalSample],
    path: Path | str = DEFAULT_DATASET_PATH,
) -> Path:
    """Serialize ``EvalSample`` records to a UTF-8 JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [sample.model_dump_json() for sample in samples]
    content = "\n".join(lines) + ("\n" if lines else "")
    target.write_text(content, encoding="utf-8")
    logger.info("Saved evaluation dataset", path=str(target), sample_count=len(samples))
    return target


def load_dataset_jsonl(path: Path | str = DEFAULT_DATASET_PATH) -> list[EvalSample]:
    """Load and validate ``EvalSample`` records from a UTF-8 JSONL file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {source}")

    samples: list[EvalSample] = []
    for line_no, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
            samples.append(EvalSample.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Invalid EvalSample at {source}:{line_no}: {exc}") from exc
    return samples


async def asave_dataset_jsonl(
    samples: Sequence[EvalSample],
    path: Path | str = DEFAULT_DATASET_PATH,
) -> Path:
    """Async wrapper over ``save_dataset_jsonl`` using a worker thread."""
    return await asyncio.to_thread(save_dataset_jsonl, samples, path)


async def aload_dataset_jsonl(path: Path | str = DEFAULT_DATASET_PATH) -> list[EvalSample]:
    """Async wrapper over ``load_dataset_jsonl`` using a worker thread."""
    return await asyncio.to_thread(load_dataset_jsonl, path)


def build_offline_dataset(
    chunks: Sequence[DocumentChunk],
    *,
    single_hop_count: int = 2,
    multi_hop_count: int = 2,
    seed: int = 42,
) -> list[EvalSample]:
    """Deterministically construct offline ``EvalSample`` items without any LLM call."""
    raw_samples = build_offline_sample_dicts(
        chunks,
        single_hop_count=single_hop_count,
        multi_hop_count=multi_hop_count,
        seed=seed,
    )
    return [EvalSample.model_validate(item) for item in raw_samples]


def load_chunks_from_paths(
    paths: Sequence[Path | str],
    *,
    pipeline_version: str = "v3",
) -> list[DocumentChunk]:
    """Read Markdown or plain-text files and chunk them via ``make_chunker``."""
    chunker = make_chunker(pipeline_version)
    all_chunks: list[DocumentChunk] = []

    resolved_files: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path)
        if candidate.is_dir():
            resolved_files.extend(
                sorted(
                    p
                    for p in candidate.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".md", ".txt"}
                )
            )
        elif candidate.is_file():
            resolved_files.append(candidate)

    for file_path in resolved_files:
        text = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem
        chunks = chunker.chunk(text=text, source=str(file_path), doc_id=doc_id)
        all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Async Synthetic Dataset Generator
# ---------------------------------------------------------------------------


class SyntheticDatasetGenerator:
    """Async single-hop and multi-hop QA dataset synthesizer backed by ``LLMRouter``."""

    def __init__(
        self,
        router: Any | None = None,
        *,
        settings: Settings | None = None,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        seed: int = 42,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
        if retry_base_delay < 0.0:
            raise ValueError(f"retry_base_delay must be >= 0.0, got {retry_base_delay}.")

        resolved_settings = settings if settings is not None else _try_get_settings()
        if router is not None:
            self._router = router
        elif resolved_settings is not None:
            self._router = LLMRouter(settings=resolved_settings)
        else:
            raise ValueError("Either 'router' or valid 'settings' must be provided.")

        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._seed = seed

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    async def _invoke_with_retry(
        self,
        messages: list[tuple[str, str]],
        semaphore: asyncio.Semaphore,
        *,
        hop_type: HopType,
    ) -> GeneratedQAPair | None:
        async with semaphore:
            for attempt in range(self._max_retries + 1):
                try:
                    raw = await self._router.ainvoke_structured(
                        GeneratedQAPair,
                        messages,
                        task_type=TaskType.EVAL_JUDGE,
                        force_tier=RouteTier.STRONG,
                    )
                    pair = (
                        raw
                        if isinstance(raw, GeneratedQAPair)
                        else GeneratedQAPair.model_validate(raw)
                    )
                    if not _is_valid_qa_pair(pair):
                        logger.debug(
                            "Filtered invalid synthetic QA pair",
                            hop_type=hop_type,
                            is_valid=pair.is_valid,
                        )
                        return None
                    return pair
                except Exception as exc:
                    is_last = attempt >= self._max_retries
                    logger.warning(
                        "Synthetic QA generation attempt failed",
                        hop_type=hop_type,
                        attempt=attempt + 1,
                        max_attempts=self._max_retries + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if is_last:
                        return None
                    delay = self._retry_base_delay * (2**attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
        return None

    async def _generate_single_hop_sample(
        self,
        chunk: DocumentChunk,
        semaphore: asyncio.Semaphore,
    ) -> EvalSample | None:
        messages = _build_single_hop_messages(chunk)
        pair = await self._invoke_with_retry(messages, semaphore, hop_type="single_hop")
        if pair is None:
            return None
        return EvalSample(
            question=pair.question,
            ground_truth=pair.ground_truth,
            reference_contexts=[chunk.text.strip()],
            hop_type="single_hop",
            source_doc_ids=[chunk.metadata.doc_id],
        )

    async def _generate_multi_hop_sample(
        self,
        group: Sequence[DocumentChunk],
        semaphore: asyncio.Semaphore,
    ) -> EvalSample | None:
        messages = _build_multi_hop_messages(group)
        pair = await self._invoke_with_retry(messages, semaphore, hop_type="multi_hop")
        if pair is None:
            return None
        return EvalSample(
            question=pair.question,
            ground_truth=pair.ground_truth,
            reference_contexts=[chunk.text.strip() for chunk in group],
            hop_type="multi_hop",
            source_doc_ids=[chunk.metadata.doc_id for chunk in group],
        )

    async def generate(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        single_hop_count: int = 5,
        multi_hop_count: int = 5,
        chunks_per_multi_hop: int = 2,
    ) -> list[EvalSample]:
        """Generate deduplicated single-hop and multi-hop ``EvalSample`` records."""
        if not chunks or (single_hop_count <= 0 and multi_hop_count <= 0):
            return []

        rng = random.Random(self._seed)
        semaphore = asyncio.Semaphore(self._max_concurrency)

        single_candidates = _select_single_hop_candidates(chunks, single_hop_count, rng)
        multi_candidates = _select_multi_hop_candidates(
            chunks,
            multi_hop_count,
            rng,
            chunks_per_group=chunks_per_multi_hop,
        )

        tasks: list[Any] = [
            *(self._generate_single_hop_sample(chunk, semaphore) for chunk in single_candidates),
            *(self._generate_multi_hop_sample(group, semaphore) for group in multi_candidates),
        ]
        if not tasks:
            return []

        raw_results = await asyncio.gather(*tasks)

        seen_questions: set[str] = set()
        samples: list[EvalSample] = []
        for sample in raw_results:
            if sample is None:
                continue
            normalized_q = sample.question.strip().lower()
            if normalized_q in seen_questions:
                continue
            seen_questions.add(normalized_q)
            samples.append(sample)

        logger.info(
            "Generated synthetic evaluation dataset",
            total_samples=len(samples),
            single_hop=sum(1 for s in samples if s.hop_type == "single_hop"),
            multi_hop=sum(1 for s in samples if s.hop_type == "multi_hop"),
        )
        return samples

    async def generate_and_save(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        output_path: Path | str = DEFAULT_DATASET_PATH,
        single_hop_count: int = 5,
        multi_hop_count: int = 5,
        chunks_per_multi_hop: int = 2,
    ) -> tuple[list[EvalSample], Path]:
        """Generate samples and persist them directly to a JSONL file."""
        samples = await self.generate(
            chunks,
            single_hop_count=single_hop_count,
            multi_hop_count=multi_hop_count,
            chunks_per_multi_hop=chunks_per_multi_hop,
        )
        saved_path = await asave_dataset_jsonl(samples, output_path)
        return samples, saved_path


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate single-hop and multi-hop RAG evaluation datasets in JSONL format."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input Markdown/text files or directories to chunk and synthesize QA pairs from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Output JSONL file path (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--single-hop",
        type=int,
        default=5,
        help="Target number of single-hop samples (default: 5).",
    )
    parser.add_argument(
        "--multi-hop",
        type=int,
        default=5,
        help="Target number of multi-hop samples (default: 5).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic offline sample construction without calling cloud LLMs.",
    )
    return parser


async def _async_cli(args: argparse.Namespace) -> int:
    chunks = await asyncio.to_thread(load_chunks_from_paths, args.inputs)
    if not chunks:
        logger.warning("No valid document chunks found from inputs", inputs=args.inputs)
        return 1

    if args.offline:
        samples = build_offline_dataset(
            chunks,
            single_hop_count=args.single_hop,
            multi_hop_count=args.multi_hop,
        )
        await asave_dataset_jsonl(samples, args.output)
        return 0

    generator = SyntheticDatasetGenerator(max_concurrency=args.concurrency)
    await generator.generate_and_save(
        chunks,
        output_path=args.output,
        single_hop_count=args.single_hop,
        multi_hop_count=args.multi_hop,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m evals.dataset_gen``."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_async_cli(args))


__all__: Sequence[str] = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BASE_DELAY",
    "EvalSample",
    "GeneratedQAPair",
    "HopType",
    "SyntheticDatasetGenerator",
    "aload_dataset_jsonl",
    "asave_dataset_jsonl",
    "build_offline_dataset",
    "load_chunks_from_paths",
    "load_dataset_jsonl",
    "main",
    "save_dataset_jsonl",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
