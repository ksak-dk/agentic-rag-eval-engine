"""
app/core/logging.py
===================
Structured single-line JSON logging, async request metrics context via
``contextvars``, safe CUDA VRAM sampling, and stage latency tracking.

Hardware contract (RTX 3060 / 6 GB VRAM)
----------------------------------------
``sample_cuda_vram_mb()`` never raises if CUDA is unavailable (e.g. CI runners
or CPU-only environments) and returns ``0.0`` safely.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Sequence
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, TextIO

import structlog
from pydantic import BaseModel, ConfigDict, Field

try:
    import torch
except ImportError:  # pragma: no cover - defensive fallback for minimal envs
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Request Metrics Model & Async ContextVar Store
# ---------------------------------------------------------------------------


class RequestMetrics(BaseModel):
    """Per-request telemetry tracked across async tasks via ``contextvars``."""

    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    request_id: str = Field(default="", description="Unique trace identifier for the request.")
    session_id: str = Field(default="", description="Conversation or LangGraph thread ID.")
    route_tier: str = Field(default="", description="Selected LLM route tier (fast | strong).")
    model_name: str = Field(default="", description="Concrete LLM model identifier used.")
    cache_hit: bool = Field(default=False, description="Whether the Redis semantic cache hit.")
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time to first token in milliseconds.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="End-to-end or active stage latency in milliseconds.",
    )
    prompt_tokens: int = Field(default=0, ge=0, description="Input prompt token count.")
    completion_tokens: int = Field(default=0, ge=0, description="Generated completion token count.")
    vram_allocated_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampled CUDA allocated memory in megabytes.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of LangGraph grader/rewriter or fallback retries.",
    )
    generation_mode: str = Field(
        default="",
        description="Generation mode: live_llm | demo_synthesis | fallback.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a plain JSON-ready dictionary (Pydantic V2)."""
        return self.model_dump()


_REQUEST_METRICS_VAR: ContextVar[RequestMetrics | None] = ContextVar(
    "request_metrics",
    default=None,
)
_ACTIVE_PIPELINE_VERSION_VAR: ContextVar[str] = ContextVar(
    "active_pipeline_version",
    default="",
)

# Standard LogRecord attributes excluded when promoting extra record attributes.
_RESERVED_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "extra_fields",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def sample_cuda_vram_mb() -> float:
    """Safely sample currently allocated CUDA VRAM in megabytes.

    Returns ``0.0`` when CUDA is unavailable or if PyTorch CUDA inspection
    raises an unexpected runtime error.
    """
    if torch is None:
        return 0.0
    try:
        if not torch.cuda.is_available():
            return 0.0
        allocated_bytes = float(torch.cuda.memory_allocated())
        return round(allocated_bytes / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


def get_request_metrics() -> RequestMetrics:
    """Return a validated copy of the active task's ``RequestMetrics``."""
    current = _REQUEST_METRICS_VAR.get()
    if current is None:
        return RequestMetrics()
    return current.model_copy()


def set_request_metrics(
    metrics: RequestMetrics | None = None,
    **kwargs: Any,
) -> Token[RequestMetrics | None]:
    """Bind a ``RequestMetrics`` instance to the current ``contextvars`` context."""
    if metrics is None:
        bound = RequestMetrics.model_validate(kwargs)
    elif kwargs:
        merged = {**metrics.model_dump(), **kwargs}
        bound = RequestMetrics.model_validate(merged)
    else:
        bound = metrics.model_copy()
    return _REQUEST_METRICS_VAR.set(bound)


def update_request_metrics(**kwargs: Any) -> RequestMetrics:
    """Update fields on the active ``RequestMetrics`` context and return it.

    Mutates the bound ``RequestMetrics`` instance in place when already scoped so
    updates from child ``asyncio.Task`` nodes (e.g. LangGraph Pregel) remain
    visible to the parent request context without cross-request leakage.
    """
    current = _REQUEST_METRICS_VAR.get()
    if current is None:
        updated = RequestMetrics.model_validate(kwargs)
        _REQUEST_METRICS_VAR.set(updated)
        return updated.model_copy()

    updated = RequestMetrics.model_validate({**current.model_dump(), **kwargs})
    for field_name in RequestMetrics.model_fields:
        setattr(current, field_name, getattr(updated, field_name))
    return current.model_copy()


def reset_request_metrics(token: Token[RequestMetrics | None] | None = None) -> None:
    """Reset the active ``RequestMetrics`` context via token or clear to ``None``."""
    if token is not None:
        _REQUEST_METRICS_VAR.reset(token)
    else:
        _REQUEST_METRICS_VAR.set(None)


def get_active_pipeline_version() -> str:
    """Return the active pipeline version bound to the current async context, if any."""
    return _ACTIVE_PIPELINE_VERSION_VAR.get()


def set_active_pipeline_version(pipeline_version: str) -> Token[str]:
    """Bind the active pipeline version (v1/v2/v3) into the current async context."""
    return _ACTIVE_PIPELINE_VERSION_VAR.set(pipeline_version.strip().lower())


def reset_active_pipeline_version(token: Token[str] | None = None) -> None:
    """Reset the active pipeline version context."""
    if token is not None:
        _ACTIVE_PIPELINE_VERSION_VAR.reset(token)
    else:
        _ACTIVE_PIPELINE_VERSION_VAR.set("")


class RequestMetricsContext:
    """Dual sync/async context manager scoping per-request ``RequestMetrics``."""

    def __init__(self, **initial_fields: Any) -> None:
        self._pipeline_version: str | None = initial_fields.pop("pipeline_version", None)
        self._initial_fields = initial_fields
        self._token: Token[RequestMetrics | None] | None = None
        self._pipeline_token: Token[str] | None = None
        self._metrics: RequestMetrics | None = None

    def _enter_scope(self) -> RequestMetrics:
        fields = dict(self._initial_fields)
        if not fields.get("request_id"):
            fields["request_id"] = str(uuid.uuid4())
        if "vram_allocated_mb" not in fields:
            fields["vram_allocated_mb"] = sample_cuda_vram_mb()
        metrics = RequestMetrics.model_validate(fields)
        self._token = _REQUEST_METRICS_VAR.set(metrics)
        if self._pipeline_version:
            self._pipeline_token = _ACTIVE_PIPELINE_VERSION_VAR.set(
                str(self._pipeline_version).strip().lower()
            )
        self._metrics = metrics
        return metrics.model_copy()

    def _exit_scope(self) -> None:
        if self._pipeline_token is not None:
            _ACTIVE_PIPELINE_VERSION_VAR.reset(self._pipeline_token)
            self._pipeline_token = None
        if self._token is not None:
            _REQUEST_METRICS_VAR.reset(self._token)
            self._token = None

    def __enter__(self) -> RequestMetrics:
        return self._enter_scope()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._exit_scope()
        return False

    async def __aenter__(self) -> RequestMetrics:
        return self._enter_scope()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._exit_scope()
        return False


def request_metrics_context(**kwargs: Any) -> RequestMetricsContext:
    """Create a scoped ``RequestMetrics`` context for sync or async execution."""
    return RequestMetricsContext(**kwargs)


# ---------------------------------------------------------------------------
# Single-Line JSON Formatter
# ---------------------------------------------------------------------------


class StructuredJsonFormatter(logging.Formatter):
    """Formats every ``logging.LogRecord`` as a single-line JSON object.

    Automatically merges all 11 ``RequestMetrics`` fields from ``contextvars``
    alongside standard record metadata and any structured extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        metrics = get_request_metrics().to_dict()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **metrics,
        }

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            for key, value in extra_fields.items():
                if key not in payload:
                    payload[key] = value

        for key, value in record.__dict__.items():
            if (
                key not in _RESERVED_LOG_RECORD_ATTRS
                and not key.startswith("_")
                and key not in payload
            ):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Stage Latency Context Manager
# ---------------------------------------------------------------------------


class StageLatencyTracker:
    """Dual sync/async context manager that measures stage duration and VRAM."""

    def __init__(
        self,
        stage: str,
        *,
        logger_name: str = "app.stage",
        level: int = logging.INFO,
        update_total_latency: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._stage = stage
        self._logger = logging.getLogger(logger_name)
        self._level = level
        self._update_total_latency = update_total_latency
        self._extra = extra or {}
        self._start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def _start(self) -> StageLatencyTracker:
        self._start_time = time.perf_counter()
        return self

    def _finish(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
    ) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        vram_mb = sample_cuda_vram_mb()

        current = get_request_metrics()
        metric_updates: dict[str, Any] = {"vram_allocated_mb": vram_mb}
        if self._update_total_latency or current.latency_ms is None:
            metric_updates["latency_ms"] = self.elapsed_ms
        update_request_metrics(**metric_updates)

        log_fields: dict[str, Any] = {
            **self._extra,
            "stage": self._stage,
            "stage_latency_ms": self.elapsed_ms,
            "vram_allocated_mb": vram_mb,
            "status": "error" if exc_type is not None else "ok",
        }
        if exc_type is not None and exc_val is not None:
            log_fields["error_type"] = exc_type.__name__
            log_fields["error_message"] = str(exc_val)
            self._logger.log(
                logging.ERROR,
                "Stage failed: %s",
                self._stage,
                extra={"extra_fields": log_fields},
            )
        else:
            self._logger.log(
                self._level,
                "Stage completed: %s",
                self._stage,
                extra={"extra_fields": log_fields},
            )

    def __enter__(self) -> StageLatencyTracker:
        return self._start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._finish(exc_type, exc_val)
        return False

    async def __aenter__(self) -> StageLatencyTracker:
        return self._start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._finish(exc_type, exc_val)
        return False


def log_stage_latency(
    stage: str,
    *,
    logger_name: str = "app.stage",
    level: int = logging.INFO,
    update_total_latency: bool = False,
    **extra: Any,
) -> StageLatencyTracker:
    """Return a sync/async context manager that logs stage latency and CUDA VRAM."""
    return StageLatencyTracker(
        stage,
        logger_name=logger_name,
        level=level,
        update_total_latency=update_total_latency,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Logger Configuration & Structlog Bridge
# ---------------------------------------------------------------------------


def _resolve_log_level(level: str | int | None) -> int:
    """Resolve log level cleanly without forcing full Settings validation at import time."""
    if isinstance(level, int):
        return level
    raw_level = level
    if raw_level is None:
        raw_level = os.getenv("LOG_LEVEL", "INFO")
    normalized = raw_level.strip().upper()
    resolved = logging.getLevelName(normalized)
    return resolved if isinstance(resolved, int) else logging.INFO


def _structlog_to_stdlib_processor(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Route structlog event dicts into stdlib LogRecord ``extra_fields``."""
    event = str(event_dict.pop("event", ""))
    return (event,), {"extra": {"extra_fields": event_dict}}


def configure_logging(
    level: str | int | None = None,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure root stdlib logging and structlog to emit single-line JSON."""
    numeric_level = _resolve_log_level(level)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _structlog_to_stdlib_processor,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    return root_logger


def get_logger(name: str = "app") -> logging.Logger:
    """Return a standard logger configured to propagate to the JSON handler."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__: Sequence[str] = [
    "RequestMetrics",
    "RequestMetricsContext",
    "StageLatencyTracker",
    "StructuredJsonFormatter",
    "configure_logging",
    "get_active_pipeline_version",
    "get_logger",
    "get_request_metrics",
    "log_stage_latency",
    "request_metrics_context",
    "reset_active_pipeline_version",
    "reset_request_metrics",
    "sample_cuda_vram_mb",
    "set_active_pipeline_version",
    "set_request_metrics",
    "update_request_metrics",
]
