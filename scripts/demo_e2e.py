"""
scripts/demo_e2e.py
===================
One-click end-to-end streaming demo script for ``agentic-rag-eval-engine``.

Modes
-----
- ``--mode asgi`` (default, zero external dependencies):
  Spins up an in-memory FastAPI ASGI instance with ``MOCK_LLM_STREAM`` stubs
  and a temporary ``AsyncSqliteSaver`` checkpoint DB, sequentially streaming
  ``v1``, ``v2``, ``v3`` (cold cache miss + cache write), and a repeat ``v3``
  query triggering a **Redis 8 vector semantic cache hit**. Formats dual-channel
  SSE events (``[TOKEN]``, ``[CITATIONS]``, ``[METRICS]``, ``[DONE]``) to the
  terminal and saves the complete transcript to ``./data/e2e_demo_transcript.md``.

- ``--mode live`` (live Uvicorn + Docker integration mode):
  Probes ``GET /api/v1/health`` on ``--url`` (default ``http://127.0.0.1:8000``),
  streams real SSE responses from ``POST /api/v1/chat``, saves the Markdown
  transcript, and prints instructions for opening ``tests/sse_viewer.html``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import LogLevel, Settings
from app.core.config import PipelineVersion as ConfigPipelineVersion
from app.main import create_app, lifespan

logger = structlog.get_logger(__name__)

DemoMode = Literal["asgi", "live"]
PipelineLiteral = Literal["v1", "v2", "v3"]

DEFAULT_LIVE_URL: str = "http://127.0.0.1:8000"
DEFAULT_TRANSCRIPT_PATH: Path = Path("./data/e2e_demo_transcript.md")
SSE_VIEWER_PATH: Path = Path("./tests/sse_viewer.html")


# ---------------------------------------------------------------------------
# Pydantic V2 Models
# ---------------------------------------------------------------------------


class DemoScenario(BaseModel):
    """Single end-to-end demonstration step executed against ``/api/v1/chat``."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    pipeline_version: PipelineLiteral
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    expect_cache_hit: bool = False


class HealthSnapshot(BaseModel):
    """Parsed response from ``GET /api/v1/health``."""

    model_config = ConfigDict(frozen=True)

    status: str
    pipeline_version: str
    cuda_available: bool = False
    vram_allocated_mb: float = 0.0
    components: dict[str, bool | str] = Field(default_factory=dict)


class DemoRunResult(BaseModel):
    """Captured dual-channel SSE output and telemetry for one demo scenario."""

    model_config = ConfigDict(frozen=True)

    scenario: DemoScenario
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    done_payload: dict[str, Any] = Field(default_factory=dict)
    event_counts: dict[str, int] = Field(default_factory=dict)
    client_elapsed_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None


def default_demo_scenarios() -> list[DemoScenario]:
    """Return the canonical 4-step showcase across v1, v2, v3, and cache hit."""
    repeat_query = (
        "How does the v3 LangGraph pipeline prevent infinite loops and speed up repeat queries?"
    )
    return [
        DemoScenario(
            step=1,
            title="v1 Baseline — Fixed Chunking + Qdrant Dense-Only",
            pipeline_version="v1",
            session_id="demo-v1-baseline",
            query="How does Qdrant store dense vectors for fast baseline retrieval?",
        ),
        DemoScenario(
            step=2,
            title="v2 Hybrid + CUDA Rerank — Dense/BM25 RRF + FP16 bge-reranker-v2-m3",
            pipeline_version="v2",
            session_id="demo-v2-hybrid",
            query="How do Dense bge-m3 and BM25 Sparse vectors combine with CUDA reranking?",
        ),
        DemoScenario(
            step=3,
            title="v3 Agentic Loop (Cold Miss) — LangGraph Self-Correction + Cache Write",
            pipeline_version="v3",
            session_id="demo-v3-agentic",
            query=repeat_query,
        ),
        DemoScenario(
            step=4,
            title="v3 Repeat Query (Semantic Cache Hit) — Redis 8 Vector Short-Circuit",
            pipeline_version="v3",
            session_id="demo-v3-cache-hit",
            query=repeat_query,
            expect_cache_hit=True,
        ),
    ]


# ---------------------------------------------------------------------------
# In-Memory ASGI Demo App Builder (Powered by ServiceContainer demo mode)
# ---------------------------------------------------------------------------


def build_demo_asgi_app(
    *,
    checkpoint_db: Path,
    checkpointer: Any = None,
) -> FastAPI:
    """Create a self-contained offline FastAPI app powered by ``ServiceContainer`` demo mode."""
    settings = Settings(
        runtime_mode="demo",
        pipeline_version=ConfigPipelineVersion.V3,
        mock_llm_stream=True,
        langgraph_checkpoint_db=checkpoint_db,
        log_level=LogLevel.WARNING,
    )
    return create_app(
        settings=settings,
        checkpointer=checkpointer,
        checkpoint_db=checkpoint_db,
    )


# ---------------------------------------------------------------------------
# SSE Parsing & Terminal Rendering
# ---------------------------------------------------------------------------


def parse_sse_frame(frame: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a single raw SSE frame into ``(event_name, json_payload)``."""
    if not frame.strip():
        return None
    event_name = "message"
    data_lines: list[str] = []
    for line in frame.split("\n"):
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    raw_data = "\n".join(data_lines)
    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        payload = {"raw": raw_data}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return event_name, payload


def format_metrics_line(metrics: dict[str, Any]) -> str:
    """Format key telemetry metrics into a concise single-line terminal banner."""
    ttft = metrics.get("ttft_ms")
    latency = metrics.get("latency_ms")
    cache_hit = bool(metrics.get("cache_hit", False))
    vram_mb = float(metrics.get("vram_allocated_mb") or 0.0)
    route_tier = metrics.get("route_tier") or "-"
    model_name = metrics.get("model_name") or "-"
    retry_count = int(metrics.get("retry_count") or 0)
    ttft_str = f"{float(ttft):.2f}ms" if ttft is not None else "-"
    lat_str = f"{float(latency):.2f}ms" if latency is not None else "-"
    return (
        f"[METRICS: ttft_ms={ttft_str} / latency_ms={lat_str} / "
        f"cache_hit={cache_hit} / vram_mb={vram_mb:.2f}] "
        f"(route={route_tier}, model={model_name}, retries={retry_count})"
    )


async def probe_health(client: httpx.AsyncClient) -> HealthSnapshot:
    """Query ``GET /api/v1/health`` and validate the returned readiness payload."""
    response = await client.get("/api/v1/health")
    response.raise_for_status()
    return HealthSnapshot.model_validate(response.json())


async def stream_scenario(
    client: httpx.AsyncClient,
    scenario: DemoScenario,
    *,
    quiet: bool = False,
) -> DemoRunResult:
    """Execute one streaming ``POST /api/v1/chat`` scenario and print live SSE output."""
    if not quiet:
        print(f"\n{'=' * 78}")
        print(f"Step {scenario.step}: {scenario.title}")
        print(f"Pipeline: {scenario.pipeline_version} | Session: {scenario.session_id}")
        print(f"Query   : {scenario.query}")
        print(f"{'-' * 78}")
        print("[TOKEN] ", end="", flush=True)

    body = {
        "query": scenario.query,
        "session_id": scenario.session_id,
        "pipeline_version": scenario.pipeline_version,
        "stream": True,
    }

    tokens: list[str] = []
    citations: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    done_payload: dict[str, Any] = {}
    event_counts: dict[str, int] = {}
    error_message: str | None = None

    started = time.perf_counter()
    async with client.stream(
        "POST",
        "/api/v1/chat",
        json=body,
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code != 200:
            raw_err = (await response.aread()).decode("utf-8", errors="replace")
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if not quiet:
                print(f"\n[ERROR] HTTP {response.status_code}: {raw_err}")
            return DemoRunResult(
                scenario=scenario,
                client_elapsed_ms=elapsed_ms,
                error=f"HTTP {response.status_code}: {raw_err}",
            )

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk.replace("\r\n", "\n")
            frames = buffer.split("\n\n")
            buffer = frames.pop()

            for frame in frames:
                parsed = parse_sse_frame(frame)
                if parsed is None:
                    continue
                event_name, payload = parsed
                event_counts[event_name] = event_counts.get(event_name, 0) + 1

                if event_name == "token":
                    piece = str(payload.get("token", ""))
                    tokens.append(piece)
                    if not quiet:
                        print(piece, end="", flush=True)
                elif event_name == "citations":
                    raw_citations = payload.get("citations")
                    if isinstance(raw_citations, list):
                        citations = [c for c in raw_citations if isinstance(c, dict)]
                elif event_name == "metrics":
                    metrics = payload
                elif event_name == "done":
                    done_payload = payload
                elif event_name == "error":
                    error_message = f"{payload.get('error_type', 'Error')}: {payload.get('error')}"

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    if not quiet:
        print()
        if citations:
            print(f"[CITATIONS] ({len(citations)} source(s))")
            for idx, item in enumerate(citations, start=1):
                sections = " > ".join(item.get("section_path") or []) or "root"
                score = float(item.get("score") or 0.0)
                print(
                    f"  {idx}. [{item.get('doc_id', 'unknown')}] "
                    f"{item.get('source', 'unknown')} ({sections}) | score={score:.4f}"
                )
        else:
            print("[CITATIONS] None")

        print(format_metrics_line(metrics))
        status_label = done_payload.get("status", "completed" if error_message is None else "error")
        print(
            f"[DONE] status={status_label} | pipeline={scenario.pipeline_version} | "
            f"client_elapsed_ms={elapsed_ms:.2f}"
        )
        if error_message:
            print(f"[ERROR] {error_message}")

    return DemoRunResult(
        scenario=scenario,
        answer="".join(tokens).strip(),
        citations=citations,
        metrics=metrics,
        done_payload=done_payload,
        event_counts=event_counts,
        client_elapsed_ms=elapsed_ms,
        error=error_message,
    )


# ---------------------------------------------------------------------------
# Transcript Rendering & Runner
# ---------------------------------------------------------------------------


def render_transcript_markdown(
    *,
    mode: DemoMode,
    target_url: str,
    health: HealthSnapshot,
    results: Sequence[DemoRunResult],
) -> str:
    """Render all demo runs and telemetry into a Markdown transcript report."""
    generated_at = datetime.now(UTC).isoformat()
    lines: list[str] = [
        "# End-to-End Streaming Demo Transcript",
        "",
        f"- **Generated At**: `{generated_at}`",
        f"- **Demo Mode**: `{mode}`",
        f"- **Target**: `{target_url}`",
        f"- **Health Status**: `{health.status}` (default pipeline: `{health.pipeline_version}`, "
        f"CUDA: `{health.cuda_available}`, VRAM: `{health.vram_allocated_mb:.2f} MB`)",
        "",
        "## 1. Execution Summary",
        "",
        "| Step | Pipeline | Cache Hit | TTFT (ms) | Server Latency (ms) | Client Elapsed (ms) | Citations | Status |",
        "| ---: | :---: | :---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    for res in results:
        m = res.metrics
        cache_hit = bool(m.get("cache_hit", False))
        ttft = m.get("ttft_ms")
        lat = m.get("latency_ms")
        ttft_str = f"{float(ttft):.2f}" if ttft is not None else "-"
        lat_str = f"{float(lat):.2f}" if lat is not None else "-"
        status_txt = "ERROR" if res.error else str(res.done_payload.get("status", "completed"))
        lines.append(
            f"| {res.scenario.step} | `{res.scenario.pipeline_version}` | "
            f"`{cache_hit}` | {ttft_str} | {lat_str} | {res.client_elapsed_ms:.2f} | "
            f"{len(res.citations)} | {status_txt} |"
        )

    lines.extend(["", "## 2. Detailed Dual-Channel SSE Transcript", ""])

    for res in results:
        lines.extend(
            [
                f"### Step {res.scenario.step}: {res.scenario.title}",
                "",
                f"- **Pipeline**: `{res.scenario.pipeline_version}` | **Session**: `{res.scenario.session_id}`",
                f"- **Query**: {res.scenario.query}",
                f"- **SSE Events**: `{json.dumps(res.event_counts, ensure_ascii=False)}`",
                "",
                "**`[TOKEN]` Streamed Answer:**",
                "",
                f"> {res.answer or '(empty)'}",
                "",
                "**`[CITATIONS]` Channel:**",
                "",
            ]
        )
        if res.citations:
            for item in res.citations:
                sections = " > ".join(item.get("section_path") or []) or "root"
                score = float(item.get("score") or 0.0)
                lines.append(
                    f"- `[{item.get('doc_id', 'unknown')}]` `{item.get('source', 'unknown')}` "
                    f"({sections}, score=`{score:.4f}`)"
                )
        else:
            lines.append("- None")

        lines.extend(
            [
                "",
                "**`[METRICS]` Channel:**",
                "",
                f"- `{format_metrics_line(res.metrics)}`",
                "",
            ]
        )
        if res.error:
            lines.append(f"- **Error**: `{res.error}`\n")

    return "\n".join(lines)


def save_transcript(markdown: str, output_path: Path = DEFAULT_TRANSCRIPT_PATH) -> Path:
    """Persist the Markdown demo transcript to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


@asynccontextmanager
async def _open_demo_client(mode: DemoMode, url: str) -> AsyncIterator[httpx.AsyncClient]:
    if mode == "asgi":
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "demo_checkpoints.sqlite3"
            demo_app = build_demo_asgi_app(checkpoint_db=db_path)
            async with lifespan(demo_app):
                transport = httpx.ASGITransport(app=demo_app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://asgi.demo",
                    timeout=30.0,
                ) as client:
                    yield client
        return

    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=30.0) as client:
        yield client


async def run_demo(
    *,
    mode: DemoMode = "asgi",
    url: str = DEFAULT_LIVE_URL,
    output_path: Path = DEFAULT_TRANSCRIPT_PATH,
    scenarios: Sequence[DemoScenario] | None = None,
    quiet: bool = False,
) -> tuple[HealthSnapshot, list[DemoRunResult], Path]:
    """Execute the end-to-end SSE demo in ``asgi`` or ``live`` mode."""
    os.environ.setdefault("MOCK_LLM_STREAM", "true" if mode == "asgi" else "false")
    active_scenarios = list(scenarios or default_demo_scenarios())
    target_label = "http://asgi.demo (in-memory ASGI)" if mode == "asgi" else url.rstrip("/")

    async with _open_demo_client(mode, url) as client:
        health = await probe_health(client)
        if not quiet:
            print("=" * 78)
            print("agentic-rag-eval-engine — End-to-End Dual-Channel SSE Demo")
            print(f"Mode   : {mode} | Target: {target_label}")
            print(
                f"Health : status={health.status} | default={health.pipeline_version} | "
                f"cuda={health.cuda_available} | vram={health.vram_allocated_mb:.2f} MB"
            )
            print(f"Modules: {json.dumps(health.components)}")
            print("=" * 78)

        results: list[DemoRunResult] = []
        for scenario in active_scenarios:
            res = await stream_scenario(client, scenario, quiet=quiet)
            results.append(res)

    markdown = render_transcript_markdown(
        mode=mode,
        target_url=target_label,
        health=health,
        results=results,
    )
    saved_path = save_transcript(markdown, output_path=output_path)

    if not quiet:
        print(f"\n{'=' * 78}")
        print(f"Transcript saved to: {saved_path}")
        if mode == "live":
            print(
                f"Visual SSE Dashboard: open {SSE_VIEWER_PATH.as_posix()} in your browser "
                f"and point it to {url.rstrip('/')}/api/v1/chat"
            )
        else:
            print(
                "Tip: To run with live Docker + Uvicorn + browser UI, start `docker compose up -d`, "
                "run `uv run uvicorn app.main:app --port 8000`, open `tests/sse_viewer.html`, "
                "or re-run with `--mode live`."
            )
        print("=" * 78)

    return health, results, saved_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end dual-channel SSE demo for agentic-rag-eval-engine.",
    )
    parser.add_argument(
        "--mode",
        choices=("asgi", "live"),
        default="asgi",
        help="Execution mode: 'asgi' (zero-dependency in-memory) or 'live' (HTTP server).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_LIVE_URL,
        help=f"Target base URL when --mode live is used (default: {DEFAULT_LIVE_URL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TRANSCRIPT_PATH,
        help=f"Markdown transcript output path (default: {DEFAULT_TRANSCRIPT_PATH}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live terminal streaming output (still saves transcript).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``scripts/demo_e2e.py``."""
    args = _parse_args(argv)
    try:
        _, results, _ = asyncio.run(
            run_demo(
                mode=args.mode,
                url=args.url,
                output_path=args.output,
                quiet=args.quiet,
            )
        )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        print(
            f"[LIVE CONNECTION ERROR] Could not reach {args.url}: {exc}\n"
            "Start infrastructure and API server first:\n"
            "  1. docker compose up -d\n"
            "  2. uv run uvicorn app.main:app --host 0.0.0.0 --port 8000\n"
            "  3. uv run python scripts/demo_e2e.py --mode live",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        logger.error("E2E demo execution failed", error_type=type(exc).__name__, error=str(exc))
        print(f"[DEMO FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if any(r.error is not None for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
