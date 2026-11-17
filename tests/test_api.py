from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from app.api.v1.chat import ChatServiceDependencies, get_chat_dependencies
from app.cache.semantic_cache import CachedResponse
from app.core.config import LogLevel, PipelineVersion, Settings
from app.core.logging import RequestMetrics
from app.core.router import LLMGenerationResult, RouteDecision, RouteTier, TaskType
from app.graph.nodes import DocumentGradeOutput, HallucinationGradeOutput
from app.main import create_app, lifespan
from app.retrieval.chunker import ChunkMetadata
from app.retrieval.hybrid_store import RetrievedChunk


def _make_test_settings(
    tmp_path: Path,
    *,
    pipeline_version: PipelineVersion = PipelineVersion.V3,
) -> Settings:
    return Settings(
        groq_api_key="test-groq-key",
        google_api_key="test-google-key",
        pipeline_version=pipeline_version,
        mock_llm_stream=True,
        langgraph_checkpoint_db=tmp_path / "checkpoints.sqlite3",
        log_level=LogLevel.WARNING,
    )


def _make_chunk(doc_id: str, text: str, *, score: float = 0.93, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            source=f"docs/{doc_id}.md",
            section_path=["Architecture", "Pipeline"],
            chunk_index=rank,
        ),
        score=score,
        rank=rank,
    )


class FakeRetriever:
    """Offline retriever stub recording pipeline_version and query arguments."""

    def __init__(self, chunks: list[RetrievedChunk], *, fail: bool = False) -> None:
        self._chunks = chunks
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        query: str,
        pipeline_version: str = "v3",
        top_k: int = 5,
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "pipeline_version": pipeline_version,
                "top_k": top_k,
                "top_n": top_n,
            }
        )
        if self._fail:
            raise RuntimeError("Simulated Qdrant retrieval outage")
        return list(self._chunks[:top_k])


class FakeSemanticCache:
    """In-memory async cache stub mimicking SemanticCache interface."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], CachedResponse] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str]] = []

    async def get(self, query: str, *, namespace: str = "v3") -> CachedResponse | None:
        self.get_calls.append((namespace, query))
        return self.entries.get((namespace, query))

    async def set(
        self,
        query: str,
        response: str,
        *,
        namespace: str = "v3",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.set_calls.append((namespace, query))
        self.entries[(namespace, query)] = CachedResponse(
            query=query,
            response=response,
            namespace=namespace,
            similarity=1.0,
            metadata=dict(metadata or {}),
        )
        return f"rag:cache:{namespace}:fake"


class FakeRouter:
    """Offline LLM router stub supporting select_route, ainvoke, astream, and structured calls."""

    def __init__(
        self,
        *,
        answer_text: str = "Qdrant INT8 hybrid search uses RRF fusion [doc-arch].",
        stream_tokens: list[str] | None = None,
    ) -> None:
        self.answer_text = answer_text
        self.stream_tokens = stream_tokens or [
            "Qdrant INT8 ",
            "hybrid search ",
            "uses RRF fusion [doc-arch].",
        ]
        self.invoke_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def select_route(
        self,
        *,
        task_type: TaskType | str | None = None,
        query: str | None = None,
        retry_count: int = 0,
        force_tier: RouteTier | str | None = None,
    ) -> RouteDecision:
        if force_tier == RouteTier.STRONG or retry_count > 0:
            return RouteDecision(
                tier=RouteTier.STRONG,
                model_name="gemini-2.5-flash",
                fallback_model_name=None,
                reason="Strong route selected",
            )
        return RouteDecision(
            tier=RouteTier.FAST,
            model_name="llama-3.1-8b-instant",
            fallback_model_name="gemini-2.5-flash-lite",
            reason="Fast route selected",
        )

    async def ainvoke(
        self,
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> LLMGenerationResult:
        tier = force_tier if isinstance(force_tier, RouteTier) else RouteTier.STRONG
        model_name = "llama-3.1-8b-instant" if tier == RouteTier.FAST else "gemini-2.5-flash"
        self.invoke_calls.append(
            {
                "task_type": str(task_type),
                "force_tier": tier.value,
                "retry_count": retry_count,
            }
        )
        return LLMGenerationResult(
            content=self.answer_text,
            route_tier=tier,
            model_name=model_name,
            mocked=True,
            ttft_ms=4.5,
            latency_ms=8.0,
            prompt_tokens=42,
            completion_tokens=16,
        )

    async def astream(
        self,
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        tier = force_tier if isinstance(force_tier, RouteTier) else RouteTier.STRONG
        self.stream_calls.append(
            {
                "task_type": str(task_type),
                "force_tier": tier.value,
                "retry_count": retry_count,
            }
        )
        for token in self.stream_tokens:
            yield token

    async def ainvoke_structured(
        self,
        schema: type[Any],
        messages: Any,
        *,
        task_type: TaskType | str | None = None,
        force_tier: RouteTier | str | None = None,
        retry_count: int = 0,
        temperature: float = 0.0,
    ) -> Any:
        if schema is DocumentGradeOutput:
            return DocumentGradeOutput(relevant=True, reason="Relevant chunk")
        if schema is HallucinationGradeOutput:
            return HallucinationGradeOutput(grounded=True, reason="Grounded in context")
        return schema.model_validate({})


def _parse_sse_events(raw_body: str) -> list[tuple[str, dict[str, Any]]]:
    normalized = raw_body.replace("\r\n", "\n")
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in normalized.split("\n\n"):
        stripped = frame.strip()
        if not stripped:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in stripped.split("\n"):
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


async def test_health_endpoint_returns_200(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, pipeline_version=PipelineVersion.V2)
    app = create_app(
        settings=settings,
        retriever=FakeRetriever([]),
        router=FakeRouter(),
        cache=FakeSemanticCache(),
        testing=True,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["pipeline_version"] == "v2"
    assert isinstance(payload["cuda_available"], bool)
    assert payload["vram_allocated_mb"] >= 0.0
    assert payload["components"] == {
        "retriever": True,
        "router": True,
        "cache": True,
        "checkpointer": False,
    }


async def test_chat_sync_v1_v2_v3_routing_and_response(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, pipeline_version=PipelineVersion.V3)
    chunks = [
        _make_chunk("doc-arch", "Qdrant INT8 hybrid search uses RRF fusion.", score=0.96, rank=0),
    ]
    retriever = FakeRetriever(chunks)
    router = FakeRouter()
    cache = FakeSemanticCache()

    app = create_app(
        settings=settings,
        retriever=retriever,
        router=router,
        cache=cache,
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        testing=True,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for version, expected_tier in [("v1", "fast"), ("v2", "strong"), ("v3", "strong")]:
            resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": f"Explain hybrid search ({version})",
                    "session_id": f"sess-{version}",
                    "pipeline_version": version,
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["pipeline_version"] == version
            assert body["session_id"] == f"sess-{version}"
            assert "[doc-arch]" in body["answer"]
            assert len(body["citations"]) == 1
            assert body["citations"][0]["doc_id"] == "doc-arch"

            metrics = body["metrics"]
            assert metrics["request_id"]
            assert metrics["session_id"] == f"sess-{version}"
            assert metrics["route_tier"] == expected_tier
            assert metrics["latency_ms"] is not None
            assert metrics["latency_ms"] >= 0.0
            assert metrics["cache_hit"] is False

        # Omitting pipeline_version falls back to Settings.pipeline_version ("v3")
        default_resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "Explain hybrid search (v3)",
                "session_id": "sess-v3-cached",
                "stream": False,
            },
        )
        assert default_resp.status_code == 200
        default_body = default_resp.json()
        assert default_body["pipeline_version"] == "v3"
        assert default_body["metrics"]["cache_hit"] is True

    assert [call["pipeline_version"] for call in retriever.calls] == ["v1", "v2", "v3"]
    assert (tmp_path / "checkpoints.sqlite3").exists()


async def test_chat_dependency_overrides_support(tmp_path: Path) -> None:
    app = create_app(testing=True)
    settings = _make_test_settings(tmp_path, pipeline_version=PipelineVersion.V2)
    chunks = [_make_chunk("doc-di", "Dependency override snippet.", score=0.91, rank=0)]
    fake_deps = ChatServiceDependencies(
        settings=settings,
        retriever=FakeRetriever(chunks),
        router=FakeRouter(answer_text="Answer using injected DI [doc-di]."),
        cache=FakeSemanticCache(),
        checkpoint_db=tmp_path / "di_checkpoints.sqlite3",
    )
    app.dependency_overrides[get_chat_dependencies] = lambda: fake_deps

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "Verify dependency overrides",
                "session_id": "sess-di",
                "stream": False,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["pipeline_version"] == "v2"
    assert body["answer"] == "Answer using injected DI [doc-di]."
    assert [c["doc_id"] for c in body["citations"]] == ["doc-di"]


async def test_chat_sse_stream_event_order_and_payloads(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, pipeline_version=PipelineVersion.V3)
    chunks = [_make_chunk("doc-arch", "RRF merges dense and sparse rankings.", score=0.95)]
    app = create_app(
        settings=settings,
        retriever=FakeRetriever(chunks),
        router=FakeRouter(
            answer_text="Hybrid search merges rankings [doc-arch].",
            stream_tokens=["Hybrid search ", "merges rankings", " [doc-arch]."],
        ),
        cache=FakeSemanticCache(),
        checkpoint_db=tmp_path / "sse_checkpoints.sqlite3",
        testing=True,
    )

    expected_metric_keys = set(RequestMetrics().to_dict().keys())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for version in ("v2", "v3"):
            async with client.stream(
                "POST",
                "/api/v1/chat",
                json={
                    "query": f"Stream SSE for {version}",
                    "session_id": f"sess-sse-{version}",
                    "pipeline_version": version,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                raw_body = (await response.aread()).decode("utf-8")

            parsed = _parse_sse_events(raw_body)
            event_names = [name for name, _ in parsed]
            assert event_names[-3:] == ["citations", "metrics", "done"]
            token_events = [data for name, data in parsed if name == "token"]
            assert len(token_events) >= 1
            assert all(name == "token" for name in event_names[:-3])

            full_answer = "".join(item["token"] for item in token_events)
            assert "[doc-arch]" in full_answer

            citations_payload = parsed[-3][1]
            assert citations_payload["session_id"] == f"sess-sse-{version}"
            assert len(citations_payload["citations"]) == 1
            assert citations_payload["citations"][0]["doc_id"] == "doc-arch"

            metrics_payload = parsed[-2][1]
            assert expected_metric_keys.issubset(metrics_payload.keys())
            assert metrics_payload["request_id"]
            assert metrics_payload["session_id"] == f"sess-sse-{version}"
            assert metrics_payload["latency_ms"] is not None

            done_payload = parsed[-1][1]
            assert done_payload == {
                "status": "completed",
                "session_id": f"sess-sse-{version}",
                "pipeline_version": version,
            }


async def test_chat_sse_stream_emits_error_event_on_exception(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, pipeline_version=PipelineVersion.V2)
    app = create_app(
        settings=settings,
        retriever=FakeRetriever([], fail=True),
        router=FakeRouter(),
        testing=True,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream(
            "POST",
            "/api/v1/chat",
            json={
                "query": "Trigger retrieval error",
                "session_id": "sess-err",
                "pipeline_version": "v2",
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            raw_body = (await response.aread()).decode("utf-8")

    parsed = _parse_sse_events(raw_body)
    assert len(parsed) == 1
    event_name, payload = parsed[0]
    assert event_name == "error"
    assert payload["error_type"] == "RuntimeError"
    assert "Simulated Qdrant retrieval outage" in payload["error"]
    assert payload["session_id"] == "sess-err"


async def test_lifespan_initializes_and_cleans_up_without_gpu(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path, pipeline_version=PipelineVersion.V3)
    app = create_app(
        settings=settings,
        retriever=FakeRetriever([_make_chunk("doc-life", "Lifespan test.")]),
        checkpoint_db=tmp_path / "lifespan.sqlite3",
        testing=False,
    )

    async with lifespan(app):
        assert app.state.router is not None
        assert app.state.checkpointer is not None
        assert app.state.workflow is not None
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/api/v1/health")
            assert health.status_code == 200
            assert health.json()["components"]["checkpointer"] is True
