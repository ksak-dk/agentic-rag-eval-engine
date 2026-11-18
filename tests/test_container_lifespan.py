from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.cache.semantic_cache import SingleFlight
from app.core.config import LogLevel, PipelineVersion, Settings, get_settings
from app.core.container import ServiceContainer, is_placeholder_api_key
from app.main import create_app, lifespan
from tests.test_api import FakeRetriever, FakeRouter, _make_chunk


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


def _clear_cloud_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "RUNTIME_MODE",
        "MOCK_LLM_STREAM",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(tmp_path / "auto_checkpoints.sqlite3"))
    get_settings.cache_clear()


async def test_zero_config_create_app_lifespan_health_and_viewer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Calling create_app() with zero arguments auto-bootstraps demo mode and health=ok."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()

    async with lifespan(app):
        assert app.state.container is not None
        assert app.state.workflow is not None
        assert app.state.retriever is not None
        assert app.state.router is not None
        assert app.state.semantic_cache is not None

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            health_resp = await client.get("/api/v1/health")
            root_resp = await client.get("/")
            viewer_resp = await client.get("/viewer")

    assert health_resp.status_code == 200
    payload = health_resp.json()
    assert payload["status"] == "ok"
    assert payload["pipeline_version"] == "v3"
    assert payload["components"]["runtime_mode"] == "demo"
    for key in ("retriever", "router", "semantic_cache", "workflow", "checkpointer"):
        assert payload["components"][key] == "ready (demo)"

    assert root_resp.status_code == 200
    assert "Agentic RAG Dual-Channel SSE Viewer" in root_resp.text
    assert viewer_resp.status_code == 200
    assert "Agentic RAG Dual-Channel SSE Viewer" in viewer_resp.text


async def test_default_lifespan_supports_v1_v2_v3_sync_and_sse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All three pipeline versions work out-of-the-box in sync and SSE modes."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()

    queries = {
        "v1": "How does Qdrant store dense vectors?",
        "v2": "How do BM25 sparse vectors and RRF combine with bge-reranker-v2-m3?",
        "v3": "What is the RTX 3060 6GB VRAM budget and how does v3 prevent infinite loops?",
    }

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for version, query in queries.items():
                sync_resp = await client.post(
                    "/api/v1/chat",
                    json={
                        "query": query,
                        "session_id": f"sync-{version}",
                        "pipeline_version": version,
                        "stream": False,
                    },
                )
                assert sync_resp.status_code == 200
                sync_body = sync_resp.json()
                assert sync_body["pipeline_version"] == version
                assert sync_body["answer"]
                assert len(sync_body["citations"]) >= 1
                assert sync_body["metrics"]["latency_ms"] is not None

                async with client.stream(
                    "POST",
                    "/api/v1/chat",
                    json={
                        "query": f"{query} (sse)",
                        "session_id": f"sse-{version}",
                        "pipeline_version": version,
                        "stream": True,
                    },
                ) as sse_resp:
                    assert sse_resp.status_code == 200
                    raw_body = (await sse_resp.aread()).decode("utf-8")

                parsed = _parse_sse_events(raw_body)
                event_names = [name for name, _ in parsed]
                assert "error" not in event_names
                assert event_names[-3:] == ["citations", "metrics", "done"]
                assert all(name == "token" for name in event_names[:-3])
                assert len(event_names[:-3]) >= 1
                assert len(parsed[-3][1]["citations"]) >= 1
                assert parsed[-1][1]["status"] == "completed"
                assert parsed[-1][1]["pipeline_version"] == version


async def test_default_lifespan_v3_semantic_cache_hit_and_singleflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Consecutive identical v3 queries hit the in-memory semantic cache and SingleFlight deduplicates."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()
    query = "Explain Ragas v0.2+ benchmark scores and v3 semantic cache threshold."

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            first_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": query,
                    "session_id": "sf-miss",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            second_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": query,
                    "session_id": "sf-hit",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )

    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
    assert first_resp.json()["metrics"]["cache_hit"] is False
    assert second_resp.json()["metrics"]["cache_hit"] is True
    assert second_resp.json()["answer"] == first_resp.json()["answer"]

    # Verify SingleFlight executes leader coroutine once under concurrent callers
    flight = SingleFlight()
    executions = 0

    async def _compute() -> str:
        nonlocal executions
        executions += 1
        await asyncio.sleep(0.02)
        return "deduped-value"

    results = await asyncio.gather(*(flight.do("same-key", _compute) for _ in range(5)))
    assert results == ["deduped-value"] * 5
    assert executions == 1


async def test_auto_mode_resolution_fallback_and_custom_injection_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auto mode degrades to demo without keys; live outage falls back; custom state is preserved."""
    _clear_cloud_env(monkeypatch, tmp_path)
    auto_settings = Settings(
        runtime_mode="auto",
        groq_api_key="gsk_your_groq_api_key_here",
        google_api_key="",
        langgraph_checkpoint_db=tmp_path / "auto.sqlite3",
        log_level=LogLevel.WARNING,
    )
    assert is_placeholder_api_key(auto_settings.groq_api_key) is True

    container = ServiceContainer(auto_settings)
    await container.initialize()
    try:
        assert container.mode == "demo"
        assert container.component_status["runtime_mode"] == "demo"
        assert container.workflow is not None
    finally:
        await container.shutdown()

    # Forced live mode with unreachable Qdrant/Redis falls back cleanly without blocking startup
    live_settings = Settings(
        runtime_mode="live",
        groq_api_key="gsk_non_empty_test_key_12345",
        qdrant_url="http://127.0.0.1:19999",
        redis_url="redis://127.0.0.1:19998",
        langgraph_checkpoint_db=tmp_path / "live_fallback.sqlite3",
        log_level=LogLevel.WARNING,
    )
    live_container = ServiceContainer(live_settings)
    await live_container.initialize()
    try:
        assert live_container.mode == "live"
        assert live_container.component_status["retriever"] == "ready (demo-fallback)"
        assert live_container.component_status["semantic_cache"] == "ready (in-memory-fallback)"
        assert live_container.workflow is not None
    finally:
        await live_container.shutdown()

    # Pre-injected custom retriever/router on app.state must not be overwritten by container
    custom_retriever = FakeRetriever([_make_chunk("doc-custom", "Custom injected chunk.")])
    custom_router = FakeRouter(answer_text="Custom router preserved [doc-custom].")
    custom_settings = Settings(
        runtime_mode="auto",
        pipeline_version=PipelineVersion.V3,
        langgraph_checkpoint_db=tmp_path / "custom.sqlite3",
        log_level=LogLevel.WARNING,
    )
    custom_app = create_app(
        settings=custom_settings,
        retriever=custom_retriever,
        router=custom_router,
        checkpoint_db=tmp_path / "custom.sqlite3",
    )
    async with lifespan(custom_app):
        assert custom_app.state.container is None
        assert custom_app.state.retriever is custom_retriever
        assert custom_app.state.router is custom_router
        assert custom_app.state.workflow is not None
        transport = httpx.ASGITransport(app=custom_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/v1/chat",
                json={"query": "Test custom injection", "stream": False},
            )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Custom router preserved [doc-custom]."


async def test_v1_v2_v3_produce_differentiated_clean_outputs_and_auto_live_router(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """v1/v2/v3 emit distinct, HTML-free answers and auto mode activates live router with keys."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()
    query = "How does this engine combine hybrid retrieval, reranking, and LangGraph?"

    answers: dict[str, str] = {}
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for version in ("v1", "v2", "v3"):
                resp = await client.post(
                    "/api/v1/chat",
                    json={
                        "query": query,
                        "session_id": f"diff-{version}",
                        "pipeline_version": version,
                        "stream": False,
                    },
                )
                assert resp.status_code == 200
                body = resp.json()
                answer = body["answer"]
                answers[version] = answer
                assert "<div" not in answer
                assert "img.shields.io" not in answer
                assert "Based on retrieved source" not in answer
                assert "Grounded synthesis from" not in answer
                assert body["metrics"]["generation_mode"] == "demo_synthesis"
                assert body["metrics"]["ttft_ms"] is not None

            # Second identical v3 request hits cache and still populates ttft_ms
            cached_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": query,
                    "session_id": "diff-v3-cached",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            assert cached_resp.status_code == 200
            assert cached_resp.json()["metrics"]["cache_hit"] is True
            assert cached_resp.json()["metrics"]["ttft_ms"] is not None

    assert answers["v1"] != answers["v2"]
    assert answers["v2"] != answers["v3"]
    assert answers["v1"] != answers["v3"]
    assert "**Hybrid Retrieval Summary**" in answers["v2"]
    assert "**Agentic Synthesis**" in answers["v3"]
    assert "**Verified Evidence & Context**" in answers["v3"]

    # RUNTIME_MODE=auto with valid cloud API key activates live LLMRouter even without Qdrant/Redis
    auto_live_settings = Settings(
        runtime_mode="auto",
        groq_api_key="gsk_valid_cloud_key_123456",
        google_api_key="",
        qdrant_url="http://127.0.0.1:19999",
        redis_url="redis://127.0.0.1:19998",
        langgraph_checkpoint_db=tmp_path / "auto_live.sqlite3",
        log_level=LogLevel.WARNING,
    )
    auto_live_container = ServiceContainer(auto_live_settings)
    await auto_live_container.initialize()
    try:
        assert auto_live_container.mode == "live"
        assert auto_live_container.component_status["router"] == "ready (live)"
        assert auto_live_container.component_status["retriever"] == "ready (demo-fallback)"
        assert (
            auto_live_container.component_status["semantic_cache"] == "ready (in-memory-fallback)"
        )
    finally:
        await auto_live_container.shutdown()


async def test_korean_and_chinese_adaptive_synthesis_and_citations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Korean and Chinese queries emit localized v1/v2/v3 answers with [doc-...] citations and zero Chinese leakage in Korean mode."""
    import re

    from app.adapters.mock.router import detect_query_language
    from app.graph.nodes import build_multilingual_system_prompt

    assert (
        detect_query_language("이 프로젝트는 RTX 3060 6GB VRAM 제한 환경에서 어떻게 동작합니까?")
        == "ko"
    )
    assert detect_query_language("本项目如何在 RTX 3060 6GB 显存限制下实现高精度混合检索?") == "zh"
    assert detect_query_language("How does Qdrant int8 hybrid retrieval work?") == "en"

    ko_sys_prompt = build_multilingual_system_prompt("한국어 질문입니다")
    assert "한국어(Korean)" in ko_sys_prompt
    assert "[doc_id]" in ko_sys_prompt

    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()

    ko_pills = [
        "이 프로젝트는 RTX 3060 6GB VRAM 제한 환경에서 Qdrant int8 양자화, BM25 희소 검색, RRF 융합 및 로컬 FP16 bge-reranker-v2-m3를 결합하여 어떻게 고정밀 하이브리드 검색을 구현합니까?",
        "v3 Agentic RAG 파이프라인은 LangGraph 상태 머신을 통해 문서 관련성 평가, 최대 2회 쿼리 재작성(Query Rewrite), 그리고 is_cacheable_response 가드레일을 통한 폴백 응답의 시맨틱 캐시 오염 방지를 어떻게 구현합니까?",
        "이 프로젝트의 Redis 시맨틱 캐시(Cosine >= 0.92), SingleFlight 동시성 잠금, 그리고 문서 수집 시 자동 캐시 무효화 메커니즘을 설명해 주세요. (팁: 이 버튼을 두 번 연속 클릭하여 ~20ms 캐시 적중 TTFT를 확인해 보세요!)",
        "이 프로젝트의 v1, v2, v3 3세대 RAG 파이프라인 간 핵심 아키텍처 차이점과 Ragas 평가 지표 및 고동시성 P95 < 1800ms 지연 시간 SLA 성능을 비교해 주세요.",
    ]
    zh_char_re = re.compile(r"[\u4e00-\u9fff]")
    ko_char_re = re.compile(r"[\uac00-\ud7af]")

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for idx, ko_query in enumerate(ko_pills):
                for version in ("v1", "v2", "v3"):
                    resp = await client.post(
                        "/api/v1/chat",
                        json={
                            "query": ko_query,
                            "session_id": f"ko-pill-{idx}-{version}",
                            "pipeline_version": version,
                            "stream": False,
                        },
                    )
                    assert resp.status_code == 200
                    body = resp.json()
                    answer = body["answer"]
                    assert ko_char_re.search(answer) is not None
                    assert zh_char_re.search(answer) is None, (
                        f"Chinese leaked in KO answer: {answer}"
                    )
                    assert "├──" not in answer
                    assert "[doc-" in answer
                    assert len(body["citations"]) >= 1
                    for cit in body["citations"]:
                        assert f"[{cit['doc_id']}]" in answer
                        assert zh_char_re.search(cit["snippet"]) is None

                    if version == "v1":
                        assert "v1 베이스라인은 고정 길이 청킹" in answer
                    elif version == "v2":
                        assert "**하이브리드 검색 요약 (Hybrid Retrieval Summary):**" in answer
                    else:
                        assert "**에이전틱 종합 분석 (Agentic Synthesis):**" in answer
                        assert (
                            "**검증된 근거 및 컨텍스트 (Verified Evidence & Context):**" in answer
                        )
                        assert "**그라운딩 검증 (Grounding Check):**" in answer

            zh_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": "本项目如何在 RTX 3060 6GB 显存限制下实现高精度混合检索?",
                    "session_id": "zh-v3-test",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            assert zh_resp.status_code == 200
            zh_answer = zh_resp.json()["answer"]
            assert "**智能体综合分析:**" in zh_answer
            assert "**已验证证据与上下文:**" in zh_answer
            assert "**事实锚定校验:**" in zh_answer
            assert "[doc-" in zh_answer
