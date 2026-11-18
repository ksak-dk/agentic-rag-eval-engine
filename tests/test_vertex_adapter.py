from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.adapters.mock.router import DemoLLMRouter
from app.adapters.vertex.router import (
    VertexAIRouter,
    VertexAIRouterError,
    has_valid_vertex_credentials,
)
from app.core.config import LogLevel, Settings, get_settings
from app.graph.nodes import GraphDependencies, generate_node
from app.graph.state import make_initial_state
from app.retrieval.chunker import ChunkMetadata
from app.retrieval.hybrid_store import RetrievedChunk


def _make_chunk(doc_id: str, text: str, *, score: float = 0.92, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            source="README.md",
            section_path=["Architecture", "Hybrid Retrieval"],
            chunk_index=rank,
        ),
        score=score,
        rank=rank,
    )


class _StubAsyncModels:
    def __init__(self, responses_by_lang: dict[str, str], *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = responses_by_lang
        self._fail = fail

    async def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: Any = None,
    ) -> Any:
        system_instruction = getattr(config, "system_instruction", "")
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "system_instruction": system_instruction,
            }
        )
        if self._fail:
            raise RuntimeError("Simulated Vertex AI quota error")
        if "**하이브리드 검색 요약:**" in system_instruction:
            text = self._responses["ko"]
        elif "**混合检索摘要：**" in system_instruction:  # noqa: RUF001
            text = self._responses["zh"]
        else:
            text = self._responses["en"]
        return SimpleNamespace(text=text)


class _StubGenAIClient:
    def __init__(self, responses_by_lang: dict[str, str], *, fail: bool = False) -> None:
        self.models = _StubAsyncModels(responses_by_lang, fail=fail)
        self.aio = SimpleNamespace(models=self.models)


def _clear_vertex_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "USE_VERTEX_AI",
        "VERTEX_PROJECT_ID",
        "VERTEX_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "CLOUDSDK_CONFIG",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "empty_appdata"))
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "empty_home"))
    get_settings.cache_clear()


async def test_vertex_ai_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify USE_VERTEX_AI defaults to False and never triggers Vertex AI calls."""
    _clear_vertex_env(monkeypatch, tmp_path)
    settings = Settings(
        _env_file=None,
        langgraph_checkpoint_db=tmp_path / "checkpoints.sqlite3",
        log_level=LogLevel.WARNING,
    )
    assert settings.use_vertex_ai is False
    assert settings.vertex_project_id is None
    assert has_valid_vertex_credentials(settings) is False

    stub_client = _StubGenAIClient({"en": "Should never be called [doc-1].", "ko": "", "zh": ""})
    vertex_router = VertexAIRouter(settings=settings, client=stub_client)
    assert vertex_router.has_valid_credentials() is False

    with pytest.raises(VertexAIRouterError, match="USE_VERTEX_AI=false"):
        await vertex_router.generate_response(
            "How does Qdrant hybrid search work?",
            ["[doc-1] Qdrant uses INT8 dense and BM25 sparse vectors."],
            "en",
        )
    assert stub_client.models.calls == []

    demo_router = DemoLLMRouter(settings)
    chunk = _make_chunk(
        "doc-1",
        "Qdrant uses INT8 dense vectors and BM25 sparse vectors fused with RRF.",
    )
    deps = GraphDependencies(
        retriever=None,
        router=demo_router,
        settings=settings,
        vertex_router=vertex_router,
    )
    state = make_initial_state(
        "How does Qdrant hybrid search work?",
        session_id="vertex-off-test",
        pipeline_version="v3",
    )
    state["graded_documents"] = [chunk]

    result = await generate_node(state, deps)
    assert result["generation_mode"] == "demo_synthesis"
    assert result["fallback_triggered"] is False
    assert result["citations"][0].doc_id == "doc-1"
    assert stub_client.models.calls == []


async def test_vertex_ai_feature_flag_toggle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify USE_VERTEX_AI=true routes trilingual queries via VertexAIRouter and falls back safely."""
    _clear_vertex_env(monkeypatch, tmp_path)
    monkeypatch.setenv("USE_VERTEX_AI", "true")
    monkeypatch.setenv("VERTEX_PROJECT_ID", "gcp-rag-eval-demo")
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(tmp_path / "vertex_checkpoints.sqlite3"))
    get_settings.cache_clear()

    toggled_settings = get_settings()
    assert toggled_settings.use_vertex_ai is True
    assert toggled_settings.vertex_project_id == "gcp-rag-eval-demo"

    # Without ADC file, has_valid_vertex_credentials stays False and falls back to DemoLLMRouter.
    assert has_valid_vertex_credentials(toggled_settings) is False
    demo_router = DemoLLMRouter(toggled_settings)
    chunk = _make_chunk(
        "doc-vram",
        "RTX 3060 6GB VRAM hosts FP16 BAAI/bge-m3 (~1.1GB) and bge-reranker-v2-m3 (~1.1GB).",
    )
    no_creds_deps = GraphDependencies(
        retriever=None,
        router=demo_router,
        settings=toggled_settings,
    )
    no_creds_state = make_initial_state(
        "What is the RTX 3060 VRAM limit?",
        session_id="vertex-no-creds",
        pipeline_version="v3",
    )
    no_creds_state["graded_documents"] = [chunk]
    fallback_res = await generate_node(no_creds_state, no_creds_deps)
    assert fallback_res["generation_mode"] == "demo_synthesis"
    assert fallback_res["fallback_triggered"] is False

    # With valid ADC file + mocked client, VertexAIRouter executes trilingual aligned generation.
    adc_file = tmp_path / "adc.json"
    adc_file.write_text('{"type": "authorized_user"}', encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc_file))
    assert has_valid_vertex_credentials(toggled_settings) is True

    stub_client = _StubGenAIClient(
        {
            "ko": "**하이브리드 검색 요약:** RTX 3060 6GB VRAM 환경에서 FP16 모델만 사용합니다 [doc-vram].",
            "zh": "**混合检索摘要：** 在 RTX 3060 6GB 显存限制下仅驻留 FP16 检索与重排模型 [doc-vram]。",  # noqa: RUF001
            "en": "Under the RTX 3060 6GB VRAM ceiling, only FP16 bge-m3 and reranker run locally [doc-vram].",
        }
    )
    vertex_router = VertexAIRouter(settings=toggled_settings, client=stub_client)
    deps = GraphDependencies(
        retriever=None,
        router=demo_router,
        settings=toggled_settings,
        vertex_router=vertex_router,
    )

    queries = {
        "ko": "RTX 3060 6GB VRAM 제한 환경에서 하이브리드 검색은 어떻게 동작합니까?",
        "zh": "本项目如何在 RTX 3060 6GB 显存限制下实现混合检索?",
        "en": "How does hybrid retrieval respect the RTX 3060 6GB VRAM limit?",
    }
    for lang, query in queries.items():
        state = make_initial_state(query, session_id=f"vertex-{lang}", pipeline_version="v3")
        state["graded_documents"] = [chunk]
        out = await generate_node(state, deps)
        assert out["generation_mode"] == "vertex_ai"
        assert out["fallback_triggered"] is False
        assert out["citations"][0].doc_id == "doc-vram"
        if lang == "ko":
            assert "**하이브리드 검색 요약:**" in out["answer"]
        elif lang == "zh":
            assert "**混合检索摘要：**" in out["answer"]  # noqa: RUF001
        else:
            assert "RTX 3060 6GB VRAM" in out["answer"]

    assert len(stub_client.models.calls) == 3
    assert "**하이브리드 검색 요약:**" in stub_client.models.calls[0]["system_instruction"]
    assert "**混合检索摘要：**" in stub_client.models.calls[1]["system_instruction"]  # noqa: RUF001

    # When VertexAIRouter errors at runtime, generate_node gracefully falls back to DemoLLMRouter.
    failing_router = VertexAIRouter(
        settings=toggled_settings,
        client=_StubGenAIClient({}, fail=True),
    )
    failing_deps = GraphDependencies(
        retriever=None,
        router=demo_router,
        settings=toggled_settings,
        vertex_router=failing_router,
    )
    fail_state = make_initial_state(
        queries["en"], session_id="vertex-err-fb", pipeline_version="v3"
    )
    fail_state["graded_documents"] = [chunk]
    recovered = await generate_node(fail_state, failing_deps)
    assert recovered["generation_mode"] == "demo_synthesis"
    assert recovered["citations"][0].doc_id == "doc-vram"

    get_settings.cache_clear()
