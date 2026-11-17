from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.config import get_settings
from app.main import create_app, lifespan
from app.retrieval import ingest as ingest_cli


def _clear_cloud_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "RUNTIME_MODE",
        "MOCK_LLM_STREAM",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(tmp_path / "ingest_checkpoints.sqlite3"))
    get_settings.cache_clear()


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


async def test_live_document_ingest_and_immediate_retrieval_qa(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingesting a new Markdown doc makes it immediately searchable via sync and SSE /chat."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()

    hyperion_doc = (
        "# Project-Hyperion-9921 Specification\n\n"
        "Project-Hyperion-9921 operates with telemetry heartbeat interval 42ms and "
        "quantum-shard-factor 7781 in isolated edge clusters.\n\n"
        "## Failover Policy\n\n"
        "When Project-Hyperion-9921 exceeds thermal threshold 68C, traffic shifts to node-delta-9."
    )

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            stats_before = (await client.get("/api/v1/documents/stats")).json()
            initial_chunks = stats_before["total_chunks_in_store"]

            ingest_resp = await client.post(
                "/api/v1/documents/ingest",
                json={
                    "documents": [
                        {
                            "source": "docs/hyperion_spec.md",
                            "content": hyperion_doc,
                            "metadata": {"owner": "edge-team"},
                        }
                    ],
                    "pipeline_version": "v3",
                    "invalidate_cache": True,
                },
            )
            assert ingest_resp.status_code == 200
            ingest_body = ingest_resp.json()
            assert ingest_body["status"] == "ok"
            assert ingest_body["documents_processed"] == 1
            assert ingest_body["documents_skipped"] == 0
            assert ingest_body["total_chunks_added"] >= 1
            assert (
                ingest_body["total_chunks_in_store"]
                == initial_chunks + ingest_body["total_chunks_added"]
            )
            assert ingest_body["items"][0]["skipped_duplicate"] is False
            assert len(ingest_body["items"][0]["doc_sha256"]) == 64

            # Sync query against newly ingested content
            chat_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": "What telemetry heartbeat interval and quantum-shard-factor does Project-Hyperion-9921 use?",
                    "session_id": "ingest-sync-1",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            assert chat_resp.status_code == 200
            chat_body = chat_resp.json()
            assert "42ms" in chat_body["answer"] or "7781" in chat_body["answer"]
            assert any(c["source"] == "docs/hyperion_spec.md" for c in chat_body["citations"])

            # Streaming SSE query against newly ingested failover section
            async with client.stream(
                "POST",
                "/api/v1/chat",
                json={
                    "query": "Where does Project-Hyperion-9921 shift traffic when thermal threshold 68C is exceeded?",
                    "session_id": "ingest-sse-1",
                    "pipeline_version": "v3",
                    "stream": True,
                },
            ) as sse_resp:
                assert sse_resp.status_code == 200
                raw_sse = (await sse_resp.aread()).decode("utf-8")

            events = _parse_sse_events(raw_sse)
            assert [name for name, _ in events][-3:] == ["citations", "metrics", "done"]
            streamed_answer = "".join(
                payload["token"] for name, payload in events if name == "token"
            )
            assert "node-delta-9" in streamed_answer or "Project-Hyperion-9921" in streamed_answer
            citations_payload = events[-3][1]["citations"]
            assert any(c["source"] == "docs/hyperion_spec.md" for c in citations_payload)


async def test_ingest_sha256_idempotent_deduplication_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Re-ingesting identical document content skips chunk creation without store bloat."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()

    payload = {
        "documents": [
            {
                "source": "docs/idempotent_note.md",
                "content": "# Idempotent Note\n\nCodename Aurora-551 uses SHA-256 deduplication.",
            }
        ],
        "pipeline_version": "v3",
    }

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            empty_resp = await client.post("/api/v1/documents/ingest", json={"documents": []})
            assert empty_resp.status_code == 422

            first = (await client.post("/api/v1/documents/ingest", json=payload)).json()
            assert first["documents_processed"] == 1
            assert first["documents_skipped"] == 0
            assert first["total_chunks_added"] >= 1
            count_after_first = first["total_chunks_in_store"]

            second = (await client.post("/api/v1/documents/ingest", json=payload)).json()
            assert second["documents_processed"] == 0
            assert second["documents_skipped"] == 1
            assert second["total_chunks_added"] == 0
            assert second["total_chunks_in_store"] == count_after_first
            assert second["items"][0]["skipped_duplicate"] is True
            assert second["items"][0]["doc_sha256"] == first["items"][0]["doc_sha256"]

            stats = (await client.get("/api/v1/documents/stats")).json()
            assert stats["total_chunks_in_store"] == count_after_first
            assert stats["ingested_documents_count"] == 1
            assert stats["runtime_mode"] == "demo"


async def test_ingest_invalidates_semantic_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ingesting new documents invalidates cached v3 answers so stale entries are purged."""
    _clear_cloud_env(monkeypatch, tmp_path)
    app = create_app()
    query = "How does the v3 pipeline use Redis semantic cache?"

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            miss_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": query,
                    "session_id": "cache-1",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            assert miss_resp.json()["metrics"]["cache_hit"] is False

            hit_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": query,
                    "session_id": "cache-2",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            assert hit_resp.json()["metrics"]["cache_hit"] is True

            ingest_resp = await client.post(
                "/api/v1/documents/ingest",
                json={
                    "documents": [
                        {
                            "source": "docs/cache_update.md",
                            "content": "# Cache Update\n\nCache policy updated to version 2026-Q4.",
                        }
                    ],
                    "pipeline_version": "v3",
                    "invalidate_cache": True,
                },
            )
            assert ingest_resp.status_code == 200
            assert ingest_resp.json()["cache_entries_invalidated"] >= 1

            post_ingest_resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": query,
                    "session_id": "cache-3",
                    "pipeline_version": "v3",
                    "stream": False,
                },
            )
            assert post_ingest_resp.json()["metrics"]["cache_hit"] is False


def test_ingest_cli_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI ingestion executes cleanly against README.md and prints JSON summary."""
    exit_code = ingest_cli.main(["--path", "README.md", "--pipeline", "v3"])
    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["status"] == "ok"
    assert parsed["documents_processed"] == 1
    assert parsed["total_chunks_added"] >= 1
    assert parsed["items"][0]["source"] == "README.md"
