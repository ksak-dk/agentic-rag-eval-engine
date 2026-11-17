from __future__ import annotations

from pathlib import Path

from scripts.demo_e2e import (
    format_metrics_line,
    main,
    parse_sse_frame,
    run_demo,
)


def test_parse_sse_frame_and_metrics_formatting() -> None:
    parsed = parse_sse_frame('event: metrics\ndata: {"ttft_ms": 12.5, "cache_hit": true}')
    assert parsed is not None
    event_name, payload = parsed
    assert event_name == "metrics"
    assert payload["cache_hit"] is True

    banner = format_metrics_line(
        {
            "ttft_ms": 12.5,
            "latency_ms": 18.25,
            "cache_hit": True,
            "vram_allocated_mb": 0.0,
            "route_tier": "fast",
            "model_name": "llama-3.1-8b-instant",
            "retry_count": 0,
        }
    )
    assert "ttft_ms=12.50ms" in banner
    assert "latency_ms=18.25ms" in banner
    assert "cache_hit=True" in banner
    assert "vram_mb=0.00" in banner


async def test_run_demo_asgi_streams_v1_v2_v3_and_cache_hit(tmp_path: Path) -> None:
    transcript_path = tmp_path / "e2e_demo_transcript.md"
    health, results, saved_path = await run_demo(
        mode="asgi",
        output_path=transcript_path,
        quiet=True,
    )

    assert health.status == "ok"
    assert saved_path == transcript_path
    assert transcript_path.exists()
    assert len(results) == 4

    v1_res, v2_res, v3_cold, v3_hit = results
    assert v1_res.scenario.pipeline_version == "v1"
    assert v2_res.scenario.pipeline_version == "v2"
    assert v3_cold.scenario.pipeline_version == "v3"
    assert v3_hit.scenario.pipeline_version == "v3"

    for res in results:
        assert res.error is None
        assert res.answer
        assert len(res.citations) >= 1
        assert res.event_counts.get("token", 0) >= 1
        assert res.event_counts.get("citations") == 1
        assert res.event_counts.get("metrics") == 1
        assert res.event_counts.get("done") == 1

    assert v3_cold.metrics["cache_hit"] is False
    assert v3_hit.metrics["cache_hit"] is True

    transcript_text = transcript_path.read_text(encoding="utf-8")
    assert "# End-to-End Streaming Demo Transcript" in transcript_text
    assert "`v1`" in transcript_text
    assert "`v2`" in transcript_text
    assert "`v3`" in transcript_text
    assert "`True`" in transcript_text


def test_demo_cli_asgi_exit_code(tmp_path: Path) -> None:
    transcript_path = tmp_path / "cli_transcript.md"
    exit_code = main(["--mode", "asgi", "--output", str(transcript_path), "--quiet"])
    assert exit_code == 0
    assert transcript_path.exists()
