"""
app/adapters/vertex/router.py
=============================
Feature-flagged Google Cloud Vertex AI Gemini adapter.

Zero-Cost Safety Contract
-------------------------
- Disabled by default (``USE_VERTEX_AI=false``).
- When disabled, never initializes GCP credentials, never creates a Vertex AI
  client, and never performs network I/O.
- Even when enabled, validates ``vertex_project_id`` and local GCP credentials
  before issuing any async API call; callers gracefully fall back to standard
  routes on missing credentials or runtime errors.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_PLACEHOLDER_PROJECT_MARKERS: tuple[str, ...] = (
    "your_",
    "your-",
    "placeholder",
    "changeme",
    "replace_me",
    "example",
)


class VertexAIRouterError(RuntimeError):
    """Raised when Vertex AI routing is disabled, unconfigured, or fails."""


def _is_valid_project_id(project_id: str | None) -> bool:
    if not project_id:
        return False
    cleaned = project_id.strip()
    if len(cleaned) < 4 or cleaned.startswith("<") or cleaned.endswith(">"):
        return False
    lowered = cleaned.lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_PROJECT_MARKERS)


def _locate_adc_credentials_file() -> Path | None:
    """Locate local GCP ADC JSON file without triggering GCE metadata network probes."""
    env_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env_creds:
        candidate = Path(env_creds)
        return candidate if candidate.is_file() else None

    cloudsdk_cfg = os.getenv("CLOUDSDK_CONFIG", "").strip()
    if cloudsdk_cfg:
        candidate = Path(cloudsdk_cfg) / "application_default_credentials.json"
        if candidate.is_file():
            return candidate

    appdata = os.getenv("APPDATA", "").strip()
    if appdata:
        win_candidate = Path(appdata) / "gcloud" / "application_default_credentials.json"
        if win_candidate.is_file():
            return win_candidate

    posix_candidate = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if posix_candidate.is_file():
        return posix_candidate

    return None


def has_valid_vertex_credentials(settings: Settings) -> bool:
    """Return True only when Vertex AI is explicitly enabled with valid project & ADC file."""
    if not settings.use_vertex_ai:
        return False
    if not _is_valid_project_id(settings.vertex_project_id):
        return False
    return _locate_adc_credentials_file() is not None


class VertexAIRouter:
    """Async Vertex AI Gemini adapter gated behind ``Settings.use_vertex_ai``."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        location: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = (model_name or self._settings.strong_llm_model).strip()
        self._location = (location or self._settings.vertex_location).strip() or "us-central1"
        self._client = client

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def model_name(self) -> str:
        return self._model_name

    def has_valid_credentials(self) -> bool:
        """Check feature flag, project ID, and GCP credentials (or injected test client)."""
        if not self._settings.use_vertex_ai:
            return False
        if not _is_valid_project_id(self._settings.vertex_project_id):
            return False
        if self._client is not None:
            return True
        return _locate_adc_credentials_file() is not None

    @staticmethod
    def build_system_instruction(lang: str) -> str:
        """Build trilingual-aligned Vertex AI system instructions with ``[doc_id]`` citations."""
        normalized = lang.strip().lower()
        base = (
            "Answer the user question using ONLY the provided context snippets. "
            "Cite supporting sources inline using their exact [doc_id] tag. "
            "You MUST respond strictly in the exact same language as the user question."
        )
        if normalized == "ko":
            return (
                f"{base} "
                "사용자의 질문이 한국어(Korean)이므로, 컨텍스트에 중국어나 영어가 포함되어 있더라도 "
                "모든 사실과 수치를 전문적인 한국어로 번역 및 종합하고 "
                "'**하이브리드 검색 요약:**' 형식의 구조화된 요약을 제공하십시오. "
                "중국어 원문을 복사하지 말고 인용 태그 [doc_id]만 정확히 유지하십시오."
            )
        if normalized == "zh":
            return (
                f"{base} "
                "用户提问为中文，请严格使用专业、流畅的简体中文进行结构化综合回答，"  # noqa: RUF001
                "可使用 '**混合检索摘要：**' 开头并准确保留 [doc_id] 格式的内联引用标签。"  # noqa: RUF001
            )
        return base

    def _get_or_create_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.has_valid_credentials():
            raise VertexAIRouterError(
                "Vertex AI credentials or vertex_project_id are missing or invalid."
            )

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise VertexAIRouterError("google-genai package is required for Vertex AI.") from exc

        project_id = (self._settings.vertex_project_id or "").strip()
        self._client = genai.Client(
            vertexai=True,
            project=project_id,
            location=self._location,
        )
        return self._client

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, Sequence):
            parts_out: list[str] = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None)
                if isinstance(parts, Sequence):
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_out.append(part_text.strip())
            if parts_out:
                return "\n".join(parts_out)
        return ""

    async def generate_response(self, query: str, context: list[str], lang: str) -> str:
        """Generate a grounded, trilingual-aligned response via Vertex AI Gemini."""
        if not self._settings.use_vertex_ai:
            raise VertexAIRouterError(
                "Vertex AI routing is disabled (USE_VERTEX_AI=false); refusing paid invocation."
            )
        if not self.has_valid_credentials():
            raise VertexAIRouterError(
                "Vertex AI is enabled but GCP credentials or vertex_project_id are unavailable."
            )

        client = self._get_or_create_client()
        system_instruction = self.build_system_instruction(lang)
        context_block = "\n\n---\n\n".join(item for item in context if item and item.strip())
        prompt = (
            f"Question: {query.strip()}\n\nContext:\n{context_block}"
            if context_block
            else f"Question: {query.strip()}"
        )

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
            )
            call_result = client.aio.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            response = await call_result if inspect.isawaitable(call_result) else call_result
        except VertexAIRouterError:
            raise
        except Exception as exc:
            logger.warning(
                "Vertex AI generation failed",
                model=self._model_name,
                project_id=self._settings.vertex_project_id,
                location=self._location,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise VertexAIRouterError(f"Vertex AI request failed: {exc}") from exc

        text = self._extract_response_text(response)
        if not text:
            raise VertexAIRouterError("Vertex AI returned an empty response.")
        return text


__all__: Sequence[str] = [
    "VertexAIRouter",
    "VertexAIRouterError",
    "has_valid_vertex_credentials",
]
