"""
app/core/config.py
==================
Centralized application settings via Pydantic V2 BaseSettings.

All values are read from environment variables (or .env file).
No magic defaults that could silently misconfigure production.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineVersion(StrEnum):
    """Controls which RAG pipeline variant is active.

    v1 — Fixed-length chunking + Qdrant Dense only + single-step generation.
    v2 — Semantic chunking + Dense & Sparse (BM25) + RRF + local CUDA reranker.
    v3 — LangGraph agentic loop + citation tracing + Redis cache + model router.
    """

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Production settings for agentic-rag-eval-engine.

    Hardware contract (RTX 3060 / 6 GB VRAM):
      - Only bge-m3 (~1.1 GB) + bge-reranker-v2-m3 (~1.1 GB) run on CUDA.
      - Total VRAM budget: 2.3 GB. Never load any local LLM.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Cloud LLM API Keys
    # -------------------------------------------------------------------------
    groq_api_key: str = Field(..., description="Groq API key for fast-route LLM calls.")
    google_api_key: str = Field(..., description="Google AI Studio key for strong-route LLM calls.")

    # -------------------------------------------------------------------------
    # Langfuse (optional observability — requires profile: observability)
    # -------------------------------------------------------------------------
    langfuse_public_key: str = Field(
        default="pk-lf-local-dev",
        description="Langfuse public key. Set to empty string to disable tracing.",
    )
    langfuse_secret_key: str = Field(
        default="sk-lf-local-dev",
        description="Langfuse secret key.",
    )
    langfuse_host: str = Field(
        default="http://localhost:3000",
        description="Langfuse server URL.",
    )

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant REST endpoint.")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL.")

    # -------------------------------------------------------------------------
    # Pipeline Control
    # -------------------------------------------------------------------------
    pipeline_version: PipelineVersion = Field(
        default=PipelineVersion.V3,
        description="Active pipeline variant: v1 | v2 | v3.",
    )

    mock_llm_stream: bool = Field(
        default=False,
        description=(
            "When True: mock LLM responses (300ms TTFT, simulated stream). "
            "Used in CI and high-concurrency load tests to avoid API quota burn."
        ),
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Structured log verbosity.")

    # -------------------------------------------------------------------------
    # FastAPI
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="Uvicorn bind port.")

    # -------------------------------------------------------------------------
    # Qdrant Collection
    # -------------------------------------------------------------------------
    qdrant_collection_name: str = Field(
        default="rag_documents",
        description="Name of the Qdrant collection for dual-vector storage.",
    )

    # -------------------------------------------------------------------------
    # Retrieval Hyperparameters
    # -------------------------------------------------------------------------
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Final Top-K context chunks sent to the LLM after reranking.",
    )
    retrieval_top_n: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Candidate pool size fetched from Qdrant before local reranking.",
    )

    # -------------------------------------------------------------------------
    # Redis Semantic Cache
    # -------------------------------------------------------------------------
    cache_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic cache hit.",
    )
    cache_ttl_seconds: int = Field(
        default=86400,
        ge=60,
        description="TTL for cached responses in seconds (default: 24 h).",
    )

    # -------------------------------------------------------------------------
    # LangGraph
    # -------------------------------------------------------------------------
    langgraph_checkpoint_db: Path = Field(
        default=Path("./data/langgraph_checkpoints.sqlite3"),
        description="Path to AsyncSqliteSaver checkpoint database.",
    )
    max_retries: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Hard cap on grader→rewriter retry loops. Prevents infinite LangGraph cycles.",
    )

    # -------------------------------------------------------------------------
    # Model Identifiers (pinned versions — update here only)
    # -------------------------------------------------------------------------
    embed_model_name: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model ID for dense embedding. Loaded via sentence-transformers.",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace model ID for cross-encoder reranking. FP16 on CUDA.",
    )
    fast_llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Fast-route LLM model name (Groq). Used for routing, rewriting, simple QA.",
    )
    strong_llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Strong-route LLM model (Google AI Studio). Used for generation + eval judge.",
    )
    fast_llm_fallback_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Fallback fast-route model when Groq rate-limits.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("langgraph_checkpoint_db", mode="before")
    @classmethod
    def coerce_checkpoint_path(cls, v: str | Path) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def validate_top_n_gt_top_k(self) -> Settings:
        if self.retrieval_top_n <= self.retrieval_top_k:
            raise ValueError(
                f"retrieval_top_n ({self.retrieval_top_n}) must be greater than "
                f"retrieval_top_k ({self.retrieval_top_k})."
            )
        return self

    @model_validator(mode="after")
    def ensure_checkpoint_dir_exists(self) -> Settings:
        self.langgraph_checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        return self

    # -------------------------------------------------------------------------
    # Convenience properties
    # -------------------------------------------------------------------------
    @property
    def langfuse_enabled(self) -> bool:
        """True when both Langfuse keys are non-empty (non-default) and host is reachable."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def groq_base_url(self) -> str:
        return "https://api.groq.com/openai/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Uses lru_cache so the .env file is read exactly once per process.
    Call `get_settings.cache_clear()` in tests to reload from a patched environment.
    """
    return Settings()
