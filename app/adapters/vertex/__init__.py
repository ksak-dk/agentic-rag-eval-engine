"""
app/adapters/vertex/__init__.py
===============================
Feature-flagged Google Cloud Vertex AI adapter package.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.adapters.vertex.router import (
    VertexAIRouter,
    VertexAIRouterError,
    has_valid_vertex_credentials,
)

__all__: Sequence[str] = [
    "VertexAIRouter",
    "VertexAIRouterError",
    "has_valid_vertex_credentials",
]
