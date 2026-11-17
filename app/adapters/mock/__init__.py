"""
app/adapters/mock/__init__.py
=============================
Public re-exports for offline, demo, and heuristic simulation adapters.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.adapters.mock.eval_heuristics import (
    BENCHMARK_CORPUS,
    DEFAULT_LOAD_TEST_QUERIES,
    LoadTestRetriever,
    LoadTestRouter,
    LoadTestSemanticCache,
    build_mock_load_test_app,
    build_offline_dataset,
    build_offline_sample_dicts,
    clamp_score,
    compute_percentile,
    heuristic_ragas_scorer,
    offline_simulated_runner,
    select_multi_hop_candidates,
    select_single_hop_candidates,
    tokenize_lexical,
    truncate_words,
)
from app.adapters.mock.retriever import DEFAULT_FALLBACK_CORPUS, InMemoryHybridRetriever
from app.adapters.mock.router import DemoLLMRouter

__all__: Sequence[str] = [
    "BENCHMARK_CORPUS",
    "DEFAULT_FALLBACK_CORPUS",
    "DEFAULT_LOAD_TEST_QUERIES",
    "DemoLLMRouter",
    "InMemoryHybridRetriever",
    "LoadTestRetriever",
    "LoadTestRouter",
    "LoadTestSemanticCache",
    "build_mock_load_test_app",
    "build_offline_dataset",
    "build_offline_sample_dicts",
    "clamp_score",
    "compute_percentile",
    "heuristic_ragas_scorer",
    "offline_simulated_runner",
    "select_multi_hop_candidates",
    "select_single_hop_candidates",
    "tokenize_lexical",
    "truncate_words",
]
