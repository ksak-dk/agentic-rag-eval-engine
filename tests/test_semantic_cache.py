from __future__ import annotations

import math
import re
from typing import Any

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from app.cache.semantic_cache import (
    InMemorySemanticCache,
    SemanticCache,
    VectorIndexAlgorithm,
    cosine_similarity,
    pack_float32_vector,
    unpack_float32_vector,
)
from app.core.logging import get_request_metrics, request_metrics_context
from app.retrieval.hybrid_store import DENSE_DIM


def _unit_vec_with_cosine(target_cos: float, dim: int = DENSE_DIM) -> list[float]:
    """Return a unit vector whose cosine similarity with e_0=[1, 0, ...] equals target_cos."""
    vec = [0.0] * dim
    vec[0] = target_cos
    vec[1] = math.sqrt(max(0.0, 1.0 - target_cos * target_cos))
    return vec


class DeterministicDenseEmbedder:
    """Offline 1024-dim DenseEmbedder stub mapping registered queries to unit vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[text] for text in texts]

    def get_vram_usage_mb(self) -> float:
        return 0.0


class FakeAsyncRedis:
    """In-memory async Redis 8 + RediSearch mock supporting VECTOR COSINE KNN search."""

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.ttls: dict[str, int] = {}
        self.commands: list[tuple[Any, ...]] = []

    async def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        cmd = str(args[0]).upper()

        if cmd == "FT.INFO":
            index_name = str(args[1])
            if index_name not in self.indexes:
                raise ResponseError(f"Unknown index name '{index_name}'")
            return ["index_name", index_name]

        if cmd == "FT.CREATE":
            index_name = str(args[1])
            if index_name in self.indexes:
                raise ResponseError("Index already exists")
            self.indexes[index_name] = {"args": args[2:]}
            return b"OK"

        if cmd == "FT.SEARCH":
            index_name = str(args[1])
            if index_name not in self.indexes:
                raise ResponseError(f"Unknown index name '{index_name}'")

            ft_query = str(args[2])
            ns_match = re.search(r"@namespace:\{([^}]+)\}", ft_query)
            target_ns = ns_match.group(1).replace("\\", "") if ns_match else None

            params_idx = args.index("PARAMS")
            query_vec_bytes = args[params_idx + 3]
            query_vec = unpack_float32_vector(query_vec_bytes)

            candidates: list[tuple[float, str, dict[bytes, bytes]]] = []
            for key, fields in self.hashes.items():
                entry_ns = fields[b"namespace"].decode("utf-8")
                if target_ns is not None and entry_ns != target_ns:
                    continue
                entry_vec = unpack_float32_vector(fields[b"embedding"])
                sim = cosine_similarity(query_vec, entry_vec)
                distance = max(0.0, 1.0 - sim)
                candidates.append((distance, key, fields))

            if not candidates:
                return [0]

            candidates.sort(key=lambda item: item[0])
            best_dist, best_key, best_fields = candidates[0]
            return [
                len(candidates),
                best_key.encode("utf-8"),
                [
                    b"query",
                    best_fields[b"query"],
                    b"response",
                    best_fields[b"response"],
                    b"namespace",
                    best_fields[b"namespace"],
                    b"metadata",
                    best_fields[b"metadata"],
                    b"created_at",
                    best_fields[b"created_at"],
                    b"vector_distance",
                    str(best_dist).encode("utf-8"),
                ],
            ]

        raise NotImplementedError(f"Unsupported fake command: {cmd}")

    async def hset(self, key: str, mapping: dict[str, str | bytes]) -> int:
        encoded: dict[bytes, bytes] = {}
        for field_name, val in mapping.items():
            k_bytes = field_name.encode("utf-8")
            v_bytes = val if isinstance(val, bytes) else str(val).encode("utf-8")
            encoded[k_bytes] = v_bytes
        self.hashes[key] = encoded
        return len(encoded)

    async def expire(self, key: str, seconds: int) -> bool:
        if key not in self.hashes:
            return False
        self.ttls[key] = seconds
        return True

    async def scan_iter(self, match: str = "*") -> Any:
        prefix = match.rstrip("*")
        for key in list(self.hashes):
            if key.startswith(prefix):
                yield key.encode("utf-8")

    async def delete(self, *keys: Any) -> int:
        removed = 0
        for raw_key in keys:
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            if key in self.hashes:
                del self.hashes[key]
                self.ttls.pop(key, None)
                removed += 1
        return removed

    async def aclose(self) -> None:
        self.commands.append(("ACLOSE",))


def test_pack_and_unpack_float32_roundtrip() -> None:
    vec = _unit_vec_with_cosine(0.92, dim=DENSE_DIM)
    packed = pack_float32_vector(vec)
    assert len(packed) == DENSE_DIM * 4
    unpacked = unpack_float32_vector(packed)
    assert len(unpacked) == DENSE_DIM
    assert abs(unpacked[0] - 0.92) < 1e-6


async def test_semantic_cache_set_creates_index_and_persists_vector_with_ttl() -> None:
    redis = FakeAsyncRedis()
    embedder = DeterministicDenseEmbedder({"What is RAG?": _unit_vec_with_cosine(1.0)})
    cache = SemanticCache(
        redis_client=redis,
        embedder=embedder,
        ttl_seconds=3600,
        algorithm=VectorIndexAlgorithm.HNSW,
    )

    stored_key = await cache.set(
        "What is RAG?",
        "Retrieval-Augmented Generation.",
        namespace="v3",
        metadata={"route_tier": "fast"},
    )

    assert stored_key is not None
    assert stored_key.startswith("rag:cache:v3:")
    assert redis.ttls[stored_key] == 3600
    assert len(redis.hashes[stored_key][b"embedding"]) == DENSE_DIM * 4
    assert redis.hashes[stored_key][b"response"] == b"Retrieval-Augmented Generation."

    create_cmds = [cmd for cmd in redis.commands if cmd[0] == "FT.CREATE"]
    assert len(create_cmds) == 1
    assert "HNSW" in create_cmds[0]
    assert str(DENSE_DIM) in create_cmds[0]
    assert "COSINE" in create_cmds[0]


async def test_semantic_cache_hit_and_miss_threshold_boundary() -> None:
    redis = FakeAsyncRedis()
    embedder = DeterministicDenseEmbedder(
        {
            "cached query": _unit_vec_with_cosine(1.0),
            "high sim query": _unit_vec_with_cosine(0.95),
            "exact boundary query": _unit_vec_with_cosine(0.92),
            "just below boundary query": _unit_vec_with_cosine(0.919),
            "unrelated query": _unit_vec_with_cosine(0.75),
        }
    )
    cache = SemanticCache(redis_client=redis, embedder=embedder, similarity_threshold=0.92)
    await cache.set("cached query", "Cached answer", namespace="v3", metadata={"doc": "arch.md"})

    async with request_metrics_context(request_id="req-hit-high"):
        hit_high = await cache.get("high sim query", namespace="v3")
        assert hit_high is not None
        assert hit_high.response == "Cached answer"
        assert hit_high.similarity == 0.95
        assert hit_high.metadata == {"doc": "arch.md"}
        assert get_request_metrics().cache_hit is True

    async with request_metrics_context(request_id="req-hit-boundary"):
        hit_exact = await cache.get("exact boundary query", namespace="v3")
        assert hit_exact is not None
        assert hit_exact.similarity == 0.92
        assert get_request_metrics().cache_hit is True

    async with request_metrics_context(request_id="req-miss-boundary", cache_hit=True):
        miss_below = await cache.get("just below boundary query", namespace="v3")
        assert miss_below is None
        assert get_request_metrics().cache_hit is False

    async with request_metrics_context(request_id="req-miss-low"):
        miss_low = await cache.get("unrelated query", namespace="v3")
        assert miss_low is None
        assert get_request_metrics().cache_hit is False


async def test_semantic_cache_namespace_isolation_and_invalidation() -> None:
    redis = FakeAsyncRedis()
    embedder = DeterministicDenseEmbedder({"explain rrf": _unit_vec_with_cosine(1.0)})
    cache = SemanticCache(redis_client=redis, embedder=embedder)

    await cache.set("explain rrf", "v2 answer", namespace="v2")
    await cache.set("explain rrf", "v3 answer", namespace="v3")

    async with request_metrics_context(request_id="req-ns"):
        hit_v2 = await cache.get("explain rrf", namespace="v2")
        hit_v3 = await cache.get("explain rrf", namespace="v3")
        miss_v1 = await cache.get("explain rrf", namespace="v1")

    assert hit_v2 is not None and hit_v2.response == "v2 answer"
    assert hit_v3 is not None and hit_v3.response == "v3 answer"
    assert miss_v1 is None

    removed = await cache.invalidate_namespace("v2")
    assert removed == 1
    assert await cache.get("explain rrf", namespace="v2") is None
    assert await cache.get("explain rrf", namespace="v3") is not None


async def test_semantic_cache_fail_open_on_redis_failure() -> None:
    class BrokenRedis:
        async def execute_command(self, *args: Any) -> Any:
            raise RedisConnectionError("redis connection refused")

        async def hset(self, key: str, mapping: dict[str, Any]) -> int:
            raise RedisConnectionError("redis connection refused")

        async def expire(self, key: str, seconds: int) -> bool:
            raise RedisConnectionError("redis connection refused")

    embedder = DeterministicDenseEmbedder({"test query": _unit_vec_with_cosine(1.0)})
    cache = SemanticCache(redis_client=BrokenRedis(), embedder=embedder)

    async with request_metrics_context(request_id="req-broken", cache_hit=True):
        result = await cache.get("test query")
        stored = await cache.set("test query", "answer")
        assert result is None
        assert stored is None
        assert get_request_metrics().cache_hit is False


async def test_semantic_cache_handles_blank_query_and_bad_embedder_dim() -> None:
    class BadDimEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2] for _ in texts]

        def get_vram_usage_mb(self) -> float:
            return 0.0

    redis = FakeAsyncRedis()
    cache = SemanticCache(redis_client=redis, embedder=BadDimEmbedder())

    async with request_metrics_context(request_id="req-edge"):
        assert await cache.get("   ") is None
        assert get_request_metrics().cache_hit is False
        assert await cache.set("   ", "ignored") is None
        assert await cache.get("valid query") is None
        assert get_request_metrics().cache_hit is False
        assert await cache.set("valid query", "ignored") is None


async def test_semantic_cache_blocks_fallback_and_html_polluted_writes() -> None:
    redis = FakeAsyncRedis()
    embedder = DeterministicDenseEmbedder(
        {
            "fallback query": _unit_vec_with_cosine(1.0),
            "html query": _unit_vec_with_cosine(1.0),
            "clean query": _unit_vec_with_cosine(1.0),
        }
    )
    redis_cache = SemanticCache(redis_client=redis, embedder=embedder)
    mem_cache = InMemorySemanticCache(embedder=embedder)

    for cache in (redis_cache, mem_cache):
        assert (
            await cache.set(
                "fallback query",
                "Offline fallback answer",
                namespace="v3",
                metadata={"is_fallback": True, "generation_mode": "fallback"},
            )
            is None
        )
        assert (
            await cache.set(
                "html query",
                'Based on README.md: <div align="center"> [**简体中文**](README.md) </div>',
                namespace="v3",
                metadata={"generation_mode": "demo_synthesis"},
            )
            is None
        )
        assert (
            await cache.set(
                "fallback query",
                "I could not find sufficiently relevant and verified context in the knowledge base.",
                namespace="v3",
            )
            is None
        )
        assert await cache.get("fallback query", namespace="v3") is None
        assert await cache.get("html query", namespace="v3") is None

        clean_key = await cache.set(
            "clean query",
            "**Agentic Synthesis**: Clean grounded answer [doc-readme-0].",
            namespace="v3",
            metadata={"generation_mode": "demo_synthesis", "is_fallback": False},
        )
        assert clean_key is not None
        hit = await cache.get("clean query", namespace="v3")
        assert hit is not None
        assert "Clean grounded answer" in hit.response
